import re
import typing as ty
from dataclasses import dataclass, field
from pathlib import Path

import cxxfilt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from ..core.analysis import AnalysisTask, DatasetAnalysisTask
from ..core.artefact import (DataFrameTarget, DataRunAnalysisFileTarget, LocalFileTarget)
from ..core.config import Config, ConfigPath, validate_path_exists
from ..core.pandas_util import generalized_xs
from ..core.plot import DatasetPlotTask, PlotTarget, new_figure
from ..core.task import DataGenTask, dependency, output
from ..core.util import gzopen
from .gui import C18NTraceInspector
from .model import C18NDomainTransitionTraceModel


@dataclass
class TraceImportConfig(Config):
    # #: install rootfs where the binaries recorded in the kdump are found
    # rootfs: ConfigPath = None
    #: kdump file to import
    kdump: ConfigPath = None


@dataclass
class PrettifyTraceConfig(Config):
    #: Compress the trace output
    compress_output: bool = True
    #: Do not generate the text file, only useful for GUI output
    skip_file_output: bool = False


@dataclass
class TransitionGraphConfig(Config):
    #: Drop redundant transitions
    drop_redundant: bool = True


class C18NKtraceImport(DataGenTask):
    """
    Import a ktrace file for a c18n program run.

    One trace must be available for every benchmark instance, this allows to
    import parameterized data points for later aggregation.
    """
    public = True
    task_namespace = "c18n"
    task_name = "ktrace-import"
    task_config_class = TraceImportConfig

    def run(self):
        self.logger.debug("Ingest c18n domain transition kdump: %s", self.config.kdump)

        with gzopen(self.config.kdump, "r") as kdump:
            df = self._parse(kdump)
        df.to_csv(self.data_file.path)

    @output
    def data_file(self):
        return LocalFileTarget(self, prefix="trace", ext="csv.gz", model=C18NDomainTransitionTraceModel)

    def _parse(self, kdump: ty.IO) -> pd.DataFrame:
        line_matcher = re.compile(
            r"(?P<pid>[0-9]+) (?P<thread>[0-9]+)\s+(?P<binary>[\w.-]+).*RTLD: c18n: (?P<op>enter|leave) "
            r"\[?(?P<compartment>[\w./<>+-]+)\]?(?: from \[?(?P<parent_compartment>[\w./<>+-]+)\]?)? "
            r"at \[(?P<symbol_number>[0-9]+)\] (?P<symbol>[\w<>-]+)\s*\(?(?P<address>[\dxabcdef]*)\)?")
        data = []
        failed = 0
        for line in kdump:
            line = line.strip()
            if not line:
                continue
            m = line_matcher.match(line)
            if not m:
                failed += 1
                if self.session.user_config.verbose:
                    self.logger.warn("Failed to process %s", line)
                continue
            data.append(m.groupdict())
        if failed:
            self.logger.warn("Failed to process %d records from %s", failed, self.config.kdump)
        self.logger.info("Imported %d records from kdump %s", len(data), self.config.kdump)

        df = pd.DataFrame(data)
        # Normalize hex values
        df["address"] = df["address"].map(lambda v: int(v, base=16) if v else None)
        df.index.name = "sequence"
        df["trace_source"] = self.config.kdump.name
        return df


class C18NTransitionsSummary(DatasetPlotTask):
    """
    Generate a simple histogram of the top 20 most frequent domain transitions.
    """
    public = True
    task_namespace = "c18n"
    task_name = "plot-top-transitions"

    @dependency
    def data(self):
        generator = self.benchmark.find_exec_task(C18NKtraceImport)
        return generator.data_file.get_loader()

    @output
    def hist_plot(self):
        return PlotTarget(self)

    def run_plot(self):
        df = self.data.df.get()

        def demangler(v):
            if v == "<unknown>":
                return v
            try:
                return cxxfilt.demangle(v)
            except cxxfilt.InvalidName:
                return "<demangler error>"
            assert False, "Not Reached"

        df["symbol"] = df["symbol"].map(demangler)

        # Count transitions
        df = df.groupby(["compartment", "symbol", "op"]).size()
        # Prepare to pivot
        df = df.to_frame(name="count").reset_index()
        # Pivot enter/leave out
        show_df = df.reset_index().pivot(index=["compartment", "symbol"], columns="op", values="count")

        show_df["total"] = show_df["enter"] + show_df["leave"]
        show_df["% enter"] = 100 * show_df["enter"] / show_df["enter"].sum()
        show_df["% leave"] = 100 * show_df["leave"] / show_df["leave"].sum()
        show_df = show_df.sort_values(by="total", ascending=False).iloc[:20]
        # Build printable labels
        show_df["transition"] = show_df.index.map(lambda i: i[0] + ":" + i[1])

        sns.set_theme()

        with new_figure(self.hist_plot.paths()) as fig:
            ax_l, ax_r = fig.subplots(1, 2, sharey=True)
            # Absolute number of transitions on the left
            tmp = show_df.copy()
            # Reduce the scale
            tmp["enter"] /= 1000
            tmp["leave"] /= 1000
            tmp.reset_index().plot(x="transition", y=["enter", "leave"], kind="barh", ax=ax_l)
            ax_l.tick_params(axis="y", labelsize="xx-small")
            ax_l.set_xlabel("# of transitions (k)")
            # Relative number of transitions on the right
            show_df.reset_index().plot(x="transition", y=["% enter", "% leave"], kind="barh", ax=ax_r)
            ax_r.tick_params(axis="y", labelsize="xx-small")
            ax_r.set_xlabel("% of transitions")


class C18NTransitionGraph(DatasetPlotTask):
    """
    Generate a graph with relationship between compartments.
    """
    public = True
    task_namespace = "c18n"
    task_name = "plot-compartment-graph"
    task_config_class = TransitionGraphConfig

    @dependency
    def data(self):
        generator = self.benchmark.find_exec_task(C18NKtraceImport)
        return generator.data_file.get_loader()

    @output
    def reachability_plot(self):
        return PlotTarget(self)

    def run_plot(self):
        df = self.data.df.get()
        graph = nx.DiGraph()

        names = df.index.unique("binary")
        if len(names) > 1:
            self.logger.error("Multiple names for the traced binary?")
            raise RuntimeError("Unexpected Input")
        binary_name = names[0]

        # Simplify the compartment names for display
        def simplify_name(n):
            comp = Path(n).name.split(".")
            comp[0] = comp[0][3:] if comp[0].startswith("lib") else comp[0]
            name = ".".join(comp[:2])
            return name

        df["compartment"] = df["compartment"].map(simplify_name)
        df["parent_compartment"] = df["parent_compartment"].fillna("").map(simplify_name)

        # Need to determine the source compartment for each "enter" entry
        # while we are scanning, add entries to the graph
        self.logger.info("Computing transition graph")

        # Build the graph representation from a dataframe containing records
        thread_label_index = df.index.names.index("thread")
        assert thread_label_index != -1

        self.logger.info("Prepare transition graph")
        max_edge_weights = {tid: 0 for tid in df.index.unique("thread")}

        def fill_graph(row):
            thread = row.name[thread_label_index]
            if row["op"] != "enter":
                return
            edge_id = (row["parent_compartment"], row["compartment"])
            # Avoid creating self-edges
            if edge_id[0] == edge_id[1] and self.config.drop_redundant:
                return

            if not edge_id in graph.edges:
                graph.add_edge(*edge_id, weight=1, thread=thread)
            else:
                e = graph.edges[edge_id]
                e["weight"] += 1
            max_edge_weights[thread] = max(max_edge_weights[thread], graph.edges[edge_id]["weight"])

        df.apply(fill_graph, axis=1)
        del df

        sns.set_theme()
        cmap = sns.color_palette("crest", as_cmap=True)

        for u, v, edge in graph.edges.data():
            norm_weight = (edge["weight"] - 1) / (max_edge_weights[edge["thread"]] - 1)
            color = cmap(norm_weight, bytes=True)
            edge["color"] = "#{:2x}{:2x}{:2x}{:2x}".format(*(color))

        self.logger.info("Found edges:%d nodes:%d", len(graph.edges), len(graph.nodes))
        dot = nx.nx_pydot.to_pydot(graph)
        dot.set_prog("neato")
        dot.set_splines(True)
        for path in self.reachability_plot.paths():
            ext = path.suffix[1:]
            writer = getattr(dot, f"write_{ext}")
            writer(path)


class C18NAnnotateTrace(DatasetAnalysisTask):
    """
    Annotate a single trace to make it human-readable

    This will generate an output text file with indentation
    and names demangled.
    """
    task_namespace = "c18n"
    task_name = "prettify-one-trace"
    task_config_class = PrettifyTraceConfig

    @dependency
    def data(self):
        task = self.benchmark.find_exec_task(C18NKtraceImport)
        return task.data_file.get_loader()

    @output
    def annotated(self):
        ext = "txt"
        if self.config.compress_output:
            ext += ".gz"
        return DataRunAnalysisFileTarget(self, ext=ext)

    @output
    def trace_df(self):
        return DataFrameTarget(self, None)

    @output
    def html_output(self):
        return DataRunAnalysisFileTarget(self, prefix="webview", ext="html")

    def run(self):
        df = self.data.df.get()
        srcs = df.index.unique("trace_source")
        if len(srcs) > 1:
            self.logger.error("Multiple trace sources detected for a single datarun?")
            raise RuntimeError("Unexpected input")
        self.logger.info("Annotate trace %s", srcs[0])

        # Compute indentation depth
        df["depth"] = 1
        df["depth"] = df["depth"].mask(df["op"] == "leave", -1)
        df["depth"] = df["depth"].cumsum() - 1
        # Increase depth of leave so that they are at the same indentation level
        # of the ENTER we are leaving
        df.loc[df["op"] == "leave", "depth"] += 1

        # It is now possible to find the parent sequence for each element
        df.reset_index("sequence", inplace=True)
        df["parent_sequence"] = df[["sequence", "depth"]].groupby("depth").shift(1).fillna(-1)

        def demangler(v):
            if v == "<unknown>":
                return v
            try:
                return cxxfilt.demangle(v)
            except cxxfilt.InvalidName:
                return v + " <DEMANGLE ERR>"
            assert False, "Not Reached"

        df["symbol"] = df["symbol"].map(demangler)
        df["address"] = df["address"].map(lambda v: f"0x{int(v):x}", na_action="ignore")
        df["address"] = df["address"].fillna("")

        if self.config.compress_output:
            openfn = gzopen
        else:
            openfn = open

        if not self.config.skip_file_output:
            space = "  "
            dump = df["depth"].map(lambda d: space * d)
            dump += (df["address"] + " " + df["op"].str.upper()).str.strip()
            dump += (" " + df["compartment"] + ":" + df["symbol"] + "\n")

            self.logger.info("Writeback trace")
            with openfn(self.annotated.path, "wt") as outfile:
                outfile.writelines(dump)

        self.trace_df.assign(df)


class C18NTraceAnnotation(AnalysisTask, C18NTraceInspector):
    """
    Schedule trace annotation for all traces
    """
    public = True
    task_namespace = "c18n"
    task_name = "prettify-trace"
    task_config_class = PrettifyTraceConfig

    @dependency
    def deps(self):
        for b in self.session.all_benchmarks():
            yield C18NAnnotateTrace(b, self.analysis_config, self.config)

    def run(self):
        pass
