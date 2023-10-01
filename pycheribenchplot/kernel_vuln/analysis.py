import re
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import AutoLocator, AutoMinorLocator

from ..core.analysis import AnalysisTask
from ..core.artefact import (AnalysisFileTarget, DataFrameTarget, LocalFileTarget)
from ..core.config import Config, ConfigPath, validate_path_exists
from ..core.plot import PlotTarget, PlotTask, new_figure
from ..core.task import SessionDataGenTask, dependency, output
from .model import CheriBSDAdvisories, CheriBSDUnmitigated, History, Marker


@dataclass
class CheriBSDKernelVulnConfig(Config):
    #: Vulnerability classification input csv.
    #: This is required.
    classification_file: ConfigPath = field(metadata={"validate": validate_path_exists})
    unmitigated_file: ConfigPath = field(metadata={"validate": validate_path_exists})


class CheriBSDKernelVuln(SessionDataGenTask):
    """
    This task interprets and freezes a table from the CSV kernel vulnerability classification.
    The input is compiled separately but this is used to ingest it for plotting in a way that
    is better controlled and automated than google sheets or excel.
    """
    public = True
    task_namespace = "kernel-vuln"
    task_name = "cheribsd-sa-classification"
    task_config_class = CheriBSDKernelVulnConfig

    @output
    def advisories(self):
        return LocalFileTarget(self, ext="csv", model=CheriBSDAdvisories)

    @output
    def history(self):
        return LocalFileTarget(self, prefix="history", ext="csv", model=History)

    @output
    def unmitigated(self):
        return LocalFileTarget(self, prefix="unmitigated", ext="csv", model=CheriBSDUnmitigated)

    def _normalize_input_colname(self, column_name: str) -> str:
        """
        Cleanup a column name by removing all weird charaters and make lowercase
        """
        cleaned = re.sub(r"[()?]", "", column_name)
        cleaned = re.sub(r"[\n\s/-]", "_", cleaned)
        cleaned = re.sub(r"_+", "_", cleaned)
        return cleaned.lower()

    def _import_advisories(self):
        self.logger.debug("Reading vulnerability classification from %s", self.config.classification_file)
        input_df = pd.read_csv(self.config.classification_file, header=[0, 1])
        # Cleanup column index because google sheets generates a quite mangled one
        # We collapse everything on a single level and normalize the names.
        normalized_names = []
        for col_tuple in input_df.columns:
            if col_tuple[1].startswith("Unnamed"):
                normalized_names.append(self._normalize_input_colname(col_tuple[0]))
            else:
                normalized_names.append("cheri_" + self._normalize_input_colname(col_tuple[1]))
        input_df.columns = pd.Index(normalized_names)
        # The history column contains "date: event" pairs, we extract them as a separate
        # dataframe as they are not really related to the advisories
        history_df = input_df[["history"]].dropna()
        input_df.drop("history", axis=1)
        input_df.set_index(["date", "advisory"], inplace=True)
        # Normalize markers
        marker_cols = input_df.columns.difference(
            ["date", "advisory", "local_remote", "patch_commits", "notes", "history"])
        input_df[marker_cols] = input_df[marker_cols].applymap(lambda v: Marker.from_input(v))
        df = CheriBSDAdvisories.to_schema(self.session).validate(input_df)
        df.to_csv(self.advisories.path)

        history_df["date"] = history_df["history"].str.split(":").str.get(0).astype("datetime64[ns]")
        history_df["occurrence"] = history_df["history"].str.split(":").str.get(1)
        history_df = history_df.set_index("date").sort_index()
        history_df.drop("history", axis=1)
        df = History.to_schema(self.session).validate(history_df)
        df.to_csv(self.history.path)

    def _import_unmitigated(self):
        self.logger.debug("Reading unmitigated classification from %s", self.config.unmitigated_file)
        input_df = pd.read_csv(self.config.unmitigated_file, header=[0, 1])
        # Normalize column names
        normalized_names = []
        last_group = None
        for col_tuple in input_df.columns:
            if col_tuple[1].startswith("Unnamed"):
                normalized_names.append(self._normalize_input_colname(col_tuple[0]))
            else:
                if not col_tuple[0].startswith("Unnamed"):
                    last_group = self._normalize_input_colname(col_tuple[0])
                colname = self._normalize_input_colname(col_tuple[1])
                normalized_names.append(f"{last_group}_{colname}")
        input_df.columns = pd.Index(normalized_names)
        input_df.set_index(["date", "unmitigated_advisory"], inplace=True)
        df = CheriBSDUnmitigated.to_schema(self.session).validate(input_df)
        df.to_csv(self.unmitigated.path)

    def run(self):
        self._import_advisories()
        self._import_unmitigated()


class CheriBSDKernelVulnLoad(AnalysisTask):
    """
    Load task that grabs the cleaned-up SA classification file from the datagen task.
    """
    task_namespace = "kernel-vuln"
    task_name = "load-sa"

    @dependency
    def advisories_df(self):
        task = self.session.find_exec_task(CheriBSDKernelVuln)
        return task.advisories.get_loader()

    @dependency
    def history_df(self):
        task = self.session.find_exec_task(CheriBSDKernelVuln)
        return task.history.get_loader()

    @dependency
    def unmitigated_df(self):
        task = self.session.find_exec_task(CheriBSDKernelVuln)
        return task.unmitigated.get_loader()

    def run(self):
        # XXX the datagen task should be made session-scoped so that we don't have to
        # pass a random benchmark there.
        return
        # task = CheriBSDKernelVuln(self.session.benchmark_matrix.iloc[0, 0], None)
        # advisories = task.output_map["advisories"]
        # df = pd.read_csv(advisories.path, index_col=["date", "advisory"])
        # self.df.assign(df)

        # history = task.output_map["history"]
        # df = pd.read_csv(history.path, index_col=["date"])
        # self.history_df.assign(df)

        # unmitigated = task.output_map["unmitigated"]
        # df = pd.read_csv(unmitigated.path, index_col=["date", "unmitigated_advisory"])
        # self.unmitigated_df.assign(df)


class CheriBSDAdvisoriesHistory(PlotTask):
    """
    Generate plot of the distribution of advisories over time.
    """
    public = True
    task_namespace = "kernel-vuln"
    task_name = "plot-sa-timeline"

    @dependency
    def sa_load(self):
        return CheriBSDKernelVulnLoad(self.session, self.analysis_config)

    @output
    def timeline(self):
        return PlotTarget(self, prefix="timeline")

    @output
    def mitigated_timeline(self):
        return PlotTarget(self, prefix="mit-timeline")

    @output
    def mitigated_rel_timeline(self):
        return PlotTarget(self, prefix="mit-rel-timeline")

    @output
    def mitigated_rel_mem_timeline(self):
        return PlotTarget(self, prefix="mit-rel-mem-timeline")

    def run_plot(self):
        df = self.sa_load.advisories_df.df.get()
        history = self.sa_load.history_df.df.get()

        df["year"] = df.index.get_level_values("date").year
        df["advisory"] = df.index.get_level_values("advisory")
        # Some intermediate columns
        df["is_memory_safety"] = (df["not_memory_safety_bug"] == Marker.EMPTY)
        df["is_memory_safety_spatial"] = (df["is_memory_safety"] & ~df["uaf_uma_kmalloc"])
        df["is_mitigated_spatial"] = (df["cheri_does_not_help"] == Marker.EMPTY)
        df["is_mitigated_temporal"] = (df["cheri_does_not_help"] == Marker.TEMPORAL)
        df["is_mitigated"] = df["is_mitigated_spatial"] | df["is_mitigated_temporal"]

        aggregated = df.groupby("year").agg({
            "advisory": "size",
            "is_memory_safety": "sum",
            "is_memory_safety_spatial": "sum",
            "is_mitigated_spatial": "sum",
            "is_mitigated": "sum",
        })

        def do_melt(indf, display_columns):
            tmpdf = indf[display_columns.keys()].reset_index().melt("year", var_name="group", value_name="advisories")
            tmpdf["group"] = tmpdf["group"].map(display_columns)
            return tmpdf

        def add_history_axis(main_ax):
            # Add the twin top X axis with history markers
            ax_t = ax.twiny()
            ax_t.set_xlim(*ax.get_xlim())
            for dt in history.index.year:
                ax_t.axvline(dt, linestyle="-.", linewidth=0.2, color="gray")
            ax_t.set_xticks(history.index.year)
            ax_t.set_xticklabels(history["occurrence"], rotation=-25, ha="right", size=8)

        # timeline just shows the existing trend of advisories, without any mitigation data
        display_columns = {
            "advisory": "Total advisories",
            "is_memory_safety": "Memory safety advisory",
            "is_memory_safety_spatial": "Spatial memory safety advisory"
        }
        plot_df = do_melt(aggregated, display_columns)
        with new_figure(self.timeline.paths()) as fig:
            ax = fig.subplots()
            ax.grid(axis="y", linestyle="--")
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            sns.lineplot(ax=ax,
                         data=plot_df,
                         x="year",
                         y="advisories",
                         hue="group",
                         palette="pastel",
                         markers=True,
                         style="group")
            ax.set_ylabel("Number of advisories")
            add_history_axis(ax)

        # rel-timeline just shows the existing trend of advisories, without any mitigation data,
        # with the share of memory-safety related advisories vs the total number of advisories.
        # display_columns = {"advisory": "Total advisories", "is_memory_safety": "Memory safety advisory"}
        # plot_df = do_melt(aggregated, display_columns)
        # with new_figure(self.output_map["timeline"].path) as fig:
        #     ax = fig.subplots()
        #     ax.grid(axis="y", linestyle="--")
        #     ax.xaxis.set_minor_locator(AutoMinorLocator())
        #     sns.lineplot(ax=ax, data=plot_df, x="year", y="advisories", hue="group", palette="pastel",
        #                  markers=True, style="group")
        #     ax.set_ylabel("Number of advisories")
        #     add_history_axis(ax)

        # mitigated-timeline and mitigated-rel-timeline produce plots that show the mitigations
        # effect w.r.t. all advisories.
        display_columns = {
            "is_mitigated_spatial": "Affected by pure-capability kernel",
            "is_mitigated": "Possibly mitigated with temporal safety"
        }
        plot_df = aggregated.copy()
        plot_df["is_mitigated_spatial"] = plot_df["is_mitigated_spatial"] * 100 / plot_df["advisory"]
        plot_df["is_mitigated"] = plot_df["is_mitigated"] * 100 / plot_df["advisory"]
        plot_df = do_melt(plot_df, display_columns)
        with new_figure(self.mitigated_rel_timeline.paths()) as fig:
            ax = fig.subplots()
            ax.grid(axis="y", linestyle="--")
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            sns.lineplot(ax=ax,
                         data=plot_df,
                         x="year",
                         y="advisories",
                         hue="group",
                         palette="pastel",
                         markers=True,
                         style="group")
            ax.set_ylabel("% of all advisories")

        plot_df = do_melt(aggregated, display_columns)
        with new_figure(self.mitigated_timeline.paths()) as fig:
            ax = fig.subplots()
            ax.grid(axis="y", linestyle="--")
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            sns.lineplot(ax=ax,
                         data=plot_df,
                         x="year",
                         y="advisories",
                         hue="group",
                         palette="pastel",
                         markers=True,
                         style="group")
            ax.set_ylabel("Number of advisories")
            add_history_axis(ax)

        # mitigated-rel-mem-timeline produces a timeline of mitigations w.r.t. the total amount of
        # memory safety related advisories
        plot_df = aggregated.copy()
        plot_df["is_mitigated_spatial"] = plot_df["is_mitigated_spatial"] * 100 / plot_df["is_memory_safety"]
        plot_df["is_mitigated"] = plot_df["is_mitigated"] * 100 / plot_df["is_memory_safety"]
        plot_df = do_melt(plot_df, display_columns)
        with new_figure(self.mitigated_rel_mem_timeline.paths()) as fig:
            ax = fig.subplots()
            ax.grid(axis="y", linestyle="--")
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            sns.lineplot(ax=ax,
                         data=plot_df,
                         x="year",
                         y="advisories",
                         hue="group",
                         palette="pastel",
                         markers=True,
                         style="group")
            ax.set_ylabel("% of memory safety related advisories")
            add_history_axis(ax)

        # mitigated-distribution produces a plot with the different shares of


class CheriBSDAdvisoriesTables(AnalysisTask):
    """
    Generate summary tables CSV files from the main vulnerability analysis table.
    """
    public = True
    task_namespace = "kernel-vuln"
    task_name = "summary-tables"

    @dependency
    def sa_load(self):
        return CheriBSDKernelVulnLoad(self.session, self.analysis_config)

    @output
    def mitigation_summary(self):
        return AnalysisFileTarget(self, prefix="advisories-summary", ext="csv")

    @output
    def unmitigated_summary(self):
        return AnalysisFileTarget(self, prefix="unmitigated-summary", ext="csv")

    def _advisories_summary(self):
        df = self.sa_load.advisories_df.df.get()

        all_memory = (df["not_memory_safety_bug"] == Marker.EMPTY).sum()
        all_mitigated = ((df["cheri_does_not_help"] != Marker.YES) &
                         (df["cheri_does_not_help"] != Marker.UNKNOWN)).sum()

        mitigation_df = pd.DataFrame({
            "Description": [
                "Mitigated by bounds checking",
                "Mitigated by sub-object bounds checking",
                "Mitigated by pointer provenance and integrity",
                "Total mitigated",
                "Possibly mitigated by temporal safety",
                "Total mitigated with temporal safety",
            ]
        })
        mitigation_df["Number of advisories"] = [
            (df["cheri_bounds_checking"] == Marker.YES).sum(), (df["cheri_subobject"] == Marker.YES).sum(),
            ((df["cheri_ptr_integrity"] == Marker.YES) & (df["cheri_ptr_provenance"] == Marker.YES)).sum(),
            (df["cheri_does_not_help"] == Marker.EMPTY).sum(), (df["cheri_temporal"] == Marker.TEMPORAL).sum(),
            ((df["cheri_does_not_help"] == Marker.EMPTY) | (df["cheri_does_not_help"] == Marker.TEMPORAL)).sum()
        ]
        # Format % columns as string percentages
        mitigation_df["% Of memory safety advisories"] = (mitigation_df["Number of advisories"] * 100 /
                                                          all_memory).round(decimals=2)
        mitigation_df["% Of advisories that can be mitigated"] = (mitigation_df["Number of advisories"] * 100 /
                                                                  all_mitigated).round(decimals=2)
        mitigation_df.to_csv(self.mitigation_summary.path, index=False, header=True)

    def _unmitigated_summary(self):
        advisory_df = self.sa_load.advisories_df.df.get()
        df = self.sa_load.unmitigated_df.df.get()

        all_unmitigated = ((advisory_df["not_memory_safety_bug"] == Marker.EMPTY) &
                           (advisory_df["cheri_does_not_help"] == Marker.YES)).sum()

        out_df = pd.DataFrame({
            "Description": [
                "Missing initialization of padding bytes", "Direct VM subsystem operations", "Stack temporal safety",
                "Other"
            ]
        })
        out_df["Number of advisories"] = [(df["reason_padding_initialization"] == Marker.YES).sum(),
                                          (df["reason_direct_vm_subsystem_access"] == Marker.YES).sum(),
                                          (df["reason_stack_use_after_free"] == Marker.YES).sum(),
                                          (df["reason_other"] == Marker.YES).sum()]
        out_df["% Of unmitigated advisories"] = (out_df["Number of advisories"] * 100 /
                                                 all_unmitigated).round(decimals=2)
        out_df.to_csv(self.unmitigated_summary.path, index=False, header=True)

    def run(self):
        self._advisories_summary()
        self._unmitigated_summary()


class CheriBSDAdvisoriesTables(PlotTask):
    """
    Generate a plot showing the mitigation spectrum.
    On the X axis, we have CHERI features, on the Y axis the % of mitigated vulnerabilities
    """
    public = True
    task_namespace = "kernel-vuln"
    task_name = "feature-mitigation-cdf"

    @dependency
    def sa_load(self):
        return CheriBSDKernelVulnLoad(self.session, self.analysis_config)

    @output
    def cdf(self):
        return PlotTarget(self)

    def run_plot(self):
        df = self.sa_load.advisories_df.df.get()

        all_memory = (df["not_memory_safety_bug"] == Marker.EMPTY).sum()

        mitigated_by_spatial_safety = (((df["cheri_bounds_checking"] == Marker.YES) |
                                        (df["cheri_ptr_integrity"] == Marker.YES) |
                                        (df["cheri_ptr_provenance"] == Marker.YES)) &
                                       (df["cheri_subobject"] != Marker.YES)).sum()
        mitigated_by_subobject = (df["cheri_subobject"] == Marker.YES).sum()
        mitigated_by_temporal = (df["cheri_temporal"] == Marker.TEMPORAL).sum()

        mitigation_df = pd.DataFrame({
            "CHERI mitigation": [
                "Spatial safety", "Spatial safety + sub-object bounds", "Spatial + temporal safety",
                "Compartmentalisation"
            ],
            "Number mitigated": [
                mitigated_by_spatial_safety, mitigated_by_spatial_safety + mitigated_by_subobject,
                mitigated_by_spatial_safety + mitigated_by_subobject + mitigated_by_temporal, None
            ]
        })
        mitigation_df["Percent mitigated"] = 100 * mitigation_df["Number mitigated"] / all_memory

        sns.set_theme()
        with new_figure(self.cdf.paths()) as fig:
            ax = fig.subplots()
            sns.pointplot(mitigation_df, x="CHERI mitigation", y="Percent mitigated", ax=ax)
            ax.grid(axis="x", linestyle="--", color="gray", linewidth=.5)
            ax.grid(axis="y", linestyle="--", color="gray", linewidth=.5)
            ax.set_ylim(0, 100)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize="x-small")
