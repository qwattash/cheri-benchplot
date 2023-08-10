import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import networkx.readwrite.json_graph as nxj
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so

from ..core.analysis import AnalysisTask
from ..core.artefact import (AnalysisFileTarget, DataFrameTarget, LocalFileTarget, Target)
from ..core.config import Config, ConfigPath, InstanceKernelABI
from ..core.elf.dwarf import DWARFManager, GraphConversionVisitor
from ..core.plot import PlotTarget, PlotTask, new_figure
from ..core.task import BenchmarkTask, DataGenTask, dependency, output
from ..ext import pychericap, pydwarf
from .model import (ImpreciseSubobjectInfoModel, ImpreciseSubobjectInfoModelRecord, ImpreciseSubobjectLayoutModel)


@dataclass
class PathMatchSpec(Config):
    path: ConfigPath
    match: str | None = None


@dataclass
class ExtractImpreciseSubobjectConfig(Config):
    """
    Configure the imprecise sub-object extractor.
    """
    #: List of paths that are used to extract DWARF information.
    #: Note that multiple paths will be considered as belonging to the same "object",
    #: if multiple objects are being compared, we should use parameterization.
    #: Note that relative paths will be interpreted as relative paths into a cheribuild rootfs.
    dwarf_data_sources: list[PathMatchSpec]


class ValueTarget(Target):
    def __init__(self, task, output_id: str | None = None):
        self._value = None
        # Borg state initialization occurs here
        super().__init__(task, output_id)

    def assign(self, v: any):
        self._value = v

    def get(self) -> any:
        return self._value


class StructLayoutLoader(BenchmarkTask):
    """
    Custom task to load the structure layout graph from GML
    """
    task_namespace = "subobject"
    task_name = "imprecise-layout-loader"

    def __init__(self, benchmark, target):
        assert target.is_file()
        self.target = target
        # Borg state initialization occurs here
        super().__init__(benchmark)

    @output
    def graph(self):
        return ValueTarget(self)

    def run(self):
        g = GraphConversionVisitor.load(self.target.path)
        self.graph.assign(g)


class ExtractImpreciseSubobject(DataGenTask):
    """
    Extract CheriBSD DWARF type information for aggregate data types.
    """
    public = True
    task_namespace = "subobject"
    task_name = "extract-imprecise"
    task_config_class = ExtractImpreciseSubobjectConfig

    def __init__(self, benchmark, script, task_config):
        super().__init__(benchmark, script, task_config)

        instance_config = self.benchmark.config.instance
        if instance_config.cheri_target.is_riscv():
            # XXX support/detect riscv32
            self.cap_format = pychericap.CompressedCap128
        elif instance_config.cheri_target.is_morello():
            self.cap_format = pychericap.CompressedCap128m
        else:
            self.logger.error("DWARF TypeInfo extraction unsupported for %s", instance_config.cheri_target)
            raise RuntimeError(f"Unsupported instance target {instance_config.cheri_target}")

    @output
    def imprecise_layouts(self):
        return LocalFileTarget(self, ext="gml.gz", loader=StructLayoutLoader)

    def _check_imprecise(self, offset: int, size: int) -> tuple[int, int] | None:
        """
        Check if a specific member of a structure is representable.

        If it is not representable, return a tuple (new_base, new_top)
        """
        subobject_cap = self.cap_format.make_max_bounds_cap(offset)
        subobject_cap.setbounds(size)

        if subobject_cap.base() < offset or subobject_cap.top() > offset + size:
            return (subobject_cap.base(), subobject_cap.top())
        return None

    def _find_imprecise_for(self, g: nx.DiGraph, n: "NodeID"):
        """
        """
        g.nodes[n]["has_imprecise"] = False
        alias_group_id = 0
        imprecise = set()
        for parent, child in nx.dfs_edges(g, source=n):
            # Determine if this node is imprecise
            result = self._check_imprecise(child.member_offset, g.nodes[child]["size"])
            if result:
                g.nodes[child]["alias_group_id"] = alias_group_id
                alias_group_id += 1
                g.nodes[child]["alias_aligned_base"] = result[0]
                g.nodes[child]["alias_aligned_top"] = result[1]
                imprecise.add(child)
                g.nodes[n]["has_imprecise"] = True

        if len(imprecise) == 0:
            # Bail, nothing else to do
            return

        imprecise_table = pd.Series(map(lambda i: g.nodes[i]["alias_group_id"], imprecise),
                                    index=pd.IntervalIndex.from_arrays(map(lambda i: g.nodes[i]["alias_aligned_base"],
                                                                           imprecise),
                                                                       map(lambda i: g.nodes[i]["alias_aligned_top"],
                                                                           imprecise),
                                                                       closed="left"))
        # Now need to determine the alias groups
        for parent, child in nx.dfs_edges(g, source=n):
            # XXX reject unions?
            aliasing = imprecise_table.index.overlaps(
                pd.Interval(child.member_offset, child.member_offset + g.nodes[child]["size"], closed="left"))
            if not aliasing.any():
                continue
            groups = set(imprecise_table[aliasing].unique())
            # Ignore aliasing groups of descendants and ancestors of this node.
            # This represents the fact that the subobject for a whole structure does
            # not really alias any of its members.
            remove_groups = set()
            remove_groups.add(g.nodes[child].get("alias_group_id", None))
            for d in nx.descendants(g, child):
                remove_groups.add(g.nodes[d].get("alias_group_id", None))
            for d in nx.ancestors(g, child):
                remove_groups.add(g.nodes[d].get("alias_group_id", None))
            groups = groups.difference(remove_groups)
            if groups:
                # Can not use set because it is harder to serialize to GML cleanly
                g.nodes[child]["alias_groups"] = list(map(int, groups))
        self.logger.debug("Found %d imprecise members for %s", len(imprecise), n)

    def _find_imprecise(self, g: nx.DiGraph):
        """
        Generate alias groups for imprecise sub-object members.
        This generates a graph representing data structure layouts in the dwarf information.
        The alias_group_id attribute contains an integer identifier for the set of fields aliasing
        with the one with the attribute.
        The alias_aligned_base and alias_aligned_top contain the base and top offsets after CHERI
        representability rounding.
        The alias_groups attribute contains a list of all the group IDs that a field is aliasing with.
        """
        roots = g.graph["roots"]
        if len(roots) == 0:
            self.logger.debug("No data structures for %s", dw.path)
            return None
        for n in roots:
            self._find_imprecise_for(g, n)
        return g

    def _resolve_paths(self, target: PathMatchSpec) -> Iterator[Path]:
        """
        Resolve a path matcher to a list of paths.
        """
        if target.match is None:
            yield from [target.path]
        matcher = re.compile(target.match)

        def match_path(p: Path):
            r = p.relative_to(target.path)
            return matcher.match(str(r))

        yield from filter(match_path, target.path.rglob("*"))

    def run(self):
        iconf = self.benchmark.config.instance
        layout_graph = nx.DiGraph()

        for target in self.config.dwarf_data_sources:
            for item in self._resolve_paths(target):
                self.logger.debug("Inspect subobject bounds from %s", item)
                if not item.is_file():
                    self.logger.error("File %s is not a regular file, skipping", item)
                    continue
                # XXX may want to make the manager use a more predictable target name
                # XXX the pointer size detection should go away
                dw = self.benchmark.dwarf.register_object(item, item, arch_pointer_size=iconf.user_pointer_size)
                info = dw.load_type_info()
                dw.build_struct_layout_graph(info, layout_graph)

        self._find_imprecise(layout_graph)
        layout_graph.graph["dataset_id"] = str(self.benchmark.uuid)
        layout_graph.graph["dataset_gid"] = str(self.benchmark.g_uuid)
        # Now we can dump the layout
        GraphConversionVisitor.dump(layout_graph, self.imprecise_layouts.path)


class ImpreciseSubobjectBoundsUnion(AnalysisTask):
    """
    Merge all imprecise subobject bounds warnings.

    This merges the datasets by platform g_uuid, so it is possible
    to observe the difference between the behaviour for Morello and
    RISC-V variants.
    """
    task_namespace = "subobject"
    task_name = "imprecise-subobject-union"

    @dependency
    def layout_data(self):
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(ExtractImpreciseSubobject)
            yield task.imprecise_layouts.get_loader()

    def run(self):
        # Note that we want to remove duplicates that show the same structure label across
        # different files if they have the same type, offset and size
        data = [d.graph.get() for d in self.layout_data]
        self.all_layouts.assign(data)

    @output
    def all_layouts(self):
        return ValueTarget(self)


class ImpreciseSetboundsPlot(PlotTask):
    """
    Produce a plot showing the difference in size and alignment caused by CHERI
    representability rounding on each sub-object.

    Each different platform (gid) is rendered separately.
    """
    public = True
    task_namespace = "subobject"
    task_name = "imprecise-subobject-plot"

    @dependency
    def data(self):
        return ImpreciseSubobjectBoundsUnion(self.session, self.analysis_config)

    def _collect_imprecise_members(self, layouts: list[nx.DiGraph]):
        """
        Construct a dataframe containing all the imprecise structure members
        """
        imprecise = []
        for g in layouts:
            gid = uuid.UUID(g.graph["dataset_gid"])
            records = []
            for struct_root in g.graph["roots"]:
                if "has_imprecise" not in g.nodes[struct_root] or not g.nodes[struct_root]["has_imprecise"]:
                    continue
                for desc in nx.descendants(g, struct_root):
                    if "alias_group_id" not in g.nodes[desc]:
                        continue
                    # This is an imprecise member, remember it
                    r = ImpreciseSubobjectInfoModelRecord(file=desc.file,
                                                          line=desc.line,
                                                          base_name=desc.base_name,
                                                          member_name=desc.member_name,
                                                          member_offset=desc.member_offset,
                                                          member_size=g.nodes[desc]["size"],
                                                          member_aligned_base=g.nodes[desc]["alias_aligned_base"],
                                                          member_aligned_top=g.nodes[desc]["alias_aligned_top"])
                    records.append(r)
            df = pd.DataFrame.from_records(records, columns=ImpreciseSubobjectInfoModelRecord._fields)
            if len(df) == 0:
                continue
            df["dataset_gid"] = gid
            imprecise.append(df)
        if imprecise:
            df = pd.concat(imprecise, axis=0)
        else:
            df = pd.DataFrame([], columns=("dataset_gid", ) + ImpreciseSubobjectInfoModelRecord._fields)
        df = df.set_index(["dataset_gid", "file", "line", "base_name", "member_name", "member_offset"])
        return df[~df.index.duplicated(keep="first")]

    def run_plot(self):
        df = self._collect_imprecise_members(self.data.all_layouts.get())

        # Normalize base and size with respect to the "requested" base offset
        # and size
        member_offset = df.index.get_level_values("member_offset")
        df["aligned_size"] = df["member_aligned_top"] - df["member_aligned_base"]
        df["base_rounding"] = member_offset - df["member_aligned_base"]
        df["top_rounding"] = df["member_aligned_top"] - (member_offset + df["member_size"])

        assert (df["base_rounding"] >= 0).all()
        assert (df["top_rounding"] >= 0).all()

        sns.set_theme()

        with new_figure(self.imprecise_fields_plot.paths(), figsize=(10, 50)) as fig:
            ax = fig.subplots()
            show_df = df.reset_index().melt(id_vars=df.index.names,
                                            value_vars=["base_rounding", "top_rounding"],
                                            var_name="source_of_imprecision")
            show_df["legend"] = (show_df["source_of_imprecision"] + " " +
                                 show_df["dataset_gid"].map(self.g_uuid_to_label))
            show_df["label"] = show_df["base_name"] + "::" + show_df["member_name"]
            (so.Plot(show_df, y="label", x="value", color="legend").on(ax).add(so.Bar(),
                                                                               so.Dodge(by=["dataset_gid"]),
                                                                               so.Stack(),
                                                                               dataset_gid=show_df["dataset_gid"],
                                                                               orient="y").plot())
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Capability imprecision (Bytes)")
            ax.set_ylabel("Sub-object field")
            plt.setp(ax.get_yticklabels(), ha="right", va="center", fontsize="xx-small")

    @output
    def imprecise_fields_plot(self):
        return PlotTarget(self)


class ImpreciseSetboundsLayouts(AnalysisTask):
    """
    Produce a D3-js html document that allows to browse the structure layouts with imprecision
    and inspect the data members that are affected by imprecision.
    """
    task_namespace = "subobject"
    task_name = "imprecise-subobject-layouts"

    @dependency
    def data(self):
        return ImpreciseSubobjectBoundsUnion(self.session, self.analysis_config)

    @output
    def html_view(self):
        return AnalysisFileTarget(self, ext="html")

    def run(self):
        layouts_df = self.data.all_layouts.get()

        for group in layouts_df.groupby(["base_name"]):
            json_layouts = self._build_hierarchical_layout(layouts_df)
            self._generate_layout_data(group)
