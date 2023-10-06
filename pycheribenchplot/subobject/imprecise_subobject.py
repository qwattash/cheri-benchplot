import json
import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so

from ..core.analysis import AnalysisTask
from ..core.artefact import (AnalysisFileTarget, DataFrameTarget, HTMLTemplateTarget, LocalFileTarget, Target)
from ..core.config import Config, ConfigPath, InstanceKernelABI
from ..core.dwarf import DWARFManager, StructLayoutGraph
from ..core.plot import PlotTarget, PlotTask, new_facet, new_figure
from ..core.task import BenchmarkTask, DataGenTask, dependency, output
from ..ext import pychericap, pydwarf
from .model import (ImpreciseSubobjectInfoModel, ImpreciseSubobjectInfoModelRecord, ImpreciseSubobjectLayoutModel)


def legend_on_top(fig, ax=None, **kwargs):
    """
    Fixup the legend to appear at the top of the axes
    """
    if not fig.legends:
        return
    # Hack to remove the legend as we can not easily move it
    # Not sure why seaborn puts the legend in the figure here
    legend = fig.legends.pop()
    if ax is None:
        owner = fig
    else:
        owner = ax
    kwargs.setdefault("loc", "center")
    owner.legend(legend.legend_handles,
                 map(lambda t: t.get_text(), legend.texts),
                 bbox_to_anchor=(0, 1.02),
                 ncols=4,
                 **kwargs)


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
        g = StructLayoutGraph.load(self.benchmark, self.target.path)
        self.graph.assign(g)


class ExtractImpreciseSubobject(DataGenTask):
    """
    Extract CheriBSD DWARF type information for aggregate data types.

    This will produce a graph containing structure layouts as trees.
    XXX Document tree structure and node attributes.

    Example configuration:

    .. code-block:: json

        {
            "instance_config": {
                "instances": [{
                    "kernel": "KERNEL-CONFIG-WITH-SUBOBJ-STATS",
                    "cheri_target": "riscv64-purecap",
                    "kernelabi": "purecap"
                }, {
                    "kernel": "KERNEL-CONFIG-WITH-SUBOBJ-STATS",
                    "cheri_target": "morello-purecap",
                    "kernelabi": "purecap"
                }]
            },
            "benchmark_config": [{
                "name": "example",
                "generators": [{ "handler": "kernel-static.cheribsd-subobject-stats" }]
            }]
        }

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

    def _check_imprecise(self, offset: float, size: float) -> tuple[int, int] | None:
        """
        Check if a specific member of a structure is representable.

        If it is not representable, return a tuple (new_base, new_top).
        Note that offset and size may be floating point numbers if the sub-object
        involves bit-fields.
        """
        offset = int(offset)
        size = int(np.ceil(size))
        subobject_cap = self.cap_format.make_max_bounds_cap(offset)
        is_exact = subobject_cap.setbounds(size)

        if is_exact:
            assert subobject_cap.base() == offset and subobject_cap.top() == offset + size
            return None
        assert subobject_cap.base() < offset or subobject_cap.top() >= offset + size
        return (subobject_cap.base(), subobject_cap.top())

    def _check_flex_array(self, g: nx.DiGraph, n: StructLayoutGraph.NodeID) -> bool:
        """
        Check whether the given structure contains a flexible array member at the end.

        Note that this uses an heuristic to skip structures with zero-length padding
        arrays at the end that are generated for CheriBSD syscall arguments.
        """
        # Special case for kernel system call structures that may have 0 padding members.
        # We know that there are no flex arrays in there
        if n.file.endswith("sys/sys/sysproto.h") or re.search(r"sys/compat/.*_proto\.h$", n.file):
            self.logger.warning("Note: %s @ %s:%d ignores flex array matching due to CheriBSD heuristic",
                                g.nodes[n]["type_name"], n.file, n.line)
            return False
        top_level_members = sorted(g.successors(n), key=lambda m: g.nodes[m]["offset"])
        if not top_level_members:
            return False

        last = top_level_members[-1]
        if g.nodes[last]["flags"] & pydwarf.TypeInfoFlags.kIsArray == 0 or g.nodes[last]["nitems"] != 0:
            return False

        # This is likely a flex array.
        self.logger.debug("Found flex array structure %s", n)
        return True

    def _find_imprecise_for(self, g: nx.DiGraph, n: StructLayoutGraph.NodeID):
        """
        Given a NodeID in the graph representing a top-level struct description,
        find whether the structure contains any imprecise members.

        If the structure contains imprecise sub-objects, set the "has_imprecise" attribute.
        The node will additionally define the following attributes:
        - alias_group_id: an integer (unique within the top-level structure members) that identifies
        the group of struct members that alias with the owner of that ID.
        - alias_aligned_base: the representable base for the sub-object capability
        - alias_aligned_top: the representable top for the sub-object capability
        - alias_pointer_members: the sub-object capability would alias some pointers.

        If the structure contains a flexible array at the end, mark it with "has_flexarray".
        Note that it is impossible to have nested members with flexible arrays.

        Each node that aliases with an imprecise sub-object capability for a member has the
        following attributes:
        - alias_groups: A list/sequence containing the alias_group_id values of all the
        sub-objects that alias with the struct member corresponding to the node.
        """
        g.nodes[n]["has_imprecise"] = False
        alias_group_id = 0
        imprecise = set()
        for parent, child in nx.dfs_edges(g, source=n):
            # Determine if this node is imprecise
            result = self._check_imprecise(g.nodes[child]["offset"], child.member_size)
            if result:
                g.nodes[child]["alias_group_id"] = alias_group_id
                alias_group_id += 1
                g.nodes[child]["alias_aligned_base"] = result[0]
                g.nodes[child]["alias_aligned_top"] = result[1]
                g.nodes[child]["alias_pointer_members"] = False
                imprecise.add(child)
                g.nodes[n]["has_imprecise"] = True

        if self._check_flex_array(g, n):
            g.nodes[n]["has_flexarray"] = True

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
            offset = g.nodes[child]["offset"]
            aliasing = imprecise_table.index.overlaps(pd.Interval(offset, offset + child.member_size, closed="left"))
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
                # Can not use set because it is harder to serialize cleanly when dumping the graph
                g.nodes[child]["alias_groups"] = list(map(int, groups))
            # If this node is a pointer, mark it as such in the "owner" of the alias group
            if g.nodes[child].get("flags", 0) & pydwarf.TypeInfoFlags.kIsPtr:
                for subobject_node in imprecise:
                    g.nodes[subobject_node]["alias_pointer_members"] |= (g.nodes[subobject_node]["alias_group_id"]
                                                                         in groups)

        self.logger.debug("Found %d imprecise members for %s", len(imprecise), n)

    def _find_imprecise(self, g: StructLayoutGraph):
        """
        Generate alias groups for imprecise sub-object members.
        This generates a graph representing data structure layouts in the dwarf information.
        The alias_group_id attribute contains an integer identifier for the set of fields aliasing
        with the one with the attribute.
        The alias_aligned_base and alias_aligned_top contain the base and top offsets after CHERI
        representability rounding.
        The alias_groups attribute contains a list of all the group IDs that a field is aliasing with.
        """
        roots = g.layouts.graph["roots"]
        if len(roots) == 0:
            self.logger.debug("No data structures for %s", dw.path)
            return None
        for n in roots:
            self._find_imprecise_for(g.layouts, n)
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
        layout_container = StructLayoutGraph(self.benchmark)

        for target in self.config.dwarf_data_sources:
            for item in self._resolve_paths(target):
                self.logger.debug("Inspect subobject bounds from %s", item)
                if not item.is_file():
                    self.logger.error("File %s is not a regular file, skipping", item)
                    continue
                # XXX may want to make the manager use a more predictable target name
                dw = self.benchmark.dwarf.register_object(item, item)
                dw.build_struct_layout_graph(layout_container)

        self._find_imprecise(layout_container)
        layout_container.layouts.graph["dataset_id"] = str(self.benchmark.uuid)
        layout_container.layouts.graph["dataset_gid"] = str(self.benchmark.g_uuid)
        # Now we can dump the layout
        layout_container.dump(self.imprecise_layouts.path)


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


class ImpreciseMembersPlotBase(PlotTask):
    """
    Produce a plot showing the difference in size and alignment caused by CHERI
    representability rounding on each sub-object.

    Each different platform (gid) is rendered separately.
    """
    task_namespace = "subobject"

    @dependency
    def data(self):
        return ImpreciseSubobjectBoundsUnion(self.session, self.analysis_config)

    def _collect_imprecise_members(self, struct_graphs: list[StructLayoutGraph]):
        """
        Construct a dataframe containing all the imprecise structure members
        """
        imprecise = []
        for sg in struct_graphs:
            g = sg.layouts
            gid = uuid.UUID(g.graph["dataset_gid"])
            records = []
            for struct_root in g.graph["roots"]:
                if not g.nodes[struct_root].get("has_imprecise", False):
                    continue
                for desc in nx.descendants(g, struct_root):
                    if "alias_group_id" not in g.nodes[desc]:
                        continue
                    # This is an imprecise member, remember it
                    r = ImpreciseSubobjectInfoModelRecord(file=desc.file,
                                                          line=desc.line,
                                                          base_name=desc.member.split(".")[0],
                                                          member_name=desc.member,
                                                          member_offset=g.nodes[desc]["offset"],
                                                          member_size=desc.member_size,
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

    def _prepare_dataset(self):
        """
        Prepare the imprecise subobjects dataset from the loaded graph layouts
        """
        df = self._collect_imprecise_members(self.data.all_layouts.get())
        # Normalize base and size with respect to the "requested" base offset
        # and size
        member_offset = df.index.get_level_values("member_offset")
        df["aligned_size"] = df["member_aligned_top"] - df["member_aligned_base"]
        df["base_rounding"] = member_offset - df["member_aligned_base"]
        df["top_rounding"] = df["member_aligned_top"] - (member_offset + df["member_size"])

        assert (df["base_rounding"] >= 0).all()
        assert (df["top_rounding"] >= 0).all()
        return df

    def _prepare_dataset_filter_common(self):
        """
        Same as _prepare_dataset(), but only retain structures that are built
        for all dataset_gid (E.g. common to both morello and riscv).
        """
        df = self._prepare_dataset()
        struct_graphs = self.data.all_layouts.get()

        # Filter the dataframe by structures that exist in all layout dumps
        # To do so, find the intersection between all structure identifiers, then
        # select only the nodes in df that belong to these structures.
        # This means joining on [file, line, base_name]
        common = set(struct_graphs[0].layouts.graph["roots"])
        for sg in struct_graphs[1:]:
            g = sg.layouts
            common = common.intersection(g.graph["roots"])
        common_df = pd.DataFrame.from_records(map(lambda n: (n.file, n.line, n.member.split(".")[0]), common),
                                              columns=["file", "line", "base_name"])
        return df.join(common_df.set_index(["file", "line", "base_name"]), how="inner")


class AllImpreciseMembersPlot(ImpreciseMembersPlotBase):
    """
    Variant of imprecise subobject bounds plot that shows the amount of base and top
    aliasing for each imprecise structure member.
    """
    public = True
    task_name = "imprecise-subobject-plot"

    def _plot_imprecise(self, df):
        """
        Produce the imprecise fields vs base and top rounding.
        """
        show_df = df.reset_index().melt(id_vars=df.index.names + ["member_size"],
                                        value_vars=["base_rounding", "top_rounding"],
                                        var_name="source_of_imprecision")
        show_df["legend"] = (show_df["source_of_imprecision"] + " " + show_df["dataset_gid"].map(self.g_uuid_to_label))
        show_df["label"] = show_df["base_name"] + "::" + show_df["member_name"]

        with new_figure(self.imprecise_fields_plot.paths(), figsize=(10, 50)) as fig:
            ax = fig.subplots()
            (so.Plot(show_df, y="label", x="value", color="legend").on(ax).add(
                so.Bar(),
                so.Dodge(by=["dataset_gid", "source_of_imprecision"]),
                # so.Stack(),
                dataset_gid=show_df["dataset_gid"],
                source_of_imprecision=show_df["source_of_imprecision"],
                orient="y").plot())

            legend_on_top(fig, ax)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Capability imprecision (Bytes)")
            ax.set_ylabel("Sub-object field")
            plt.setp(ax.get_yticklabels(), ha="right", va="center", fontsize="xx-small")

    def _plot_size(self, df):
        """
        Plot the imprecise_field vs size.
        """
        show_df = df.reset_index()
        show_df["legend"] = show_df["dataset_gid"].map(self.g_uuid_to_label)
        # Prepare labels, strip leading _ to pacify matplotlib warnings
        show_df["label"] = show_df["base_name"].str.strip("_") + "::" + show_df["member_name"]
        show_df = show_df.sort_values(["member_size", "dataset_gid"])

        # Basic field vs size plot (horizontal Y)
        with new_figure(self.imprecise_fields_size_plot.paths(), figsize=(10, 50)) as fig:
            ax = fig.subplots()

            (so.Plot(show_df, y="label", x="member_size", color="legend").on(ax).add(so.Bar(),
                                                                                     so.Dodge(by=["dataset_gid"]),
                                                                                     dataset_gid=show_df["dataset_gid"],
                                                                                     orient="y").plot())

            legend_on_top(fig, ax)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Sub-object size (Bytes)")
            ax.set_ylabel("Sub-object field")
            plt.setp(ax.get_yticklabels(), ha="right", va="center", fontsize="xx-small")

        # Distribution of sizes as an histogram, without field names
        with new_figure(self.imprecise_fields_size_hist.paths()) as fig:
            ax = fig.subplots()
            # min_size = int(np.log2(show_df["member_size"].min()))
            # max_size = int(np.log2(show_df["member_size"].max())) + 1
            # hist_bins = [2**x for x in range(min_size, max_size)]

            sns.histplot(show_df, x="member_size", hue="legend", log_scale=2, ax=ax, element="step")
            ax.set_xlabel("Sub-object size (Bytes)")
            ax.set_ylabel("# of imprecise sub-objects")

    def run_plot(self):
        df = self._prepare_dataset()
        sns.set_theme()

        self._plot_imprecise(df)
        self._plot_size(df)

    @output
    def imprecise_fields_plot(self):
        return PlotTarget(self, prefix="imprecision")

    @output
    def imprecise_fields_size_plot(self):
        return PlotTarget(self, prefix="size")

    @output
    def imprecise_fields_size_hist(self):
        return PlotTarget(self, prefix="size-hist")


class ImpreciseCommonMembersPlot(ImpreciseMembersPlotBase):
    """
    This is similar to :class:`AllImpreciseMembersPlot` but only displays imprecise struct
    members for structures that are present for all dataset_id values.
    This means that structures that are only compiled for Morello or RISC-V are not displayed.
    """
    public = True
    task_name = "imprecise-common-subobject-plot"

    def run_plot(self):
        df = self._prepare_dataset_filter_common()

        sns.set_theme()

        with new_figure(self.imprecise_fields_plot.paths(), figsize=(10, 50), constrained_layout=False) as fig:
            ax = fig.subplots()
            show_df = df.reset_index().melt(id_vars=df.index.names,
                                            value_vars=["base_rounding", "top_rounding"],
                                            var_name="source_of_imprecision")
            show_df["legend"] = (show_df["source_of_imprecision"] + " " +
                                 show_df["dataset_gid"].map(self.g_uuid_to_label))
            show_df["label"] = show_df["file"] + ": " + show_df["base_name"] + "::" + show_df["member_name"]
            (so.Plot(show_df, y="label", x="value", color="legend").on(ax).add(so.Bar(),
                                                                               so.Dodge(by=["dataset_gid"]),
                                                                               so.Stack(),
                                                                               dataset_gid=show_df["dataset_gid"],
                                                                               orient="y").plot())

            legend_on_top(fig, ax)
            # ax.set_xscale("log", base=2)
            ax.set_xlabel("Capability imprecision (Bytes)")
            ax.set_ylabel("Sub-object field")
            plt.setp(ax.get_yticklabels(), ha="right", va="center", fontsize="xx-small")

    @output
    def imprecise_fields_plot(self):
        return PlotTarget(self)


class RequiredSubobjectPrecision(ImpreciseMembersPlotBase):
    """
    Plot the number of bits required to ensure that imprecise sub-objects would be
    representable.
    """
    public = True
    task_name = "imprecise-subobject-bits-plot"

    @output
    def imprecise_bits_plot(self):
        return PlotTarget(self, prefix="all")

    @output
    def imprecise_bits_cdf(self):
        return PlotTarget(self, prefix="cdf")

    @output
    def imprecise_abs_bits_cdf(self):
        return PlotTarget(self, prefix="cdf-abs")

    @output
    def imprecise_common_bits_plot(self):
        return PlotTarget(self, prefix="common")

    @output
    def imprecise_common_bits_cdf(self):
        return PlotTarget(self, prefix="common-cdf")

    @output
    def imprecise_common_abs_bits_cdf(self):
        return PlotTarget(self, prefix="common-cdf-abs")

    def _compute_precision(self, base: float, top: float):
        """
        This should produce the mantissa size required for a specific (base, top) pair
        to be representable.
        """
        assert base <= top, "Invalid base > top"
        base = int(base)
        top = int(np.ceil(top))

        def lsb(x):
            if x == 0:
                return 0
            return int(np.log2(x & -x))

        len_msb = np.floor(np.log2(top - base)) if top != base else 0
        if top == 0:
            # Base must also be 0
            exponent = 0
        elif base == 0:
            exponent = lsb(top)
        else:
            exponent = min(lsb(base), lsb(top))
        return len_msb - exponent + 1

    def _platform_cap_format(self, g_uuid: uuid.UUID):
        instance_config = self.get_instance_config(g_uuid)
        if instance_config.cheri_target.is_riscv():
            # XXX support/detect riscv32
            cap_format = pychericap.CompressedCap128
        elif instance_config.cheri_target.is_morello():
            cap_format = pychericap.CompressedCap128m
        else:
            self.logger.error("DWARF TypeInfo extraction unsupported for %s", instance_config.cheri_target)
            raise RuntimeError(f"Unsupported instance target {instance_config.cheri_target}")
        return cap_format

    def _platform_mantissa_and_exp_width(self, g_uuid: uuid.UUID) -> tuple[int, int]:
        """
        Return the mantissa and exponent width of a given platform
        """
        cap_format = self._platform_cap_format(g_uuid)
        mantissa_width = cap_format.get_mantissa_width()
        exponent_width = 3
        return mantissa_width, exponent_width

    def _compute_platform_precision(self, g_uuid: uuid.UUID, base: float, top: float):
        """
        Compute the precision for a given [base, top] region on the platform identified
        by the g_uuid.

        Note that base and top may be floats if the region involves bit fields.
        """
        base = int(base)
        top = int(np.ceil(top))
        mantissa_width, exponent_width = self._platform_mantissa_and_exp_width(g_uuid)

        ie_threshold = mantissa_width - 2
        len_msb = np.floor(np.log2(top - base)) if top != base else 0
        if len_msb + 1 <= ie_threshold:
            precision = ie_threshold
        else:
            precision = mantissa_width - (1 + exponent_width)
        return precision

    def _plot_precision_bars(self, df, target):
        """
        Produce a bar plot showing the extra number of precision bits required
        for each imprecise sub-object member.
        """
        show_df = df.reset_index()
        show_df["legend"] = show_df["dataset_gid"].map(self.g_uuid_to_label)
        show_df["label"] = show_df["file"] + ": " + show_df["base_name"] + "::" + show_df["member_name"]
        show_df = show_df.sort_values(["member_size", "label", "dataset_gid"])

        with new_figure(target.paths(), figsize=(10, 50)) as fig:
            ax = fig.subplots()

            (so.Plot(show_df, y="label", x="additional_precision",
                     color="legend").on(ax).add(so.Bar(),
                                                so.Dodge(by=["dataset_gid"]),
                                                dataset_gid=show_df["dataset_gid"],
                                                orient="y").plot())

            legend_on_top(fig, ax)
            ax.set_xlabel("Increased precision (bits) required")
            ax.set_ylabel("Sub-object field")
            plt.setp(ax.get_yticklabels(), ha="right", va="center", fontsize="xx-small")

    def _plot_precision_cdf(self, df, target: PlotTarget, absolute: bool):
        """
        Produce a CDF plot showing the amount of imprecise sub-object members that
        can be "fixed" by adding a number of precision bits.
        """
        show_df = df.reset_index()
        show_df["legend"] = show_df["dataset_gid"].map(self.g_uuid_to_label)
        show_df["label"] = show_df["file"] + ": " + show_df["base_name"] + "::" + show_df["member_name"]
        show_df = show_df.sort_values(["member_size", "label", "dataset_gid"])

        with new_figure(target.paths()) as fig:
            ax = fig.subplots()

            if absolute:
                sns.ecdfplot(data=show_df, x="required_precision", hue="legend", ax=ax)
                ax.set_xlabel("Precision bits")
                ax.set_ylabel("Proportion of sub-objects that become representable")
            else:
                sns.ecdfplot(data=show_df, x="additional_precision", hue="legend", ax=ax)
                ax.set_xlabel("Additional precision bits")
                ax.set_ylabel("Proportion of sub-objects that become representable")

    def _plot_precision(self, df, bars_target, cdf_target, abs_cdf_target):
        """
        Generate the following plots for the given dataset:
        - A bar plot that shows the number of precision bits required to make an imprecise sub-object
          representable.
        - A CDF plot showing the proportion of imprecise sub-objects that become representable as
        the precision increases.
        - A CDF plot as above, but the X axis uses the absolute number of precision bits instead of the
        offset from the platform-specific precision.
        """
        sns.set_theme()
        # Compute imprecision bits
        # XXX just validate the model here?
        member_offset_index = df.index.names.index("member_offset")
        assert member_offset_index >= 0, "Missing member_offset from index"
        gid_index = df.index.names.index("dataset_gid")
        assert gid_index >= 0, "Missing dataset_gid from index"

        def _calc_precision(r):
            return self._compute_precision(r.name[member_offset_index], r.name[member_offset_index] + r["member_size"])

        df["required_precision"] = df.apply(_calc_precision, axis=1)

        def _calc_plat_precision(r):
            return self._compute_platform_precision(r.name[gid_index], r.name[member_offset_index],
                                                    r.name[member_offset_index] + r["member_size"])

        df["platform_precision"] = df.apply(_calc_plat_precision, axis=1)

        df["additional_precision"] = df["required_precision"] - df["platform_precision"]
        assert (df["additional_precision"] >= 0).all(), \
            "Something is wrong, these must be unrepresentable"

        self._plot_precision_bars(df, bars_target)
        self._plot_precision_cdf(df, cdf_target, absolute=False)
        self._plot_precision_cdf(df, abs_cdf_target, absolute=True)

    def run_plot(self):
        self._plot_precision(self._prepare_dataset(), self.imprecise_bits_plot, self.imprecise_bits_cdf,
                             self.imprecise_abs_bits_cdf)
        self._plot_precision(self._prepare_dataset_filter_common(), self.imprecise_common_bits_plot,
                             self.imprecise_common_bits_cdf, self.imprecise_common_abs_bits_cdf)


class CheriBSDSubobjectSizeDistribution(PlotTask):
    """
    Generate plots showing the distribution of subobject bounds sizes.

    This was originally designed to work with compiler diagnostics
    we should be able to adapt it to use both diagnostics and static data.
    """
    public = True
    task_namespace = "subobject"
    task_name = "subobject-size-distribution-plot"

    @dependency
    def data(self):
        return ImpreciseSubobjectBoundsUnion(self.session, self.analysis_config)

    def _plot_size_distribution(self, df: pd.DataFrame, target: PlotTarget):
        """
        Helper to plot the distribution of subobject bounds sizes
        """
        # Determine buckets we are going to use
        min_size = max(df["size"].min(), 1)
        max_size = max(df["size"].max(), 1)
        log_buckets = range(int(np.log2(min_size)), int(np.log2(max_size)) + 1)
        buckets = [2**i for i in log_buckets]

        with new_figure(target.paths()) as fig:
            ax = fig.subplots()
            sns.histplot(df, x="size", stat="count", bins=buckets, ax=ax)
            ax.set_yscale("log", base=10)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("size (bytes)")
            ax.set_ylabel("# of csetbounds")

    def run_plot(self):
        containers = self.data.all_layouts.get()
        ## XXX Port this
        # Note that it would be nice if we could also read stats
        # from the compiler-emitted setbounds and see how these distributions
        # compare.

        # # Filter only setbounds that are marked kind=subobject
        # data_df = df.loc[df["kind"] == SetboundsKind.SUBOBJECT.value]

        # sns.set_theme()

        # show_df = data_df.loc[data_df["src_module"] == "kernel"]
        # self._plot_size_distribution(show_df, self.size_distribution_kernel)

        # show_df = data_df.loc[data_df["src_module"] != "kernel"]
        # self._plot_size_distribution(show_df, self.size_distribution_modules)

        # self._plot_size_distribution(data_df, self.size_distribution_all)

    @output
    def size_distribution_kernel(self):
        return PlotTarget(self, prefix="size-distribution-kern")

    @output
    def size_distribution_modules(self):
        return PlotTarget(self, prefix="size-distribution-mods")

    @output
    def size_distribution_all(self):
        return PlotTarget(self, prefix="size-distribution-all")


class ImpreciseSetboundsSecurity(PlotTask):
    """
    Produce plots to display the impact of imprecise sub-object
    """
    public = True
    task_namespace = "subobject"
    task_name = "imprecise-subobject-security"

    @dependency
    def data(self):
        return ImpreciseSubobjectBoundsUnion(self.session, self.analysis_config)

    @output
    def categories(self):
        return PlotTarget(self, prefix="categories")

    @output
    def categories_percent(self):
        return PlotTarget(self, prefix="categories-percent")

    @output
    def categories_alias_ptr(self):
        return PlotTarget(self, prefix="alias-ptr")

    @output
    def categories_sizes_per_alias_kind(self):
        return PlotTarget(self, prefix="sizes-per-alias-kind")

    def run_plot(self):
        containers = self.data.all_layouts.get()
        records = []

        for c in containers:
            gid = uuid.UUID(c.layouts.graph["dataset_gid"])
            # Count categories for this data source
            for node, attrs in c.layouts.nodes(data=True):
                if attrs.get("alias_group_id") is None:
                    # Nothing else to do
                    continue

                if attrs["flags"] & pydwarf.TypeInfoFlags.kIsStruct:
                    category = "struct"
                elif attrs["flags"] & pydwarf.TypeInfoFlags.kIsUnion:
                    category = "union"
                elif attrs["flags"] & pydwarf.TypeInfoFlags.kIsArray:
                    category = "array"
                else:
                    self.logger.warning("Uncategorised sub-object node %s %s", node.member, attrs)
                    continue

                r = dict(dataset_gid=gid,
                         category=category,
                         alias_kind="alias-ptr" if attrs["alias_pointer_members"] else "other",
                         size=node.member_size)
                records.append(r)
        df = pd.DataFrame(records)
        df["platform"] = df["dataset_gid"].apply(self.g_uuid_to_label)
        sns.set_theme()

        # Plot the absolute number of imprecise members in each category grouped by platform
        with new_figure(self.categories.paths()) as fig:
            ax = fig.subplots()
            count_df = df.groupby(["platform", "category"], as_index=False).size().rename(columns={"size": "count"})
            sns.barplot(count_df, x="category", y="count", hue="platform", ax=ax)

        # Plot the % of imprecise members in each category grouped by platform
        with new_figure(self.categories_percent.paths()) as fig:
            ax = fig.subplots()
            total_df = df.groupby(["platform"], as_index=False).size().rename(columns={"size": "total"})
            count_df = df.groupby(["platform", "category"], as_index=False).size()
            percent_df = total_df.merge(count_df, on="platform")
            percent_df["% of all imprecise"] = 100 * percent_df["size"] / percent_df["total"]
            sns.barplot(percent_df, x="category", y="% of all imprecise", hue="platform", ax=ax)

        # Plot the % of imprecise members in each category that alias with pointer members,
        # grouped by platform
        with new_figure(self.categories_alias_ptr.paths()) as fig:
            ax = fig.subplots()
            count_df = df.groupby(["platform", "category"], as_index=False).size()
            alias_ptr = df[df["alias_kind"] == "alias-ptr"].groupby(["platform", "category"], as_index=False).size()
            percent_df = alias_ptr.merge(count_df, on=["platform", "category"], suffixes=["_ptr", "_total_by_kind"])
            percent_df["% aliasing with ptr"] = 100 * percent_df["size_ptr"] / percent_df["size_total_by_kind"]
            sns.barplot(percent_df, x="category", y="% aliasing with ptr", hue="platform", ax=ax)

        # Plot distribution of imprecise sub-object sizes grouped by platform, with one faced for
        # each aliasing kind.
        with new_facet(self.categories_sizes_per_alias_kind.paths(), df, row="platform", height=5, aspect=2.5) as facet:
            min_size = int(np.log2(df["size"].min()))
            max_size = int(np.log2(df["size"].max())) + 1
            hist_bins = [2**x for x in range(min_size, max_size)]
            # XXX can not use log_scale=2 with static bin widths
            facet.map_dataframe(sns.histplot, x="size", hue="category", bins=hist_bins, element="step")
            for ax in facet.axes.ravel():
                ax.set_xscale("log", base=2)
            facet.add_legend()


class ImpreciseSubobjectPaddingEstimator(PlotTask):
    """
    Determine the cost in terms of padding to fix the sub-object imprecision.
    """
    pass


class ImpreciseSubobjectLayoutReorderingEstimator(PlotTask):
    """
    Determine whether it is possible to reorder fields to fix the sub-object imprecision.
    """
    pass


class ImpreciseSetboundsLayouts(PlotTask):
    """
    Produce a html document that allows to browse the structure layouts with imprecision
    and inspect the data members that are affected by imprecision.
    """
    public = True
    task_namespace = "subobject"
    task_name = "imprecise-subobject-layouts"

    class LayoutDescription:
        ID_COUNTER = 0

        @classmethod
        def next_id(cls):
            cls.ID_COUNTER += 1
            return cls.ID_COUNTER

        def __init__(self, graph, node):
            self._graph = graph
            self._node = node
            self.id = f"layout-{self.next_id()}"

            # If this is a root, determine whether it contains
            # imprecise arrays or pointer aliasing.
            if self._node.member_size is None:
                self._set_root_helpers()

        def _set_root_helpers(self):
            """
            This is only meaningful on the root structure node.
            Determine whether it contains an imprecise member that aliases with a pointer.
            Determine whether it contains an imprecise array member.
            """
            self.has_imprecise_pointer_alias = False
            self.has_imprecise_array = False
            for child in nx.descendants(self._graph, self._node):
                child_desc = self.__class__(self._graph, child)
                if child_desc.is_imprecise:
                    if child_desc.is_array:
                        self.has_imprecise_array = True
                    if child_desc.is_aliasing_pointer_members:
                        self.has_imprecise_pointer_alias = True

        @property
        def depth(self):
            """Nest level"""
            return len(self._node.member.split(".")) - 2

        @property
        def base_name(self):
            """Name of the root structure"""
            return self._graph.nodes[self._node]["base_name"]

        @property
        def member_name(self):
            """Name of the member, without leading nested structs"""
            return self._node.member.split(".")[-1]

        @property
        def member_offset_str(self):
            """Offset representation as 0x<bytes>+<bits>"""
            offset = self._graph.nodes[self._node]["offset"]
            byte_offset = int(offset)
            bit_offset = int((offset - byte_offset) * 8)
            str_offset = f"{byte_offset:#x}"
            if bit_offset:
                str_offset += f":+{bit_offset:d}"
            return str_offset

        @property
        def member_type(self):
            """Type name of the member"""
            return self._graph.nodes[self._node]["type_name"]

        @property
        def base_size(self):
            """Size of the root structure"""
            return self._node.size

        @property
        def member_size_str(self):
            """Size of the member as 0x<bytes>+<bits>"""
            member_size = self._node.member_size
            byte_size = int(member_size)
            bit_size = int((member_size - byte_size) * 8)
            str_size = f"{byte_size:#x}"
            if bit_size:
                str_size += f":+{bit_size:d}"
            return str_size

        @property
        def is_imprecise(self):
            """Is this member node imprecise?"""
            return self.alias_id is not None

        @property
        def alias_id(self):
            """The alias group identifier for this member node, if any"""
            return self._graph.nodes[self._node].get("alias_group_id", None)

        @property
        def alias_groups(self):
            """
            The list of alias group identifiers of which this member is part.
            For each such group, there will be another member for which the sub-object capability
            aliases with the current member.
            """
            return ",".join(map(str, self._graph.nodes[self._node].get("alias_groups", [])))

        @property
        def is_aliasing_pointer_members(self):
            """Is this imprecise member aliasing another pointer member?"""
            if not self.is_imprecise:
                return False
            return self._graph.nodes[self._node].get("alias_pointer_members", False)

        @property
        def is_array(self):
            """Is this imprecise member an array?"""
            if not self.is_imprecise:
                return False
            return self._graph.nodes[self._node].get("flags", 0) & pydwarf.TypeInfoFlags.kIsArray

        @property
        def location(self):
            """Source code location for the root structure as <file>:<line>"""
            return f"{self._node.file}:{self._node.line}"

        def __iter__(self):
            for n in self._graph.successors(self._node):
                yield self.__class__(self._graph, n)

    @dependency
    def data(self):
        return ImpreciseSubobjectBoundsUnion(self.session, self.analysis_config)

    @output
    def layouts_table(self):
        return HTMLTemplateTarget(self, "imprecise-subobject-layout.html.jinja")

    def run(self):
        containers = self.data.all_layouts.get()

        layout_groups = {}
        for c in containers:
            group_name = self.g_uuid_to_label(c.layouts.graph["dataset_gid"])
            descriptions = []
            self.logger.debug("Prepare structures for %s", group_name)
            assert len(c.layouts.graph["roots"]) == len(set(c.layouts.graph["roots"]))
            for struct_root in c.layouts.graph["roots"]:
                if not c.layouts.nodes[struct_root].get("has_imprecise", False):
                    continue

                descriptions.append(self.LayoutDescription(c.layouts, struct_root))
            layout_groups[group_name] = descriptions

        self.layouts_table.render(layout_groups=layout_groups)
