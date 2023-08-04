import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so

from ..compile_db import (BuildConfig, Builder, CheriBSDBuild, CheriBSDBuildHelper)
from ..core.analysis import AnalysisTask
from ..core.artefact import (AnalysisFileTarget, DataFrameTarget, LocalFileTarget)
from ..core.config import InstanceKernelABI
from ..core.elf.dwarf import DWARFManager, NestedMemberVisitor
from ..core.plot import PlotTarget, PlotTask, new_figure
from ..core.task import DataGenTask, dependency, output
from ..core.util import SubprocessHelper
from ..ext import pychericap, pydwarf
from .model import (ImpreciseSubobjectInfoModel, ImpreciseSubobjectInfoModelRecord, ImpreciseSubobjectLayoutModel,
                    SetboundsKind, SubobjectBoundsModel, SubobjectBoundsUnionModel)


class CheriBSDSubobjectStats(DataGenTask):
    """
    Extract sub-object bounds from the kernel build.

    When this generator is used, a cheribsd kernel is built for every
    platform instance configuration in the pipeline.
    The task will collect clang-generated sub-object bounds statistics.
    Note that this relies on the cheribsd kernel configuration to instruct
    the kernel build to emit sub-object bounds information.
    This can be easily accomplished by adding the following line to the desired
    kernel configuration:

    .. code-block::

        makeoptions     CHERI_SUBOBJECT_BOUNDS_STATS


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
    task_namespace = "kernel-static"
    task_name = "cheribsd-subobject-stats"
    task_config_class = BuildConfig

    def __init__(self, benchmark, script, task_config):
        if not task_config.target:
            task_config.target = "cheribsd"
            task_config.builder = Builder.CheriBuild
        assert task_config.target == "cheribsd"
        super().__init__(benchmark, script, task_config)

    def _extract_kernel_stats(self, build_root) -> pd.DataFrame:
        """
        Fetch kernel subobject bounds statistics from the kernel build.

        Load the statistics into a temporary dataframe to be merged with the module data.
        """
        build_path = self.cheribsd.kernel_build_path(build_root)
        path = build_path / "kernel-subobject-bounds-stats.csv"
        if not path.exists():
            self.logger.error("Missing csetbounds stats for %s", self.benchmark)
            raise RuntimeError("Missing kernel csetbounds stats")

        df = pd.read_csv(path)
        df["src_module"] = "kernel"
        return df

    def _extract_module_stats(self, build_root) -> list[pd.DataFrame]:
        build_path = self.cheribsd.kernel_build_path(build_root)
        df_set = []
        for path in build_path.glob("modules/**/kernel-subobject-bounds-stats.csv"):
            df = pd.read_csv(path)
            df["src_module"] = path.parent.name
            df_set.append(df)
        return df_set

    @dependency
    def cheribsd(self):
        return CheriBSDBuildHelper(self.benchmark, self.script, self.config)

    @output
    def subobject_stats(self):
        return LocalFileTarget(self, ext="csv", model=SubobjectBoundsModel)

    def run(self):
        instance_config = self.benchmark.config.instance
        if instance_config.kernelabi != InstanceKernelABI.PURECAP:
            # It makes no sense to build this because no subobject stats will
            # ever be emitted, just make an empty file
            self.subobject_stats.path.touch()
            return

        # Extract the files from the build dir
        df_set = [self._extract_kernel_stats(self.cheribsd.build_root)]
        df_set.extend(self._extract_module_stats(self.cheribsd.build_root))
        df = pd.concat(df_set)

        # Patch the alignment_bits and size for unknown values
        unknown_align = df["alignment_bits"].map(str).str.startswith("<unknown")
        unknown_size = df["size"].map(str).str.startswith("<unknown")
        df["alignment_bits"] = df["alignment_bits"].mask(unknown_align, None)
        df["size"] = df["size"].mask(unknown_size, None)

        df.to_csv(self.subobject_stats.path)


class ImpreciseSubobjectVisitor(NestedMemberVisitor):
    """
    Visit a DWARFInfoSource layout and extract imprecise sub-object members
    """
    def __init__(self, benchmark, logger, root):
        super().__init__(root)
        self.logger = logger
        self.records: list[ImpreciseSubobjectInfoModelRecord] = []

        instance_config = benchmark.config.instance
        if instance_config.cheri_target.is_riscv():
            # XXX support/detect riscv32
            self.cap_format = pychericap.CompressedCap128
        elif instance_config.cheri_target.is_morello():
            self.cap_format = pychericap.CompressedCap128m
        else:
            self.logger.error("DWARF TypeInfo extraction unsupported for %s", instance_config.cheri_target)
            raise RuntimeError(f"Unsupported instance target {instance_config.cheri_target}")

    def _check_imprecise(self, member: "Member", offset: int) -> tuple[int, int] | None:
        """
        Check if a specific member of a structure is representable.

        If it is not representable, return a tuple (new_base, new_top)
        """
        offset = offset + member.offset
        subobject_cap = self.cap_format.make_max_bounds_cap(offset)
        subobject_cap.setbounds(member.size)

        if subobject_cap.base() < offset or subobject_cap.top() > offset + member.size:
            return (subobject_cap.base(), subobject_cap.top())
        return None

    def visit_member(self, parent, member, prefix, offset):
        result = self._check_imprecise(member, offset)
        if not result:
            return
        self.logger.debug("Found imprecise member %s of type %s base=%d top=%d", member.name, self.root.type_name,
                          result[0], result[1])
        record = ImpreciseSubobjectInfoModelRecord(type_id=self.root.handle,
                                                   file=self.root.file,
                                                   line=self.root.line,
                                                   type_name=self.root.base_name,
                                                   member_offset=offset + member.offset,
                                                   member_name=prefix + member.name,
                                                   member_size=member.size,
                                                   member_aligned_base=result[0],
                                                   member_aligned_top=result[1])
        self.records.append(record)


class CheriBSDExtractImpreciseSubobject(DataGenTask):
    """
    Extract CheriBSD DWARF type information for aggregate data types.
    """
    public = True
    task_namespace = "kernel-static"
    task_name = "cheribsd-extract-imprecise-subobject"
    task_config_class = BuildConfig

    def __init__(self, benchmark, script, task_config):
        if not task_config.target:
            task_config.target = "cheribsd"
            task_config.builder = Builder.CheriBuild
        assert task_config.target == "cheribsd"
        super().__init__(benchmark, script, task_config)

    @dependency
    def cheribsd(self):
        return CheriBSDBuildHelper(self.benchmark, self.script, self.config)

    @output
    def imprecise_members(self):
        return LocalFileTarget(self, ext="csv", prefix="imprecise", model=ImpreciseSubobjectInfoModel)

    @output
    def struct_layout(self):
        return LocalFileTarget(self, ext="csv", prefix="layout", model=ImpreciseSubobjectLayoutModel)

    def _find_imprecise_subobjects(self, dw: "DWARFInfoSource", info: "TypeInfoContainer") -> pd.DataFrame:
        """
        Scan all the resolved structures and find sub-object bounds that are not
        representable.
        """
        all_imprecise = []
        for type_info in info.iter_composite():
            v = ImpreciseSubobjectVisitor(self.benchmark, self.logger, type_info)
            dw.visit_nested_layout(v)
            all_imprecise += v.records

        df = pd.DataFrame.from_records(all_imprecise, columns=ImpreciseSubobjectInfoModelRecord._fields)
        df.set_index(["file", "line", "type_name", "member_name", "member_offset"], inplace=True)
        return df[~df.index.duplicated(keep="first")]

    def _extract_layout(self, dw: "DWARFInfoSource", info: "TypeInfoContainer",
                        imprecise: pd.DataFrame) -> pd.DataFrame:
        """
        Emit the layout of each structure containing imprecise sub-object bounds.

        For each member aliased by the imprecise capabilities, record that it is being
        aliased and by which field(s).
        """
        layouts = []
        for type_id, members in imprecise.groupby("type_id"):
            type_info = info.find_composite(type_id)
            if not type_info:
                raise RuntimeError("Invalid type ID")
            layout_df = dw.parse_struct_layout(info, type_info)
            layout_df["alias_group_id"] = np.nan
            layout_df["alias_aligned_base"] = np.nan
            layout_df["alias_aligned_top"] = np.nan
            layout_df["alias_groups"] = None

            # Paint the layout with imprecise members
            members["alias_group_id"] = range(len(members))
            for m_index, m in members.iterrows():
                # Find the location of member m in the layout_df
                # This relies on index ordering
                assert members.index.names[-1] == "member_offset"
                assert members.index.names[-2] == "member_name"
                m_offset = m_index[-1]
                m_name = m_index[-2]
                m_in_layout, _ = layout_df.index.get_loc_level((m_offset, m_name),
                                                               level=["member_offset", "member_name"],
                                                               drop_level=False)
                # Mark all members that alias with m bounds (except m)
                offsets = layout_df.index.get_level_values("member_offset")
                overlap_base = offsets + layout_df["member_size"] > m["member_aligned_base"]
                overlap_top = offsets < m["member_aligned_top"]
                overlap = overlap_base & overlap_top & ~m_in_layout
                layout_df.loc[overlap, "alias_groups"] = layout_df.loc[overlap, "alias_groups"].apply(
                    lambda g: g.append(m["alias_group_id"]) if g is not None else [m["alias_group_id"]])

                # And set the alias fields
                assert m_in_layout.sum() == 1, "Can not find unique imprecise member in layout"
                layout_df.loc[m_in_layout, "alias_group_id"] = m["alias_group_id"]
                layout_df.loc[m_in_layout, "alias_aligned_base"] = m["member_aligned_base"]
                layout_df.loc[m_in_layout, "alias_aligned_top"] = m["member_aligned_top"]

            layouts.append(layout_df)
        return pd.concat(layouts, axis=0)

    def run(self):
        kernel_elf = self.cheribsd.kernel_build_path(self.cheribsd.build_root) / "kernel.full"
        if not kernel_elf.exists():
            self.logger.error("Missing kernel image at %s", kernel_elf)
            raise RuntimeError("Missing kernel image")
        # XXX the kernel pointer size should become unnecessary at some point
        dw = self.benchmark.dwarf.register_object("kernel",
                                                  kernel_elf,
                                                  arch_pointer_size=self.benchmark.config.instance.kernel_pointer_size)
        info = dw.load_type_info()
        df = self._find_imprecise_subobjects(dw, info)
        df.to_csv(self.imprecise_members.path)

        # Now, for every structure with imprecise members, we do a full dump of the structure
        layout_df = self._extract_layout(dw, info, df)
        layout_df.to_csv(self.struct_layout.path)


class CheriBSDSubobjectStatsUnion(AnalysisTask):
    """
    Merge all statistics about subobject bounds from differet kernel configurations.

    Note that in order to overlap, a bounds record must match in both location, size and alignment.
    """
    task_namespace = "kernel-static"
    task_name = "subobject-bounds-stats-union"

    @dependency
    def raw_subobject_stats(self):
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(CheriBSDSubobjectStats)
            yield task.subobject_stats.get_loader()

    def run(self):
        df = pd.concat([loader.df.get() for loader in self.raw_subobject_stats])
        df = df.groupby(["source_loc", "compiler_pass", "details"]).first()
        self.subobject_stats.assign(df)

    @output
    def subobject_stats(self):
        return DataFrameTarget(self, SubobjectBoundsUnionModel)


class CheriBSDSubobjectSizeBySize(AnalysisTask):
    """
    Generate a file that shows cases where subobject bounds are larger than 4K.

    Additional outputs are generated to separate large stack bounds and unknown cases.
    """
    public = True
    task_namespace = "kernel-static"
    task_name = "subobject-size-large"

    @dependency
    def stats(self):
        return CheriBSDSubobjectStatsUnion(self.session, self.analysis_config)

    def run(self):
        df = self.stats.subobject_stats.get()
        out_cols = ["src_module", "size", "compiler_pass", "details", "source_loc"]

        # Filter by kind=subobject and deterimne the largest
        out_df = df.loc[df["kind"] == SetboundsKind.SUBOBJECT.value]
        out_df = out_df.loc[out_df["size"] > 2**12]
        out_df = out_df.sort_values(by="size", ascending=False).reset_index()
        out_df[out_cols].to_csv(self.subobject_large.path)

    @output
    def subobject_large(self):
        return AnalysisFileTarget(self, ext="csv")


class CheriBSDSubobjectSizeDistribution(PlotTask):
    """
    Generate plots showing the distribution of subobject bounds sizes.
    """
    public = True
    task_namespace = "kernel-static"
    task_name = "subobject-size-distribution-plot"

    @dependency
    def stats(self):
        return CheriBSDSubobjectStatsUnion(self.session, self.analysis_config)

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
        df = self.stats.subobject_stats.get()

        # Filter only setbounds that are marked kind=subobject
        data_df = df.loc[df["kind"] == SetboundsKind.SUBOBJECT.value]

        sns.set_theme()

        show_df = data_df.loc[data_df["src_module"] == "kernel"]
        self._plot_size_distribution(show_df, self.size_distribution_kernel)

        show_df = data_df.loc[data_df["src_module"] != "kernel"]
        self._plot_size_distribution(show_df, self.size_distribution_modules)

        self._plot_size_distribution(data_df, self.size_distribution_all)

    @output
    def size_distribution_kernel(self):
        return PlotTarget(self, prefix="size-distribution-kern")

    @output
    def size_distribution_modules(self):
        return PlotTarget(self, prefix="size-distribution-mods")

    @output
    def size_distribution_all(self):
        return PlotTarget(self, prefix="size-distribution-all")


class ImpreciseSetboundsUnion(AnalysisTask):
    """
    Merge all imprecise subobject bounds warnings.

    This merges the datasets by platform g_uuid, so it is possible
    to observe the difference between the behaviour for Morello and
    RISC-V variants.
    """
    task_namespace = "kernel-static"
    task_name = "imprecise-subobject-bounds-union"

    @dependency
    def layout_data(self):
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(CheriBSDExtractImpreciseSubobject)
            yield task.struct_layout.get_loader()

    @dependency
    def imprecise_members(self):
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(CheriBSDExtractImpreciseSubobject)
            yield task.imprecise_members.get_loader()

    def run(self):
        df = pd.concat([loader.df.get() for loader in self.layout_data])
        self.all_layouts.assign(df)
        df = pd.concat([loader.df.get() for loader in self.imprecise_members])
        self.all_imprecise_members.assign(df)

    @output
    def all_layouts(self):
        return DataFrameTarget(self, ImpreciseSubobjectLayoutModel)

    @output
    def all_imprecise_members(self):
        return DataFrameTarget(self, ImpreciseSubobjectInfoModel)


class ImpreciseSetboundsPlot(PlotTask):
    """
    Produce a plot showing the difference in size and alignment caused by CHERI
    representability rounding on each sub-object.

    Each different platform (gid) is rendered separately.
    """
    public = True
    task_namespace = "kernel-static"
    task_name = "imprecise-subobject-plot"

    @dependency
    def data(self):
        return ImpreciseSetboundsUnion(self.session, self.analysis_config)

    def run_plot(self):
        df = self.data.all_imprecise_members.get()

        def shorten_name(n):
            if n.startswith("<anon>"):
                match = re.match(r"<anon>\.(.*)\.([0-9]+)", n)
                if not match:
                    return n
                path = Path(match.group(1))
                n = str(path.relative_to(self.session.user_config.src_path)) + "." + match.group(2)
            return n

        levels = df.index.names
        df = df.reset_index("type_name")
        df["type_name"] = df["type_name"].map(shorten_name)
        df = df.set_index("type_name", append=True).reorder_levels(levels)

        # Normalize base and size with respect to the "requested" base offset
        # and size
        member_offset = df.index.get_level_values("member_offset")
        df["aligned_size"] = df["member_aligned_top"] - df["member_aligned_base"]
        df["base_rounding"] = member_offset - df["member_aligned_base"]
        df["top_rounding"] = df["member_aligned_top"] - (member_offset + df["member_size"])

        assert (df["base_rounding"] >= 0).all()
        assert (df["top_rounding"] >= 0).all()

        sns.set_theme()

        with new_figure(self.imprecise_fields_plot.paths()) as fig:
            ax = fig.subplots()
            show_df = df.reset_index().melt(id_vars=df.index.names,
                                            value_vars=["base_rounding", "top_rounding"],
                                            var_name="source_of_imprecision")
            show_df["legend"] = (show_df["source_of_imprecision"] + " " +
                                 show_df["dataset_gid"].map(self.g_uuid_to_label))
            show_df["label"] = show_df["type_name"] + "::" + show_df["member_name"]
            (so.Plot(show_df, y="label", x="value", color="legend").on(ax).add(so.Bar(),
                                                                               so.Dodge(by=["dataset_gid"]),
                                                                               so.Stack(),
                                                                               dataset_gid=show_df["dataset_gid"],
                                                                               orient="y").plot())
            # Hack to remove the legend as we can not easily move it
            legend = fig.legends[0]
            legend.set_bbox_to_anchor((0., 1.02, 1., .102))

            ax.set_xlabel("Sub-object field")
            ax.set_ylabel("Capability imprecision (Bytes)")
            # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize="x-small")

    @output
    def imprecise_fields_plot(self):
        return PlotTarget(self)
