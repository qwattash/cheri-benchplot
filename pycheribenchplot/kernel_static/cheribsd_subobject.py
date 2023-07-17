import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..compile_db import CheriBSDBuild
from ..core.analysis import AnalysisTask
from ..core.artefact import (AnalysisFileTarget, DataFrameTarget, LocalFileTarget)
from ..core.config import InstanceKernelABI
from ..core.plot import PlotTarget, PlotTask, new_figure
from ..core.task import dependency, output
from ..core.util import SubprocessHelper
from .model import (ImpreciseSubobjectModel, SetboundsKind, SubobjectBoundsModel, SubobjectBoundsUnionModel)


class CheriBSDSubobjectStats(CheriBSDBuild):
    """
    Extract sub-object bounds from the kernel build
    """
    public = True
    task_namespace = "kernel-static"
    task_name = "cheribsd-subobject-bounds-stats"

    def _extract_kernel_stats(self, build_root) -> pd.DataFrame:
        """
        Fetch kernel subobject bounds statistics from the kernel build.

        Load the statistics into a temporary dataframe to be merged with the module data.
        """
        build_path = self._kernel_build_path(build_root)
        path = build_path / "kernel-subobject-bounds-stats.csv"
        if not path.exists():
            self.logger.error("Missing csetbounds stats for %s", self.benchmark)
            raise RuntimeError("Missing kernel csetbounds stats")

        df = pd.read_csv(path)
        df["src_module"] = "kernel"
        return df

    def _extract_module_stats(self, build_root) -> list[pd.DataFrame]:
        build_path = self._kernel_build_path(build_root)
        df_set = []
        for path in build_path.glob("modules/**/kernel-subobject-bounds-stats.csv"):
            df = pd.read_csv(path)
            df["src_module"] = path.parent.name
            df_set.append(df)
        return df_set

    def _do_build(self, build_root: Path):
        instance_config = self.benchmark.config.instance

        if instance_config.kernelabi != InstanceKernelABI.PURECAP:
            # It makes no sense to build this because no subobject stats will
            # ever be emitted, just make an empty file
            self.subobject_stats.path.touch()
            return
        super()._do_build(build_root)

    def _extract(self, build_root: Path):
        # Extract the files from the build dir
        df_set = [self._extract_kernel_stats(build_root)]
        df_set.extend(self._extract_module_stats(build_root))
        df = pd.concat(df_set)

        # Patch the alignment_bits and size for unknown values
        unknown_align = df["alignment_bits"].map(str).str.startswith("<unknown")
        unknown_size = df["size"].map(str).str.startswith("<unknown")
        df["alignment_bits"] = df["alignment_bits"].mask(unknown_align, None)
        df["size"] = df["size"].mask(unknown_size, None)

        df.to_csv(self.subobject_stats.path)

    @output
    def subobject_stats(self):
        return LocalFileTarget(self, ext="csv", model=SubobjectBoundsModel)


class CheriBSDImpreciseSubobject(CheriBSDBuild):
    """
    Run a static analysis to determine whether there are nested structures with
    unrepresentable sub-object bounds.
    """
    public = True
    task_namespace = "kernel-static"
    task_name = "cheribsd-imprecise-subobject-bounds"

    def __init__(self, benchmark, script, task_config=None):
        super().__init__(benchmark, script, task_config=task_config)
        # Complain if doing this for morello for now.
        target = self.benchmark.config.instance.cheri_target
        if target.is_morello():
            self.logger.error("Imprecise subobject unsupported for Morello clang-tidy")
            raise NotImplementedError("Not supported")
        #: Path to clang-tidy
        self._clang_tidy = self.session.user_config.sdk_path / "sdk" / "bin" / "clang-tidy"
        if not self._clang_tidy.exists():
            self.logger.error("Missing clang-tidy in Cheri SDK: %s", self._clang_tidy)
            raise RuntimeError("Missing CHERI tool")

    def _extract(self, build_root: Path):
        objdir = self._kernel_build_path(build_root)
        compdb = objdir / "compile_commands.json"

        # Read compilation DB to find all files we care about
        with open(compdb, "r") as compfd:
            db_data = json.load(compfd)
        all_files = [e["file"] for e in db_data if "file" in e and Path(e["file"]).suffix == ".c"]
        tidy_options = [
            "-p", objdir, "-checks=-*,misc-cheri-representable-subobject", "--system-headers", "-header-filter=.*"
        ] + all_files

        tidy_lines = []

        def collect_tidy_output(data):
            tidy_lines.append(data)

        tidy_cmd = SubprocessHelper(self._clang_tidy, tidy_options)
        tidy_cmd.set_stderr_loglevel(logging.DEBUG)
        tidy_cmd.observe_stdout(collect_tidy_output)
        tidy_cmd.run(cwd=objdir)

        # Parse the output
        base_matcher = re.compile(r"(?P<path>[/a-zA-Z0-9_.-]+):(?P<line>[0-9]+):(?P<column>[0-9]+):.*"
                                  "Field '(?P<field>[a-zA-Z0-9_]+)'.*"
                                  "\('(?P<field_type>[a-zA-Z0-9\[\]_ \t*&]+)'.*\).*"
                                  "at (?P<offset>[0-9]+).*"
                                  "in '(?P<container>[a-zA-Z0-9_]+)'.*"
                                  "size (?P<size>[0-9]+).*"
                                  "offset.*aligned to (?P<aligned_offset>[0-9]+)")
        size_matcher = re.compile(r"(?P<path>[/a-zA-Z0-9_.-]+):(?P<line>[0-9]+):(?P<column>[0-9]+):.*"
                                  "Field '(?P<field>[a-zA-Z0-9_]+)'.*"
                                  "\('(?P<field_type>[a-zA-Z0-9\[\]_ \t*&]+)'.*\).*"
                                  "at (?P<offset>[0-9]+).*"
                                  "in '(?P<container>[a-zA-Z0-9_]+)'.*"
                                  "size (?P<size>[0-9]+).*"
                                  "top.*aligned to (?P<aligned_top>[0-9]+)")

        base_groups = []
        size_groups = []
        for line in tidy_lines:
            base_m = base_matcher.match(line)
            if base_m:
                base_groups.append(base_m.groupdict())
            size_m = size_matcher.match(line)
            if size_m:
                size_groups.append(size_m.groupdict())
            if not base_m and not size_m and "warning:" in line:
                self.logger.warning("Unmatched clang-tidy warning: %s", line)

        srcdir = self.session.user_config.cheribsd_path
        base_df = pd.DataFrame(base_groups)
        size_df = pd.DataFrame(size_groups)

        assert not (base_df.empty and size_df.empty), "Missing inputs"
        if base_df.empty:
            df = size_df
            df["aligned_offset"] = np.nan
        elif size_df.empty:
            df = base_df
            df["aligned_size"] = np.nan
        else:
            df = pd.merge(base_df,
                          size_df,
                          how="outer",
                          on=["path", "line", "column", "field", "field_type", "container", "offset", "size"])
        df["path"] = df["path"].map(lambda p: Path(p).relative_to(srcdir))
        df = df.set_index(["path", "line", "column", "field", "field_type", "container"])
        df.to_csv(self.imprecise_warnings.path)

    @output
    def imprecise_warnings(self):
        return LocalFileTarget(self, ext="csv", model=ImpreciseSubobjectModel)


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

    # Note this assumes that the data is independent of any parameterization
    group_keys = ["dataset_gid", "path", "line", "column", "field", "field_type", "container"]

    @dependency
    def raw_imprecise_bounds(self):
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(CheriBSDImpreciseSubobject)
            yield task.imprecise_warnings.get_loader()

    def run(self):
        df = pd.concat([loader.df.get() for loader in self.raw_imprecise_bounds])
        df = df.groupby(self.group_keys).first()
        self.warnings_union.assign(df)

    @output
    def warnings_union(self):
        return DataFrameTarget(self, ImpreciseSubobjectModel.as_groupby(self.group_keys))


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
        df = self.data.warnings_union.get()

        # Normalize base and size with respect to the "requested" base offset
        # and size
        df["aligned_size"] = df["aligned_top"] - df[["offset", "aligned_offset"]].min(axis=1)
        df["front_padding"] = (df["offset"] - df["aligned_offset"]).fillna(0)
        df["back_padding"] = (df["aligned_size"] - df["size"] - df["front_padding"]).fillna(0)

        assert (df["front_padding"] >= 0).all()
        assert (df["back_padding"] >= 0).all()

        sns.set_theme()
        import seaborn.objects as so

        with new_figure(self.imprecise_fields_plot.paths()) as fig:
            ax = fig.subplots()
            show_df = df.reset_index().melt(id_vars=df.index.names,
                                            value_vars=["front_padding", "back_padding"],
                                            var_name="padding_type")
            show_df["label"] = show_df["container"] + "::" + show_df["field"]
            so.Plot(show_df, x="label", y="value", color="padding_type").on(ax).add(so.Bar(), so.Stack()).plot()
            ax.set_xlabel("Sub-object field")
            ax.set_ylabel("Capability imprecision (Bytes)")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize="x-small")

    @output
    def imprecise_fields_plot(self):
        return PlotTarget(self)
