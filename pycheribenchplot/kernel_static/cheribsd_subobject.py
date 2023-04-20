from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import seaborn as sns
from git import Repo

from ..core.analysis import AnalysisTask
from ..core.artefact import DataFrameTarget, LocalFileTarget
from ..core.config import Config, InstanceKernelABI
from ..core.plot import PlotTarget, PlotTask, new_figure
from ..core.task import DataGenTask, dependency, output
from ..core.util import SubprocessHelper
from .model import (SetboundsKind, SubobjectBoundsModel, SubobjectBoundsUnionModel)


@dataclass
class SubobjectStatsConfig(Config):
    """
    Task options for the cheribsd-subobject-bounds-stats generator
    """
    ephemeral_build_root: bool = False


class CheriBSDSubobjectStats(DataGenTask):
    """
    Extract sub-object bounds from the kernel build
    """
    public = True
    task_namespace = "kernel-static"
    task_name = "cheribsd-subobject-bounds-stats"
    task_config_class = SubobjectStatsConfig

    def __init__(self, benchmark, script, task_config=None):
        super().__init__(benchmark, script, task_config=task_config)
        #: Path to cheribuild script
        self._cheribuild = self.session.user_config.cheribuild_path / "cheribuild.py"
        #: The cheribsd repo
        self._repo = Repo(self.session.user_config.cheribsd_path)

    def _kernel_build_path(self, build_root: Path) -> Path:
        """
        Retrieve the kernel build directory from the build root directory
        """
        instance_config = self.benchmark.config.instance
        build_base = build_root / f"cheribsd-{instance_config.cheri_target.value}-build"
        path_match = list(build_base.glob(f"**/sys/{instance_config.kernel}"))
        if len(path_match) == 0:
            self.logger.error("No kernel build directory for %s in %s", instance_config.kernel, build_root)
            raise FileNotFoundError("Missing kernel build directory")
        assert len(path_match) == 1
        return path_match[0]

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

    def _do_run(self, build_root: Path):
        instance_config = self.benchmark.config.instance

        if instance_config.kernelabi != InstanceKernelABI.PURECAP:
            # It makes no sense to build this because no subobject stats will
            # ever be emitted, just make an empty file
            self.subobject_stats.path.touch()

        # Need to patch the selected kernel configuration to ask for subobject
        # bounds stats, so if the repo is dirty complain.
        if self._repo.is_dirty():
            self.logger.error("Working tree is dirty, this task needs to patch "
                              "the kernel config to produce subobject bounds stats")
            raise RuntimeError("CheriBSD working tree is dirty")
        kconfig = (self.session.user_config.cheribsd_path / instance_config.cheri_target.freebsd_kconf_dir() /
                   instance_config.kernel)
        target = f"cheribsd-{instance_config.cheri_target.value}"
        cbuild_opts = [
            "--build-root", build_root, "--clean", "--skip-update", "--skip-buildworld", "--skip-install", target,
            f"--{target}/kernel-config", instance_config.kernel
        ]

        try:
            with open(kconfig, "a") as kconfig_fd:
                kconfig_fd.write("\nmakeoptions CHERI_SUBOBJECT_BOUNDS_STATS\n")

            build_cmd = SubprocessHelper(self._cheribuild, cbuild_opts)
            build_cmd.run()
        finally:
            self._repo.git.checkout("HEAD", kconfig)

        # Now extract the files from the build dir
        df_set = [self._extract_kernel_stats(build_root)]
        df_set.extend(self._extract_module_stats(build_root))
        df = pd.concat(df_set)

        # Patch the alignment_bits and size for unknown values
        unknown_align = df["alignment_bits"].map(str).str.startswith("<unknown")
        unknown_size = df["size"].map(str).str.startswith("<unknown")
        df["alignment_bits"] = df["alignment_bits"].mask(unknown_align, None)
        df["size"] = df["size"].mask(unknown_size, None)

        df.to_csv(self.subobject_stats.path)

    def run(self):
        if self.config.ephemeral_build_root:
            with TemporaryDirectory() as build_dir:
                self._do_run(Path(build_dir))
        else:
            self._do_run(self.session.user_config.build_path)

    @output
    def subobject_stats(self):
        return LocalFileTarget(self, ext="csv", model=SubobjectBoundsModel)


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

    def run_plot(self):
        df = self.stats.subobject_stats.get()

        # Filter only setbounds that are marked kind=subobject
        show_df = df.loc[df["kind"] == SetboundsKind.SUBOBJECT.value]

        # Determine buckets we are going to use
        min_size = max(df["size"].min(), 1)
        max_size = max(df["size"].max(), 1)
        log_buckets = range(int(np.log2(min_size)), int(np.log2(max_size)) + 1)
        buckets = [2**i for i in log_buckets]

        sns.set_theme()

        with new_figure(self.size_distribution_kernel.paths()) as fig:
            ax = fig.subplots()
            sns.histplot(show_df, x="size", stat="count", bins=buckets, ax=ax)
            ax.set_yscale("log", base=10)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("size (bytes)")
            ax.set_ylabel("# of csetbounds")

    @output
    def size_distribution_kernel(self):
        return PlotTarget(self, prefix="size-distribution-kern")
