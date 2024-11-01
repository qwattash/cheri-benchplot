from dataclasses import dataclass
from enum import Enum
from typing import List

import polars as pl

from ..core.artefact import RemoteBenchmarkIterationTarget
from ..core.config import Config, ConfigPath, config_field
from ..core.task import output
from ..core.tvrs import TVRSExecConfig, TVRSExecTask
from ..generic.timing import TimingConfig, TimingExecTask


class KernelMakeTarget(Enum):
    BUILD_KERNEL = "kernel"
    BUILD_WORLD = "world"

    @property
    def target_string(self):
        return f"build{self.value}"


@dataclass
class KernelBuildConfig(TVRSExecConfig, TimingConfig):
    objdir_prefix: ConfigPath = config_field("/usr/obj", desc="Kernel build directory")


@dataclass
class KernelBuildScenario(Config):
    kernel_config: str = config_field("GENERIC", desc="Name of the kernel config to build")
    kernel_src_path: ConfigPath = config_field("/usr/src", desc="Kernel sources directory")
    make_target: KernelMakeTarget = config_field(KernelMakeTarget.BUILD_KERNEL, desc="Target to build (kernel, world)")
    make_jobs: int = config_field(4, desc="Number of parallel make jobs")
    args: List[str] = config_field(list, "Extra kernel make arguments")


class KernelBuildBenchmarkExec(TVRSExecTask, TimingExecTask):
    """
    Run a parallel kernel build on the host.
    """
    task_namespace = "kernel-build"
    task_name = "exec"
    task_config_class = KernelBuildConfig
    scenario_config_class = KernelBuildScenario
    public = True

    def run(self):
        super().run()
        self.script.set_template("kernel-build.sh.jinja")
        self.script.extend_context({"kernel_build_config": self.config})
        # XXX If hwpmc is also enabled, hwpmc follow fork must be switched on!
        # sanity check this here?
