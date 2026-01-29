from dataclasses import dataclass
from enum import Enum

from ..core.config import ConfigPath, config_field
from ..generic.timing import TimingConfig, TimingExecTask


class KernelMakeTarget(Enum):
    BUILD_KERNEL = "kernel"
    BUILD_WORLD = "world"

    @property
    def target_string(self):
        return f"build{self.value}"


@dataclass
class KernelBuildConfig(TimingConfig):
    toolchain: ConfigPath | None = config_field(None, desc="Toolchain directory")
    objdir_prefix: ConfigPath = config_field("/usr/obj", desc="Kernel build directory")
    kernel_config: str = config_field(
        "GENERIC", desc="Name of the kernel config to build"
    )
    kernel_src_path: ConfigPath = config_field(
        "/usr/src", desc="Kernel sources directory"
    )
    make_target: KernelMakeTarget = config_field(
        KernelMakeTarget.BUILD_KERNEL, desc="Target to build (kernel, world)"
    )
    make_jobs: int = config_field(4, desc="Number of parallel make jobs")
    args: list[str] = config_field(list, "Extra kernel make arguments")


class KernelBuildBenchmarkExec(TimingExecTask):
    """
    Run a parallel kernel build on the host.
    """

    task_namespace = "kernel-build"
    task_name = "exec"
    task_config_class = KernelBuildConfig
    public = True

    def run(self):
        super().run()
        self.script.set_template("kernel-build.sh.jinja")
        self.script.extend_context({"kbuild_config": self.config})
        # XXX If hwpmc is also enabled, hwpmc follow fork must be switched on!
        # sanity check this here?
