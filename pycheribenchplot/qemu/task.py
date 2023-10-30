from dataclasses import dataclass

from ..core.artefact import Target
from ..core.config import InstancePlatform, ProfileConfig
from ..core.task import ExecutionTask, output


class QEMUTracingSetupTask(ExecutionTask):
    """
    This is the dependency task that modifies the current benchmark machine
    configuration to enable QEMU tracing.
    The task_config for this should be a ProfileConfig

    XXX need to split public platform options from private platform options that
    are populated here via the ProfileConfig
    """
    public = False
    task_namespace = "qemu.tracing"
    task_config_class = ProfileConfig

    @output
    def qemu_profile(self) -> Target:
        """The QEMU trace output file path"""
        return Target(self, "profile", template=f"qemu-perfetto-{self.benchmark.uuid}.pb")

    @output
    def qemu_interceptor(self) -> Target:
        """The QEMU interceptor trace output file path"""
        if self.config.qemu_trace == "perfetto-dynamorio":
            return None
        path = Path("qemu-trace-dir") / f"qemu-perfetto-interceptor-{self.benchmark.uuid}.trace.gz"
        return Target(self, "interceptor", template=str(path))

    def run(self):
        if self.benchmark.config.instance.platform != InstancePlatform.QEMU:
            raise ValueError("QEMU tracing is only allowed when running on QEMU")
        opts = self.benchmark.config.instance.platform_options
        if self.config.qemu_trace == "perfetto":
            opts.qemu_trace = "perfetto"
            opts.qemu_trace_file = self.qemu_profile.single_path()
        elif self.config.qemu_trace == "perfetto-dynamorio":
            path = self.qemu_interceptor.single_path()
            # Ensure that we have the paths
            path.parent.mkdir(exist_ok=True)
            opts.qemu_trace = "perfetto-dynamorio"
            opts.qemu_trace_file = self.qemu_profile.single_path()
            opts.qemu_interceptor_trace_file = path
        elif self.config.qemu_trace is not None:
            raise ValueError("Unknown configuration for ProfileConfig.qemu_trace")
        opts.qemu_trace_categories = self.config.qemu_trace_categories
