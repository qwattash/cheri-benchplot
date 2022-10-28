import typing
from dataclasses import dataclass, field
from pathlib import Path

from marshmallow.validate import OneOf

from ..core.config import ConfigPath, ProfileConfig, TemplateConfig
from ..core.task import DataFileTarget, ExecutionTask
from ..qemu.task import QEMUTracingSetupTask
from .model import NetperfData, NetperfStatData


@dataclass
class NetperfRunConfig(TemplateConfig):
    #: Path to netperf/netserver in the guest
    netperf_path: ConfigPath = Path("opt/{cheri_target}/netperf/bin")

    #: Actual benchmark options
    netperf_options: typing.List[str] = field(default_factory=list)

    #: Netserver options (used for both priming and the actual benchmark)
    netserver_options: typing.List[str] = field(default_factory=list)

    #: Benchmark profiling configuration
    profile: ProfileConfig = field(default_factory=ProfileConfig)


class NetperfExecTask(ExecutionTask):
    public = True
    task_name = "netperf"
    task_config_class = NetperfRunConfig

    def __init__(self, benchmark, script, **kwargs):
        super().__init__(benchmark, script, **kwargs)
        # Resolve binaries here as the configuration is stable at this point
        rootfs_netperf_base = self.benchmark.cheribsd_rootfs_path / self.config.netperf_path
        rootfs_netperf_bin = rootfs_netperf_base / "netperf"
        rootfs_netserver_bin = rootfs_netperf_base / "netserver"
        # Paths relative to the remote root directory
        self.netperf_bin = Path("/") / rootfs_netperf_bin.relative_to(self.benchmark.cheribsd_rootfs_path)
        self.netserver_bin = Path("/") / rootfs_netserver_bin.relative_to(self.benchmark.cheribsd_rootfs_path)
        self.logger.debug("Using %s %s", self.netperf_bin, self.netserver_bin)

    def get_stats_target(self, iteration: int):
        return DataFileTarget.from_task(self, iteration=iteration, extension="csv")

    def get_hwpmc_target(self, iteration: int) -> Path:
        """The remote profiling output target"""
        return DataFileTarget.from_task(self, name="hwpmc", iteration=iteration, extension="csv")

    def dependencies(self):
        if self.config.profile.qemu_trace:
            self.qemu_profile = QEMUTracingSetupTask(self.benchmark, self.script, task_config=self.config.profile)
            yield self.qemu_profile

    def run(self):
        # Prepare options
        run_env = {"STATCOUNTERS_NO_AUTOSAMPLE": "1"}
        extra_arguments = []

        # Handle profiling configuration
        if self.config.profile.hwpmc_trace == "pmc":
            extra_arguments += ["-g", "all"]
        elif self.config.profile.hwpmc_trace == "profclock":
            extra_arguments += ["-g", "profclock"]

        # Spawn the netserver process first
        s = self.script.sections["pre-benchmark"]
        netserver = s.add_cmd(self.netserver_bin, args=self.config.netserver_options, env=run_env, background=True)
        s.add_sleep(5)

        for i in range(self.benchmark.config.iterations):
            iteration_arguments = []
            if self.config.profile.hwpmc_trace:
                iteration_arguments += ["-G", self.get_profile_target(i).to_remote_path()]
            full_options = self.config.netperf_options + extra_arguments + iteration_arguments
            s = self.script.benchmark_sections[i]["benchmark"]
            s.add_cmd(self.netperf_bin, full_options, env=run_env, output=self.get_stats_target(i))

        s = self.script.sections["post-benchmark"]
        s.add_kill_cmd(netserver)

    def outputs(self):
        for i in range(self.benchmark.config.iterations):
            yield self.get_stats_target(i)
            if self.config.profile.hwpmc_trace:
                yield self.get_hwpmc_target(i)
