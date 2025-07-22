from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..core.artefact import PLDataFrameLoadTask, RemoteBenchmarkIterationTarget
from ..core.config import Config, config_field
from ..core.plot import PlotTarget, PlotTask
from ..core.plot_util import (DisplayGrid, DisplayGridConfig, grid_barplot, grid_pointplot)
from ..core.task import ExecutionTask, dependency, output


@dataclass
class SysctlConfig(Config):
    names: list[str] = config_field(list, desc="List of sysctl names to collect")


class IngestSysctl(PLDataFrameLoadTask):
    task_namespace = "generic"
    task_name = "sysctl-ingest"


class SysctlExecTask(ExecutionTask):
    """
    Base task that hooks the benchmark execution to sample the given
    set of sysctl values before and after each benchmark iteration.

    This can either be used as a base class to inherit the sysctl sampling or
    as an auxiliary generator in the pipelien configuration.
    """
    task_namespace = "generic"
    task_name = "sysctl"
    public = True

    @output
    def sample_before(self):
        return RemoteBenchmarkIterationTarget(self, "before", loader=IngestSysctl, ext="txt")

    @output
    def sample_after(self):
        return RemoteBenchmarkIterationTarget(self, "after", loader=IngestSysctl, ext="txt")

    def run(self):
        super().run()
        self.script.setup_iteration("sysctl", template="sysctl.hook.jinja")
        self.script.teardown_iteration("sysctl", template="sysctl.hook.jinja")
        self.script.extend_context({
            "sysctl_config": self.config,
            "sysctl_gen_out_before": self.sample_before.shell_path_builder(),
            "sysctl_gen_out_after": self.sample_after.shell_path_builder(),
        })
