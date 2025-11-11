import re
from dataclasses import dataclass

import cxxfilt
import polars as pl

from ..core.analysis import AnalysisTask
from ..core.artefact import RemoteBenchmarkIterationTarget
from ..core.config import Config, ConfigPath, config_field
from ..core.plot import PlotTarget, SlicePlotTask
from ..core.plot_util import DisplayGrid, DisplayGridConfig, grid_barplot
from ..core.task import ExecutionTask, dependency, output
from ..core.util import gzopen


@dataclass
class C18NKtraceConfig(Config):
    """
    Configure ktrace for c18n user probes.
    """
    c18n_utrace_enable: bool = config_field(True, desc="Enable or disable c18n tracing")


class C18NKtraceExec(ExecutionTask):
    """
    Add-on task that instruments a benchmark to run under ktrace with
    c18n user probes.
    """
    public = True
    task_namespace = "c18n"
    task_name = "ktrace"
    task_config_class = C18NKtraceConfig

    @output
    def trace_data(self):
        return RemoteBenchmarkIterationTarget(self, "c18n-trace", ext="txt")

    def run(self):
        self.script.extend_context({
            "c18n_utrace_config": self.config,
            "c18n_utrace_gen_output_path": self.trace_data.shell_path_builder()
        })
