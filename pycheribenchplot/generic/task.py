import shlex
from dataclasses import dataclass

from ..core.artefact import RemoteBenchmarkIterationTarget
from ..core.config import Config, config_field
from ..core.task import output
from .timing import TimingConfig, TimingExecTask, TimingPlotTask


@dataclass
class GenericTaskConfig(TimingConfig):
    #: The command to execute
    command: str = config_field(Config.REQUIRED, desc="Workload command to execute")
    #: Collect command output, note this is incompatible with timing.
    collect_stdout: bool = config_field(False, desc="Collect command stdout, note this is incompatible with timing")

    # XXX validation hook to check for collect_stdout && timing conflict


class GenericExecTask(TimingExecTask):
    """
    This is a simple generic executor that uses the run_options configuration entry to run
    a custom command and collect the output to the output file.
    """
    public = True
    task_namespace = "generic"
    task_config_class = GenericTaskConfig

    @output
    def stdout(self):
        # TODO handle optional outputs as optional dependencies
        return RemoteBenchmarkIterationTarget(self, "stdout", ext="txt")

    def run(self):
        super().run()
        self.script.set_template("generic.sh.jinja")
        self.script.extend_context({"generic_config": self.config, "stdout_paths": self.stdout.shell_path_builder()})
