import shlex
from dataclasses import dataclass

from ..core.artefact import BenchmarkIterationTarget
from ..core.config import Config
from ..core.task import ExecutionTask, output


@dataclass
class GenericTaskConfig(Config):
    #: The command to execute
    command: str


class GenericExecTask(ExecutionTask):
    """
    This is a simple generic executor that uses the run_options configuration entry to run
    a custom command and collect the output to the output file.
    """
    public = True
    task_namespace = "generic"
    task_config_class = GenericTaskConfig

    @output
    def stdout(self):
        return BenchmarkIterationTarget(self, "stdout", ext="txt")

    def run(self):
        for i, path_entry in range(self.benchmark.config.iterations, self.stdout.paths()):
            _, output_path = path_entry
            section = self.script.benchmark_sections[i]["benchmark"]
            parts = shlex.split(self.config.command)
            section.add_cmd(parts[0], parts[1:], output=output_path)
