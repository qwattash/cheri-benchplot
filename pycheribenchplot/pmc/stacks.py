import json
import subprocess
from collections import defaultdict

from ..core.analysis import DatasetAnalysisTask
from ..core.artefact import BenchmarkIterationTarget, ValueTarget
from ..core.plot import PlotTarget, SlicePlotTask
from ..core.task import dependency, output
from .pmc_exec import PMCExec, PMCExecConfig


class PMCStacksFlameGraph(SlicePlotTask):
    """
    Build a flame graph for each benchmark configuration.
    This uses brendandgregg's flamegraph.pl for simplicity

    TODO If we have multiple iterations, we merge the trees so we have more samples.
    """
    task_namespace = "pmc"
    task_name = "flamegraph"
    public = True

    def __init__(self, benchmark, analysis_config, task_config):
        super().__init__(benchmark, analysis_config, task_config)
        self.flamegraph_tool = self.session.user_config.flamegraph_path / "flamegraph.pl"
        self.stackcollapse_tool = self.session.user_config.flamegraph_path / "stackcollapse-pmc.pl"
        if not self.flamegraph_tool.exists():
            self.logger.warning("Tool flamegraph.pl not found, try setting user config flamegraph_path")
            self.flamegraph_tool = None
        if not self.stackcollapse_tool.exists():
            self.logger.warning("Tool stackcollapse-pmc.pl not found, try setting user config flamegraph_path")
            self.stackcollapse_tool = None

    @output
    def flamegraph(self):
        return PlotTarget(self, "stacks", ext="svg")

    @output
    def stackcollapse_data(self):
        return BenchmarkIterationTarget(self, "collapsed-stacks", ext="txt")

    def run(self):
        if not self.flamegraph_tool or not self.stackcollapse_tool:
            self.logger.warning("Skipping plot")
            return

        task = self.benchmark.find_exec_task(PMCExec)
        if not task.config.sampling_mode:
            self.logger.error("Task requires sampling mode counters")
            raise RuntimeError("Configuration error")

        # Collapse the stacks data
        for path, out_path in zip(task.pmc_data.iter_paths(), self.stackcollapse_data.iter_paths()):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as out_fd:
                self.logger.debug("Run %s %s", self.stackcollapse_tool, path)
                subprocess.run([self.stackcollapse_tool, path], stdout=out_fd)

        # Merge collapsed stack samples
        merged_data = defaultdict(lambda: 0)
        for path in self.stackcollapse_data.iter_paths():
            with open(path, "r") as fd:
                for line in fd:
                    stack, count = line.split(" ")
                    merged_data[stack] += int(count)

        self.logger.debug("Emit flamegraph %s", self.flamegraph.single_path())
        with open(self.flamegraph.single_path(), "w+") as plot_file:
            with subprocess.Popen([self.flamegraph_tool], stdin=subprocess.PIPE, stdout=plot_file) as proc:
                for key, value in merged_data.items():
                    proc.stdin.write(f"{key} {value}\n".encode("ascii"))
                proc.stdin.close()
                proc.wait()
                assert proc.returncode == 0
