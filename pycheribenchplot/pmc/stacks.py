import json
import subprocess
from collections import defaultdict

from ..core.analysis import DatasetAnalysisTask
from ..core.artefact import ValueTarget
from ..core.plot import DatasetPlotTask, PlotTarget
from ..core.task import dependency, output
from .ingest import IngestPMCStatStacks


class PMCStacksFlameGraph(DatasetPlotTask):
    """
    Build a flame graph for each benchmark configuration.
    This uses brendandgregg's flamegraph.pl for simplicity

    TODO If we have multiple iterations, we merge the trees so we have more samples.
    """
    task_namespace = "pmc"
    task_name = "flamegraph"
    public = True

    @output
    def flamegraph(self):
        return PlotTarget(self, "stacks", ext="svg")

    @output
    def flamegraph_ref(self):
        return PlotTarget(self, "ref-stacks", ext="svg")

    def run(self):
        flamegraph_tool = self.session.user_config.flamegraph_path / "flamegraph.pl"
        if not flamegraph_tool.exists():
            self.logger.warning(
                "Tool flamegraph.pl not found, try setting user config flamegraph_path -- Skipping plot")
            return

        task = self.benchmark.find_exec_task(IngestPMCStatStacks)

        # Load the stacks data
        merged_data = defaultdict(lambda: 0)
        for path in task.data.iter_paths():
            with open(path, "r") as fd:
                in_data = json.load(fd)
            for key, value in in_data["stacks"].items():
                merged_data[key] += value

        with open(self.flamegraph.single_path(), "w+") as plot_file:
            with subprocess.Popen([flamegraph_tool], stdin=subprocess.PIPE, stdout=plot_file) as proc:
                for key, vale in merged_data.items():
                    proc.stdin.write(f"{key} {value}\n".encode("ascii"))
                proc.stdin.close()
                proc.wait()
                assert proc.returncode == 0

        with open(self.flamegraph_ref.single_path(), "w+") as plot_file:
            subprocess.run([flamegraph_tool, task.collapsed_stacks.paths()[0]],
                           stdout=plot_file)
