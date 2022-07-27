import subprocess

from ..core.analysis import BenchmarkAnalysis
from ..core.config import DatasetName
from ..core.plot import get_col_or_idx


class PMCStacksPlot(BenchmarkAnalysis):
    """
    Extract stack sample data for flamegraph.pl
    """
    require = {DatasetName.PMC_PROFCLOCK_STACKSAMPLE}
    name = "pmc-stacks-folded"
    description = "Folded stacks data for flamegraph.pl"

    def get_output_path(self):
        return (self.benchmark.get_plot_path() / self.name)

    def process_datasets(self):
        ds = self.get_dataset(DatasetName.PMC_PROFCLOCK_STACKSAMPLE)
        self.logger.info("Extract folded stacks to %s", self.get_output_path())
        folded_stacks = []
        sym_chain = {}

        groups = self.benchmark.get_benchmark_groups()
        non_baseline = [g_uuid for g_uuid in groups.keys() if g_uuid != self.benchmark.g_uuid]

        baseline = self.benchmark.g_uuid
        sel = non_baseline[0]
        col_leaf = ("nsamples", "median", "sample")

        def gen_flame(node, level, parent):
            if parent is None:
                return
            if parent == ds.agg_callchain.root:
                line = f"kern`{node.sym.name}"
            else:
                line = sym_chain[parent] + f";kern`{node.sym.name}"
            sym_chain[node] = line

            baseline_count = node.df.xs(baseline, level="dataset_gid")[col_leaf].iloc[0]
            count = node.df.xs(sel, level="dataset_gid")[col_leaf].iloc[0]
            if ds.agg_callchain.ct.out_degree(node) == 0:
                line += f" {baseline_count} {count}"
                folded_stacks.append(line)

        ds.agg_callchain.visit(gen_flame)

        folded_stacks_file = self.get_output_path().with_suffix(".txt")
        self.logger.info("Emit folded stacks %s", folded_stacks_file)
        with open(folded_stacks_file, "w+") as fd:
            for line in folded_stacks:
                fd.write(line + "\n")

        flamegraph_file = self.get_output_path().with_suffix(".svg")
        flamegraph_gen = self.benchmark.user_config.flamegraph_path
        self.logger.debug("Using flamegraph.pl at %s", flamegraph_gen)
        if flamegraph_gen.exists():
            self.logger.info("Emit flamegraph %s", flamegraph_file)
            with open(flamegraph_file, "w+") as fd:
                subprocess.run([flamegraph_gen, folded_stacks_file], stdout=fd)
        else:
            self.logger.warn("Flamegraph generator not found %", flamegraph_gen)
