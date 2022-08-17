import subprocess
from pathlib import Path

import pandas as pd

from ..core.analysis import BenchmarkAnalysis
from ..core.config import DatasetName
from ..core.dataset import pivot_multi_index_level
from ..core.plot import LegendInfo, get_col_or_idx


class PMCStacksPlot(BenchmarkAnalysis):
    """
    Extract stack sample data for flamegraph.pl
    """
    require = {DatasetName.PMC_PROFCLOCK_STACKSAMPLE}
    name = "pmc-stacks"
    description = "Folded stacks data for flamegraph.pl"

    def _compute_folded_stacks_pairs(self, df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
        """
        Compute the combinations of (baseline, other) pairs of folded stacks for flamegraph output

        :return: A list of pairs as (benchmark_variant_name, folded_stacks_frame)
        """
        groups = self.benchmark.get_benchmark_groups()
        non_baseline = [g_uuid for g_uuid in groups.keys() if g_uuid != self.benchmark.g_uuid]
        group_names = {g_uuid: str(bench_list[0].config.instance.name) for g_uuid, bench_list in groups.items()}
        col_base = ("nsamples", "median", "sample", self.benchmark.g_uuid)
        col_stacks = ("folded_stacks", "", "", "")

        folded_stacks = []
        for variant_id in non_baseline:
            col_variant = ("nsamples", "median", "sample", variant_id)
            variant_name = group_names[variant_id]
            folded_stacks.append((variant_name, df[[col_stacks, col_base, col_variant]]))
        return folded_stacks

    def _emit_folded_stacks(self, suffix: str, variant_name: str, df: pd.DataFrame):
        """
        Write a folded stacks file to the appropriate location and feed the flamegraph program
        """
        stacks_path = self.get_folded_stacks_path()
        stacks_path = stacks_path.with_stem(f"{stacks_path.stem}{suffix}-{variant_name}")
        df.to_csv(stacks_path, header=False, index=False, sep=" ")

        flamegraph_gen = self.benchmark.user_config.flamegraph_path
        if not flamegraph_gen.exists():
            return
        flamegraph_path = stacks_path.with_suffix(".svg")
        self.logger.info("Emit flamegraph %s", flamegraph_path)
        with open(flamegraph_path, "w+") as fd:
            subprocess.run([flamegraph_gen, stacks_path], stdout=fd)

    def get_output_path(self):
        return (self.benchmark.get_plot_path() / self.name)

    def get_folded_stacks_path(self):
        return self.get_output_path().with_suffix(".stacks")

    async def process_datasets(self):
        flamegraph_gen = self.benchmark.user_config.flamegraph_path
        ds = self.get_dataset(DatasetName.PMC_PROFCLOCK_STACKSAMPLE)
        # We only operate on the gid groups here
        df = ds.agg_df.droplevel("dataset_id")
        self.logger.info("Extract folded stacks to %s", self.get_output_path())
        self.logger.debug("Using flamegraph.pl at %s", flamegraph_gen)
        if not flamegraph_gen.exists():
            self.logger.warning("Flamegraph generator not found %s", flamegraph_gen)

        # Now we need to pivot the dataset_gid to columns to have nsamples for each
        # kernel config
        df = pivot_multi_index_level(df, "dataset_gid").fillna(0)
        assert not df.isna().any().any()

        # And compute the folded stacks from the stacktrace index columns
        def joiner(values):
            return ";".join(filter(lambda v: bool(v), values))

        def folder(syminfo):
            return syminfo.to_folded_stack_str()

        df_stacks = df.index.to_frame()[ds.stacktrace_columns()]
        df["folded_stacks"] = df_stacks.applymap(folder).agg(joiner, axis=1)

        # Generate all combinations of folded stacks
        if ds.parameter_index_columns():
            for param_set, param_group in df.groupby(ds.parameter_index_columns()):
                if isinstance(param_set, tuple):
                    param_set_suffix = "-".join(param_set)
                else:
                    param_set_suffix = f"-{param_set}"
                folded_stacks = self._compute_folded_stacks_pairs(param_group)
                for variant_name, stacks_df in folded_stacks:
                    self._emit_folded_stacks(param_set_suffix, variant_name, stacks_df)
        else:
            folded_stacks = self._compute_folded_stacks_pairs(df)
            for variant_name, stacks_df in folded_stacks:
                self._emit_folded_stacks("", variant_name, stacks_df)
