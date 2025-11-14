from dataclasses import dataclass

import polars as pl

from ..core.artefact import Target
from ..core.plot import PlotTarget, SlicePlotTask
from ..core.plot_util import (BarPlotConfig, PlotGrid, PlotGridConfig, grid_barplot)
from ..core.task import dependency, output
from .c18n_trace import C18nKtraceExec


@dataclass
class C18nTransitionFrequencyConfig(PlotGridConfig, BarPlotConfig):
    """"""
    pass


class C18nTransitionFrequencyPlot(SlicePlotTask):
    """
    Generate frequency plot of domain transitions between every
    compartment pair observed in a trace.

    This produces the following data columns:
    - transition_count: Number of transitions for a given (caller, callee) pair

    This produces the following metadata columns:
    - caller: The transition origin compartment
    - callee: The transition destination compartment
    """
    public = True
    task_namespace = "c18n"
    task_name = "transition-frequency"
    task_config_class = C18nTransitionFrequencyConfig

    @dependency
    def data(self):
        for bench in self.slice_benchmarks:
            ktrace_exec = bench.find_exec_task(C18nKtraceExec)
            yield ktrace_exec.trace_data.get_loader()

    @output
    def frequency_plot(self):
        return PlotTarget(self, "transition-freq")

    @output
    def transition_stats(self):
        return Target(self, "transition-stats", ext="csv")

    @output
    def self_transition_stats(self):
        return Target(self, "self-transition-stats", ext="csv")

    def run_plot(self):
        # Note: we avoid concat + rechunk because it will destroy memory usage
        chunks = [loader.df.get() for loader in self.data]

        self.logger.info("Aggregate transition counts")
        # XXX note that we aggregate across iterations, if we want to extract a median
        # this needs to be changed.
        total_chunks = [df.group_by(self.param_columns).len("total_transitions") for df in chunks]
        total = pl.concat(total_chunks)
        # Count transitions across each parameterisation
        tc_chunks = [df.group_by([*self.param_columns, "caller", "callee"]).len("transition_count") for df in chunks]
        tc = pl.concat(tc_chunks)

        self.logger.info("Generate raw stats")
        df = tc.join(total, on=self.param_columns)
        df.write_csv(self.transition_stats.single_path())

        # Count self-transitions
        self_trans = df.filter(pl.col("caller") == pl.col("callee")).group_by(self.param_columns).agg(
            pl.col("transition_count").sum().alias("self_transitions"),
            pl.col("total_transitions").first())
        self_trans = self_trans.with_columns(
            (pl.col("total_transitions") - pl.col("self_transitions")).alias("cross_transitions"),
            (pl.col("self_transitions") / pl.col("total_transitions")).alias("self_transition_rate"))
        self.logger.info("Generate self-transition stats")
        self_trans.write_csv(self.self_transition_stats.single_path())

        # Compute self-transitions statistics
        # self_stats = self.compute_overhead(self_trans, "self_transitions", how="median", overhead_scale=100)
        # print(self_stats)
