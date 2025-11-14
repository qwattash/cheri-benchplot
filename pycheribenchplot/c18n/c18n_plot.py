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
        total_chunks = [df.group_by(self.param_columns_with_iter).len("total_transitions") for df in chunks]
        total = pl.concat(total_chunks)
        del total_chunks
        # Count transitions across each parameterisation
        tc_chunks = [
            df.group_by([*self.param_columns_with_iter, "caller", "callee"]).len("transition_count") for df in chunks
        ]
        tc = pl.concat(tc_chunks)
        del tc_chunks

        self.logger.info("Generate raw stats")
        df = tc.join(total, on=self.param_columns)
        df.write_csv(self.transition_stats.single_path())

        # Count self-transitions statistics
        self.logger.info("Generate self-transition stats")
        df = df.filter(pl.col("caller") == pl.col("callee")).group_by(self.param_columns_with_iter).agg(
            pl.col("transition_count").sum().alias("self_transitions"),
            pl.col("total_transitions").first())
        df = df.with_columns((pl.col("total_transitions") - pl.col("self_transitions")).alias("cross_transitions"),
                             (pl.col("self_transitions") / pl.col("total_transitions")).alias("self_transition_rate"))
        df.write_csv(self.self_transition_stats.single_path())

        # Melt the dataframe to have "transition_type" and "transition_count" columns
        df = df.unpivot(on=["total_transitions", "self_transitions", "cross_transitions"],
                        index=self.param_columns_with_iter,
                        variable_name="_transition_type",
                        value_name="transition_count")
        stats = self.compute_overhead(df,
                                      "transition_count",
                                      how="median",
                                      extra_groupby=["_transition_type"],
                                      overhead_scale=100)
        with PlotGrid(self.frequency_plot, stats, self.config) as grid:
            grid.map(grid_barplot, x=self.config.tile_xaxis, y="transition_count", config=self.config)
            grid.add_legend()
