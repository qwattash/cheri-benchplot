from dataclasses import dataclass
from typing import List, Optional

import polars as pl
import polars.selectors as cs
import seaborn as sns

from ..core.analysis import AnalysisTask
from ..core.artefact import ValueTarget
from ..core.config import config_field
from ..core.plot import PlotTarget, PlotTask
from ..core.plot_util import PlotGrid, PlotGridConfig, grid_barplot
from ..core.task import dependency, output
from ..core.tvrs import TVRSParamsMixin
from .netperf_exec import NetperfExecTask


class NetperfStatsUnion(AnalysisTask):
    """
    Load all netperf run data.
    """
    task_namespace = "netperf"
    task_name = "load"

    @dependency
    def data(self):
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(NetperfExecTask)
            yield task.results.get_loader()

    @output
    def df(self):
        return ValueTarget(self, "all-data")

    def run(self):
        df = pl.concat((t.df.get() for t in self.data), how="vertical", rechunk=True)
        self.df.assign(df)


@dataclass
class NetperfPlotConfig(PlotGridConfig):
    netperf_metrics: List[str] = config_field(lambda: ["throughput", "mean latency microseconds"],
                                              desc="Netperf metrics to plot")


class NetperfSummaryPlot(TVRSParamsMixin, PlotTask):
    """
    Produce a box plot of the netperf metrics.
    """
    task_namespace = "netperf"
    task_name = "summary-plot"
    task_config_class = NetperfPlotConfig
    public = True

    @dependency
    def stats(self):
        return NetperfStatsUnion(self.session, self.analysis_config)

    @output
    def summary(self):
        return PlotTarget(self, "summary")

    @output
    def overhead(self):
        return PlotTarget(self, "overhead")

    def run_plot(self):
        df = self.stats.df.get()
        # Make all metrics lowercase
        df = df.select(cs.all().name.to_lowercase())

        # Make the dataframe long-form
        index_cols = [*self.param_columns, "iteration"]
        df = df.unpivot(index=index_cols, on=self.config.netperf_metrics, variable_name="metric", value_name="value")

        # Compute overhead
        self.logger.info("Bootstrap netperf overhead confidence intervals")
        stats = self.compute_overhead(df, "value", extra_groupby=["metric"], how="median", overhead_scale=100)

        self.logger.info("Plot absolute netperf metrics")
        median_df = stats.filter(_metric_type="absolute")
        grid_config = self.config.set_display_defaults(param_names={
            self.config.hue: self.config.hue.capitalize(),
            "value": "Value"
        }).set_fixed(tile_row="metric")
        with PlotGrid(self.summary, median_df, grid_config) as grid:
            grid.map(grid_barplot, x="scenario", y="value", err=["value_low", "value_high"])
            grid.add_legend()

        self.logger.info("Plot overhead netperf metrics")
        overhead_df = stats.filter(_metric_type="overhead")
        grid_config = self.config.set_display_defaults(param_names={
            self.config.hue: self.config.hue.capitalize(),
            "value": "% Overhead"
        }).set_fixed(tile_row="metric")
        with PlotGrid(self.overhead, overhead_df, grid_config) as grid:
            grid.map(grid_barplot, x="scenario", y="value", err=["value_low", "value_high"])
            grid.add_legend()
