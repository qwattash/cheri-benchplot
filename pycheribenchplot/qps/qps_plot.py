from dataclasses import dataclass

import polars as pl

from ..core.config import config_field
from ..core.error import ConfigurationError
from ..core.plot import PlotTarget, SlicePlotTask
from ..core.plot_util import (BarPlotConfig, DisplayGrid, DisplayGridConfig, grid_barplot, grid_pointplot)
from ..core.task import dependency, output
from .qps_exec import QpsExecTask


@dataclass
class QpsPlotConfig(DisplayGridConfig, BarPlotConfig):
    """
    Configuration for the QPS throughput plot.
    """
    drop_relative_baseline: bool = config_field(
        True, desc="Drop baseline value from relative metric columns (delta, overhead)")


class QpsPlotTask(SlicePlotTask):
    """
    Plot the QPS throughput as a bar plot.

    The configuration can be used to select which metrics to include in the
    plot. The tiling for the metric kind is controlled via the `_metric_type`
    axis in the :class:`QpsPlotConfig`.
    """
    task_namespace = "qps"
    task_name = "qps-plot"
    task_config_class = QpsPlotConfig
    public = True

    @dependency
    def qps_data(self):
        for bench in self.slice_benchmarks:
            task = bench.find_exec_task(QpsExecTask)
            yield task.qps_driver_output.get_loader()

    @output
    def qps_plot(self):
        return PlotTarget(self, "qps")

    def _collect_data(self):
        all_df = []
        for loader in self.qps_data:
            all_df.append(loader.df.get())
        df = pl.concat(all_df, how="vertical", rechunk=True)
        return df

    def _error_metric_type_unused(self):
        self.logger.error("The synthetic parameter _metric_type must be used, "
                          "otherwise the plot will combine absolute and relative data."
                          "See DisplayGridConfig::param_filter documentation.")
        raise ConfigurationError("Floating _metric_type column")

    def run_plot(self):
        df = self._collect_data()

        self.logger.info("Compute QPS statistics")
        stats = self.compute_overhead(df, "qps", how="median", overhead_scale=100)

        # Check that the configuration is either filtering or tiling on the metric
        # type
        if not self.config.uses_param("_metric_type"):
            if (self.config.param_filter is None or "_metric_type" not in self.config.param_filter):
                self._error_metric_type_unused()

        # Drop the baseline measurement from relative metrics.
        # This can make the plot cleaner, depending on tiling.
        if self.config.drop_relative_baseline:
            view_df = stats.filter((pl.col("_metric_type") == "absolute") | ~pl.col("_is_baseline"))
        else:
            view_df = stats

        self.logger.info("Generate QPS plot")
        with DisplayGrid(self.qps_plot, view_df, self.config) as grid:
            grid.map(grid_barplot, x=self.config.tile_xaxis, y="qps", err=["qps_low", "qps_high"], config=self.config)
            grid.add_legend()
