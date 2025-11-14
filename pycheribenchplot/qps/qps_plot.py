from dataclasses import dataclass
from enum import Enum

import polars as pl
from marshmallow import ValidationError, validates_schema

from ..core.artefact import Target
from ..core.config import config_field
from ..core.error import ConfigurationError
from ..core.plot import PlotTarget, SlicePlotTask
from ..core.plot_util import (BarPlotConfig, LinePlotConfig, PlotGrid, PlotGridConfig, grid_barplot, grid_lineplot)
from ..core.task import dependency, output
from .qps_exec import QpsExecTask


class QpsPlotBase(SlicePlotTask):
    """
    Plot that operates on a slice of the QPS benchmark data.
    """
    @dependency
    def qps_data(self):
        for bench in self.slice_benchmarks:
            task = bench.find_exec_task(QpsExecTask)
            yield task.qps_driver_output.get_loader()

    def _collect_data(self):
        all_df = []
        for loader in self.qps_data:
            all_df.append(loader.df.get())
        df = pl.concat(all_df, how="vertical", rechunk=True)
        return df


@dataclass
class QpsPlotConfig(PlotGridConfig):
    """
    Configuration for the QPS throughput plot.
    """
    bar_plot: BarPlotConfig | None = config_field(None, desc="Bar plot configuration.")
    line_plot: LinePlotConfig | None = config_field(None, desc="Line plot configuration.")
    drop_relative_baseline: bool = config_field(
        True, desc="Drop baseline value from relative metric columns (delta, overhead)")

    @validates_schema
    def check_plot_kind(self, data, **kwargs):
        if data["bar_plot"] is None == data["line_plot"] is None:
            raise ValidationError("One and only one of bar_plot or line_plot must be configured.")


class QpsPlotTask(QpsPlotBase):
    """
    Plot the QPS throughput as a bar or line plot.

    The configuration can be used to select which metrics to include in the
    plot. The tiling for the metric kind is controlled via the `_metric_type`
    axis in the :class:`QpsPlotConfig`.
    """
    task_namespace = "qps"
    task_name = "qps-plot"
    task_config_class = QpsPlotConfig
    public = True

    @output
    def qps_plot(self):
        return PlotTarget(self, "qps")

    @output
    def qps_stats(self):
        return Target(self, "qps-stats", ext="csv")

    def run_plot(self):
        df = self._collect_data()

        self.logger.info("Compute QPS statistics")
        stats = self.compute_overhead(df, "qps", how="median", overhead_scale=100)

        if self.config.drop_relative_baseline:
            view_df = stats.filter((pl.col("_metric_type") == "absolute") | ~pl.col("_is_baseline"))
        else:
            view_df = stats

        self.logger.info("Generate QPS plot")
        with PlotGrid(self.qps_plot, view_df, self.config) as grid:
            # Dump the sorted tabular data using the ordering specified by the grid config
            dump_df = grid.get_grid_df()
            dump_df.write_csv(self.qps_stats.single_path())

            if plot_config := self.config.bar_plot:
                grid.map(grid_barplot,
                         x=plot_config.tile_xaxis,
                         y="qps",
                         err=["qps_low", "qps_high"],
                         config=plot_config)
            if plot_config := self.config.line_plot:
                grid.map(grid_lineplot,
                         x=plot_config.tile_xaxis,
                         y="qps",
                         err=["qps_low", "qps_high"],
                         config=plot_config)
            grid.add_legend()


class QpsLatencyPlotTask(QpsPlotBase):
    """
    Plot the QPS latency as a bar or line plot.

    The QPS output included the full histogram of the recorded latencies along with
    the pre-computed latency percentiles.
    This produces the median 90th percentile latency over the benchmark iterations.
    NOTE: This is _NOT_ the bootstrapped 90th percentile latency. It is an indication
    of the central tendency of the 90th percentile statistic.
    """
    task_namespace = "qps"
    task_name = "latency-plot"
    task_config_class = QpsPlotConfig
    public = True

    @output
    def latency_plot(self):
        return PlotTarget(self, "latency")

    @output
    def latency_stats(self):
        return Target(self, "latency-stats", ext="csv")

    def run_plot(self):
        df = self._collect_data()
        # usec to msec
        df = df.with_columns(
            pl.col("latency50") / 1000,
            pl.col("latency90") / 1000,
            pl.col("latency95") / 1000,
            pl.col("latency99") / 1000)

        self.logger.info("Compute QPS latency statistics")
        stats = self.compute_overhead(df, "latency90", how="median", overhead_scale=100)

        if self.config.drop_relative_baseline:
            view_df = stats.filter((pl.col("_metric_type") == "absolute") | ~pl.col("_is_baseline"))
        else:
            view_df = stats

        self.logger.info("Generate QPS plot")
        with PlotGrid(self.latency_plot, view_df, self.config) as grid:
            # Dump the sorted tabular data using the ordering specified by the grid config
            dump_df = grid.get_grid_df()
            dump_df.write_csv(self.latency_stats.single_path())

            if plot_config := self.config.bar_plot:
                grid.map(grid_barplot,
                         x=plot_config.tile_xaxis,
                         y="latency90",
                         err=["latency90_low", "latency90_high"],
                         config=plot_config)
            if plot_config := self.config.line_plot:
                grid.map(grid_lineplot,
                         x=plot_config.tile_xaxis,
                         y="latency90",
                         err=["latency90_low", "latency90_high"],
                         config=plot_config)
            grid.add_legend()
