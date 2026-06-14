from dataclasses import dataclass
from enum import Enum

import polars as pl

from ..core.artefact import Target
from ..core.config import Config, config_field
from ..core.plot import PlotTarget, SlicePlotTask
from ..core.plot_grid import BarPlotConfig, PlotGrid, PlotGridConfig, grid_barplot
from ..core.task import dependency, output
from .iperf_exec import IPerfExecTask


class ThroughputUnit(Enum):
    """
    Unit to use when displaying throughput values.

    The divisor is applied to bits/s values to convert them to the target unit.
    """

    bps = "bps"
    Kbps = "Kbps"
    Mbps = "Mbps"
    Gbps = "Gbps"
    KiB_s = "KiB/s"
    MiB_s = "MiB/s"
    GiB_s = "GiB/s"

    @property
    def divisor(self) -> float:
        match self:
            case ThroughputUnit.bps:
                return 1
            case ThroughputUnit.Kbps:
                return 1e3
            case ThroughputUnit.Mbps:
                return 1e6
            case ThroughputUnit.Gbps:
                return 1e9
            case ThroughputUnit.KiB_s:
                return 8 * 2**10
            case ThroughputUnit.MiB_s:
                return 8 * 2**20
            case ThroughputUnit.GiB_s:
                return 8 * 2**30


@dataclass
class IPerfSummaryConfig(PlotGridConfig):
    """
    Configure the iperf throughput bar plot.

    The bar_plot configuration controls the X axis and other bar plot parameters.
    The throughput metric is always on the Y axis; the column ref <throughput> can
    be used as tile_yaxis (default) or in derived_columns.

    Metric columns available as column refs:
    - <throughput>: Throughput in the unit selected by throughput_unit.
    - <_metric_type>: One of "absolute", "delta" or "overhead".
    """

    bar_plot: BarPlotConfig = config_field(
        Config.REQUIRED, desc="Bar plot configuration."
    )
    drop_relative_baseline: bool = config_field(
        True,
        desc="Drop baseline rows from delta/overhead metric views.",
    )
    throughput_unit: ThroughputUnit = config_field(
        ThroughputUnit.MiB_s,
        by_value=True,
        desc="Unit for throughput values. Applied after overhead computation.",
    )


class IPerfSummaryPlot(SlicePlotTask):
    """
    Generate a bar plot showing the receiver-side bitrate throughput.

    Statistics (median + bootstrap CI) are computed across iterations and
    expressed both as absolute values and as overhead relative to the
    configured baseline.  The throughput unit is configurable via
    ``throughput_unit``; scaling is applied after the overhead computation so
    that the raw bits/s values are used for statistical inference.

    Available column refs for plot configuration:
    - <throughput>: Scaled throughput value (unit set by throughput_unit).
    - <_metric_type>: Metric kind — "absolute", "delta", or "overhead".
      Use tile_row or tile_col to split the plot by metric kind.
    """

    task_namespace = "iperf"
    task_name = "summary-plot"
    task_config_class = IPerfSummaryConfig
    public = True

    @dependency
    def iperf_data(self):
        for bench in self.slice_benchmarks:
            task = bench.find_exec_task(IPerfExecTask)
            yield task.stats.get_loader()

    @output
    def summary_plot(self):
        return PlotTarget(self, "summary")

    @output
    def summary_stats(self):
        return Target(self, "summary-stats", ext="csv")

    def _collect_data(self) -> pl.DataFrame:
        return pl.concat(
            [loader.df.get() for loader in self.iperf_data],
            how="vertical",
            rechunk=True,
        )

    def run_plot(self):
        df = self._collect_data()

        # Use receiver-side throughput only.
        df = df.filter(pl.col("side") == "receiver")

        self.logger.info("Compute iperf throughput statistics")
        stats = self.compute_overhead(
            df, "bits_per_second", how="median", overhead_scale=100
        )

        # Scale absolute and delta metric types to the configured throughput unit.
        # Overhead is a dimensionless percentage — leave it unchanged.
        divisor = self.config.throughput_unit.divisor
        scale_cond = pl.col("_metric_type") != "overhead"
        stats = stats.with_columns(
            pl.when(scale_cond)
            .then(pl.col("bits_per_second") / divisor)
            .otherwise(pl.col("bits_per_second"))
            .alias("throughput"),
            pl.when(scale_cond)
            .then(pl.col("bits_per_second_low") / divisor)
            .otherwise(pl.col("bits_per_second_low"))
            .alias("throughput_low"),
            pl.when(scale_cond)
            .then(pl.col("bits_per_second_high") / divisor)
            .otherwise(pl.col("bits_per_second_high"))
            .alias("throughput_high"),
        )

        if self.config.drop_relative_baseline:
            view_df = stats.filter(
                (pl.col("_metric_type") == "absolute") | ~pl.col("_is_baseline")
            )
        else:
            view_df = stats

        self.logger.info("Generate iperf throughput plot")
        with PlotGrid(self.summary_plot, view_df, self.config) as grid:
            dump_df = grid.get_grid_df()
            dump_df.write_csv(self.summary_stats.single_path())

            plot_config = self.config.bar_plot.with_config_default(
                tile_yaxis="<throughput>"
            )
            grid.map(
                grid_barplot,
                x=plot_config.tile_xaxis,
                y=plot_config.tile_yaxis,
                err=["throughput_low", "throughput_high"],
                config=plot_config,
            )
            grid.add_legend()
