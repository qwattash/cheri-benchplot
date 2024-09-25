from dataclasses import dataclass

import polars as pl
import polars.selectors as cs
import seaborn as sns

from ..core.analysis import AnalysisTask
from ..core.artefact import ValueTarget
from ..core.config import Config, config_field
from ..core.plot import PlotTarget, PlotTask
from ..core.plot_grid import DisplayGrid, DisplayGridConfig
from ..core.plot_util import boxplot
from ..core.task import dependency, output
from ..core.tvrs import TVRSParamsMixin, TVRSPlotConfig
from .iperf_exec import IPerfExecTask


@dataclass
class _IPerfSummaryConfig(TVRSPlotConfig):
    tile_parameter: str = config_field("scenario", desc="Parameter axis to use for the facet grid")
    tile_aspect: float = config_field(1.0, desc="Tile aspect ratio")


@dataclass
class IPerfSummaryConfig(DisplayGridConfig):
    def __post_init__(self):
        super().__post_init__()
        # Set IPerf plot defaults and fixed values


class UnifiedIPerfStats(AnalysisTask):
    task_namespace = "iperf"
    task_name = "combine"

    @dependency
    def ingest(self):
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(IPerfExecTask)
            yield task.stats.get_loader()

    @output
    def df(self):
        return ValueTarget(self, "combined")

    def run(self):
        df = pl.concat((t.df.get() for t in self.ingest), how="vertical", rechunk=True)
        self.df.assign(df)


class IPerfSummaryPlot(TVRSParamsMixin, PlotTask):
    """
    Generate a box plot showing the aggregate bitrate throughput
    for each scenario.

    The target is used as the categorical X axis. Boxes are aggregated by target
    and each bar group shows combinations of variant/runtime parameterization,
    if any.

    Param columns for renaming:
    - target, variant, runtime, scenario
    - _hue: derived hue parameter, combines variant/runtime by default.

    Metric columns for renaming:
    - rcv_bits_per_second{_overhead}: The throughput as measured on the receiver
    """
    task_namespace = "iperf"
    task_name = "summary-plot"
    task_config_class = IPerfSummaryConfig
    public = True

    @dependency
    def iperf_data(self):
        return UnifiedIPerfStats(self.session, self.analysis_config)

    @output
    def summary_plot(self):
        return PlotTarget(self, "summary")

    @output
    def summary_overhead_plot(self):
        return PlotTarget(self, "summary-overhead")

    def _get_data(self):
        """
        Get the dataframe with derived columns.

        Add a new column 'side' that encodes the 'sender' column as a string
        instead of a boolean.
        """
        df = self.iperf_data.df.get()
        df = df.with_columns(
            pl.when(pl.col("sender") == True).then(pl.lit("sender")).otherwise(pl.lit("receiver")).alias("side"))
        return df

    def _plot_summary(self):
        df = self._get_data()
        df = df.with_columns((pl.col("bits_per_second") / 2**23).alias("MiB_per_second"))

        grid_config = self.config.set_display_defaults(param_names={
            self.config.hue: "Variant",
            "MiB_per_second": "Throughput (MiB/s)"
        })
        with DisplayGrid(self.summary_plot, df, grid_config) as grid:

            def _doplot(tile, chunk):
                sns.stripplot(chunk,
                              x=tile.d.block_size_bytes,
                              y=tile.d.MiB_per_second,
                              hue=tile.d[self.config.hue],
                              palette=tile.palette,
                              ax=tile.ax,
                              legend=False,
                              dodge=True)

            grid.map(_doplot)
            grid.add_legend()

        # ctx = self._get_agg_stream_stats()
        # ctx.df = ctx.df.with_columns(pl.col("rcv_bits_per_second") / (8 * 2**20))
        # ctx.relabel(default=dict(
        #     _hue="Variant",
        #     rcv_bits_per_second="Throughput (MiB/s)",
        # ))
        # ctx.sort()

        # hue_kwargs = dict()
        # if ctx.r._hue:
        #     hue_kwargs.update(dict(palette=ctx.build_palette_for("_hue"), hue=ctx.r._hue))
        # facet_col = ctx.r[self.config.tile_parameter]

        # with new_facet(self.summary_plot.paths(), ctx.df, col=facet_col, col_wrap=3) as facet:
        #     facet.map_dataframe(sns.boxplot, x=ctx.r.target, y=ctx.r.rcv_bits_per_second, **hue_kwargs)
        #     facet.add_legend()

    def _plot_overhead(self):
        pass
        # ctx = self._get_agg_stream_stats()
        # ctx = ctx.compute_overhead(["rcv_bits_per_second"])
        # ctx.relabel(default=dict(_hue="Variant", rcv_bits_per_second_overhead="% Throughput"))
        # ctx.sort()

        # hue_kwargs = dict()
        # if ctx.r._hue:
        #     hue_kwargs.update(dict(palette=ctx.build_palette_for("_hue"), hue=ctx.r._hue))
        # facet_col = ctx.r[self.config.tile_parameter]

        # with new_facet(self.summary_overhead_plot.paths(), ctx.df, col=facet_col, col_wrap=3) as facet:
        #     facet.map_dataframe(sns.boxplot, x=ctx.r.target, y=ctx.r.rcv_bits_per_second_overhead, **hue_kwargs)
        #     facet.add_legend()

    def run_plot(self):
        self._plot_summary()
        self._plot_overhead()
