from dataclasses import dataclass

import polars as pl
import polars.selectors as cs
import seaborn as sns

from ..core.analysis import AnalysisTask
from ..core.artefact import ValueTarget
from ..core.config import Config, config_field
from ..core.plot import PlotTarget, PlotTask, new_facet
from ..core.task import dependency, output
from ..core.tvrs import TVRSParamsMixin, TVRSTaskConfig
from .iperf_exec import IPerfExecTask


@dataclass
class IPerfSummaryConfig(TVRSTaskConfig):
    tile_parameter: str = config_field("scenario", desc="Parameter axis to use for the facet grid")


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

    def _get_agg_stream_stats(self):
        """
        Produce a parameterization context with aggregated data from all iperf streams.
        """
        df = self.iperf_data.df.get()
        ctx = self.make_param_context(df.clone())
        # First, we aggregate along the parameterization + iterations to
        # combine per-stream information
        ctx.df = (ctx.df.group_by(["target", "variant", "runtime", "scenario", "iteration"]).agg(
            pl.col("dataset_id").first(),
            cs.by_name(ctx.extra_params).first(),
            pl.col("^.*_bytes$").sum(),
            pl.col("^.*_seconds$").max(),
            pl.col("^.*_bits_per_second$").sum(),
        ))
        ctx.suppress_const_params(keep=["target", "scenario"])
        ctx.derived_hue_param(default=["variant", "runtime"])
        return ctx

    def _plot_summary(self):
        ctx = self._get_agg_stream_stats()
        ctx.relabel(default=dict(
            _hue="Variant",
            rcv_bits_per_second="Throughput (bits/s)",
        ))
        ctx.sort()

        hue_kwargs = dict()
        if ctx.r._hue:
            hue_kwargs.update(dict(palette=ctx.build_palette_for("_hue"), hue=ctx.r._hue))
        facet_col = ctx.r[self.config.tile_parameter]

        with new_facet(self.summary_plot.paths(), ctx.df, col=facet_col, col_wrap=3) as facet:
            facet.map_dataframe(sns.boxplot, x=ctx.r.target, y=ctx.r.rcv_bits_per_second, **hue_kwargs)
            facet.add_legend()

    def _plot_overhead(self):
        ctx = self._get_agg_stream_stats()
        ctx.compute_overhead(["rcv_bits_per_second"])
        ctx.relabel(default=dict(_hue="Variant", rcv_bits_per_second_overhead="% Throughput"))
        ctx.sort()

        hue_kwargs = dict()
        if ctx.r._hue:
            hue_kwargs.update(dict(palette=ctx.build_palette_for("_hue"), hue=ctx.r._hue))
        facet_col = ctx.r[self.config.tile_parameter]

        with new_facet(self.summary_overhead_plot.paths(), ctx.df, col=facet_col, col_wrap=3) as facet:
            facet.map_dataframe(sns.boxplot, x=ctx.r.target, y=ctx.r.rcv_bits_per_second_overhead, **hue_kwargs)
            facet.add_legend()

    def run_plot(self):
        with self.config_plotting_context():
            self._plot_summary()
        with self.config_plotting_context():
            self._plot_overhead()
