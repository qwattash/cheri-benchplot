from dataclasses import dataclass
from typing import Optional

import polars as pl
import seaborn as sns

from ..core.analysis import AnalysisTask
from ..core.artefact import ValueTarget
from ..core.config import config_field
from ..core.plot import PlotTarget, PlotTask, new_facet
from ..core.task import dependency, output
from ..core.tvrs import TVRSParamsMixin, TVRSPlotConfig
from .unixbench_exec import UnixBenchExec


@dataclass
class UnixBenchSummaryConfig(TVRSPlotConfig):
    tile_column_name: Optional[str] = config_field("runtime", desc="Parameter to use for column tiling, may be null")
    tile_row_name: Optional[str] = config_field("variant", desc="Parameter to use for row tiling, may be null")


class UnixBenchLoad(AnalysisTask):
    """
    Load all unixbench run data.
    """
    task_namespace = "unixbench"
    task_name = "load"

    @dependency
    def data(self):
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(UnixBenchExec)
            yield task.timing.get_loader()

    @output
    def df(self):
        return ValueTarget(self, "all-data")

    def run(self):
        df = pl.concat((t.df.get() for t in self.data), how="vertical", rechunk=True)
        self.df.assign(df)


class UnixBenchSummaryPlot(TVRSParamsMixin, PlotTask):
    """
    Generate a plot for each Unixbench scenario.
    """
    task_namespace = "unixbench"
    task_name = "summary"
    task_config_class = UnixBenchSummaryConfig
    public = True

    @dependency
    def data(self):
        return UnixBenchLoad(self.session, self.analysis_config)

    @output
    def summary(self):
        return PlotTarget(self, "summary")

    @output
    def overhead(self):
        return PlotTarget(self, "summary-overhead")

    def gen_summary(self, ctx):
        facet_col = ctx.r[self.config.tile_column_name]
        facet_row = ctx.r[self.config.tile_row_name]
        palette = None
        if ctx.r._hue:
            palette = ctx.build_palette_for("_hue")

        with new_facet(self.summary.paths(), ctx.df, col=facet_col, row=facet_row, margin_titles=True) as facet:
            facet.map_dataframe(sns.boxplot, x=ctx.r.scenario, y=ctx.r.times, hue=ctx.r._hue, gap=.1, palette=palette)
            facet.add_legend()
            self.adjust_legend_on_top(facet.figure)

    def gen_overhead(self, ctx):
        facet_col = ctx.r[self.config.tile_column_name]
        facet_row = ctx.r[self.config.tile_row_name]
        palette = None
        if ctx.r._hue:
            palette = ctx.build_palette_for("_hue")

        with new_facet(self.overhead.paths(), ctx.df, col=facet_col, row=facet_row, margin_titles=True) as facet:
            facet.map_dataframe(sns.boxplot,
                                x=ctx.r.scenario,
                                y=ctx.r.times_overhead,
                                hue=ctx.r._hue,
                                gap=.1,
                                palette=palette)
            facet.add_legend()
            self.adjust_legend_on_top(facet.figure)

    def run_plot(self):
        df = self.data.df.get()
        ctx = self.make_param_context(df)
        ctx.suppress_const_params(keep=["target", "scenario"])
        ovh_ctx = ctx.compute_overhead(["times"])
        ctx.relabel(default=dict(times="Time (s)"))
        ovh_ctx.relabel(default=dict(times_overhead="% Run-time overhead"))

        # Hue parameter is subject to relabeling, so do this last
        ctx.derived_hue_param(default=["target"])
        ctx.relabel(default=dict(_hue="Target"))
        ovh_ctx.derived_hue_param(default=["target"])
        ovh_ctx.relabel(default=dict(_hue="Target"))

        self.logger.info("Generate UnixBench summary plots")
        with self.config_plotting_context():
            self.gen_summary(ctx)
            self.gen_overhead(ovh_ctx)
