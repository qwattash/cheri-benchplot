from dataclasses import dataclass
from typing import List, Optional

import polars as pl
import polars.selectors as cs
import seaborn as sns

from ..core.analysis import AnalysisTask
from ..core.artefact import ValueTarget
from ..core.config import config_field
from ..core.plot import PlotTarget, PlotTask, new_facet
from ..core.task import dependency, output
from ..core.tvrs import TVRSParamsMixin, TVRSPlotConfig
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
class NetperfPlotConfig(TVRSPlotConfig):
    tile_parameter: Optional[str] = config_field(None, desc="Parameter to use for the facet column tiles")
    netperf_metrics: List[str] = config_field(lambda: ["throughput", "mean latency microseconds"],
                                              desc="Netperf metrics to plot")
    tile_aspect: float = config_field(1.0, desc="Aspect ratio of grid tiles")


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

        ctx = self.make_param_context(df)
        ctx.melt(extra_id_vars=["iteration"],
                 value_vars=self.config.netperf_metrics,
                 variable_name="_metric",
                 value_name="value")
        ctx.derived_hue_param(default=["variant", "runtime"])
        ctx.relabel(default={"_metric": "Metric"})
        ctx.sort()

        palette = ctx.build_palette_for("_hue", allow_empty=False)
        with new_facet(self.summary.paths(),
                       ctx.df,
                       row=ctx.r._metric,
                       margin_titles=True,
                       aspect=self.config.tile_aspect,
                       sharey="row",
                       sharex=True) as facet:
            facet.map_dataframe(sns.boxplot, x=ctx.r.target, y=ctx.r.value, hue=ctx.r._hue, palette=palette)
            facet.add_legend()
            self.adjust_legend_on_top(facet.figure)
