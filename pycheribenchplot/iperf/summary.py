import polars as pl
import seaborn as sns

from ..core.analysis import AnalysisTask
from ..core.artefact import ValueTarget
from ..core.config import Config, config_field
from ..core.plot import PlotTarget, PlotTask, new_facet
from ..core.task import dependency, output
from ..core.tvrs import TVRSParamsMixin
from .iperf_exec import IPerfExecTask


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
    """
    task_namespace = "iperf"
    task_name = "summary-plot"
    public = True

    @dependency
    def iperf_data(self):
        return UnifiedIPerfStats(self.session, self.analysis_config)

    @output
    def summary_plot(self):
        return PlotTarget(self, "summary")

    def run_plot(self):
        df = self.iperf_data.df.get()

        ctx = self.make_param_context(df)
        # First, we aggregate along the parameterization + iterations to
        # combine per-stream information
        ctx.df = (ctx.df.group_by(["target", "variant", "runtime", "scenario", "iteration"]).agg(
            pl.col("^.*_bytes$").sum(),
            pl.col("^.*_seconds$").max(),
            pl.col("^.*_bits_per_second$").sum(),
        ))
        ctx.suppress_const_params(keep=["target", "scenario"])
        ctx.relabel(default=dict(rcv_bits_per_second="Throughput (bits/s)"))

        with self.config_plotting_context():
            with new_facet(self.summary_plot.paths(), ctx.df, col=ctx.r.scenario, col_wrap=3) as facet:
                facet.map_dataframe(sns.boxplot, x=ctx.r.target, y=ctx.r.rcv_bits_per_second)
                facet.add_legend()
