import polars as pl
import seaborn as sns

from ..core.analysis import AnalysisTask
from ..core.artefact import ValueTarget
from ..core.plot import PlotTarget, PlotTask, new_figure
from ..core.task import dependency, output
from ..core.tvrs import TVRSParamsMixin, TVRSTaskConfig
from .unixbench_exec import UnixBenchExec


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

    def run_plot(self):
        df = self.data.df.get()
        ctx = self.make_param_context(df)
        ctx.suppress_const_params(keep=["target", "scenario"])
        ctx.derived_hue_param(default=["target"])
        ovh_ctx = ctx.compute_overhead(["times"])

        ctx.relabel(default=dict(times="Time (s)", _hue="Target"))
        ovh_ctx.relabel(default=dict(times="% Run-time overhead", _hue="Target"))

        self.logger.info("Generate UnixBench summary plots")
        with self.config_plotting_context():
            with new_figure(self.summary.paths()) as fig:
                ax = fig.subplots()
                sns.boxplot(ctx.df, ax=ax, x=ctx.r.scenario, y=ctx.r.times, hue=ctx.r._hue, gap=.1)

            with new_figure(self.overhead.paths()) as fig:
                ax = fig.subplots()
                sns.boxplot(ovh_ctx.df, ax=ax, x=ovh_ctx.r.scenario, y=ovh_ctx.r.times, hue=ovh_ctx.r._hue, gap=.1)
