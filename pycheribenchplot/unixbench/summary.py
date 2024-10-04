from dataclasses import dataclass
from typing import Optional

import polars as pl
import seaborn as sns

from ..core.analysis import AnalysisTask
from ..core.artefact import ValueTarget
from ..core.config import config_field
from ..core.plot import PlotTarget, PlotTask
from ..core.plot_util import DisplayGrid, DisplayGridConfig, grid_pointplot
from ..core.task import dependency, output
from ..core.tvrs import TVRSParamsMixin, TVRSPlotConfig
from .unixbench_exec import UnixBenchExec


@dataclass
class UnixBenchSummaryConfig(DisplayGridConfig):
    def __post_init__(self):
        super().__post_init__()
        # Set defaults for display grid parameters
        self.setdefault(tile_row="runtime", hue="target")


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

    def run_plot(self):
        df = self.data.df.get()

        # Transition to the new stats infrastructure
        stats = self.compute_overhead(df, "times", how="median")

        grid_config = self.config.set_display_defaults(param_names={
            self.config.hue: self.config.hue.capitalize(),
            "times": "Time (s)"
        })

        median_df = stats.filter(_metric_type="absolute")
        with DisplayGrid(self.summary, median_df, grid_config) as grid:
            grid.map(grid_pointplot, x="scenario", y="times", err=["times_low", "times_high"])
            grid.add_legend()

        grid_config = self.config.set_display_defaults(param_names={
            self.config.hue: self.config.hue.capitalize(),
            "times": "% Time Overhead"
        })

        ovh_df = stats.filter(_metric_type="overhead")
        with DisplayGrid(self.overhead, ovh_df, grid_config) as grid:
            grid.map(grid_pointplot, x="scenario", y="times", err=["times_low", "times_high"])
            grid.add_legend()
