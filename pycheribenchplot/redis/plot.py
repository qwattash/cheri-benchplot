from dataclasses import dataclass
import polars as pl

from ..core.analysis import DependentVariableConfig
from ..core.config import config_field
from ..core.plot import PlotTarget, SlicePlotTask
from ..core.plot_grid import BarPlotConfig, PlotGrid, PlotGridConfig, grid_barplot
from ..core.task import dependency, output
from .exec import RedisExecTask


@dataclass
class RedisPlotConfig(DependentVariableConfig, PlotGridConfig, BarPlotConfig):
    """
    Configuration for Redis benchmark bar plots.
    """

    drop_baseline_from_relative: bool = config_field(
        True,
        desc="Omit the baseline row from delta and overhead metric views.",
    )
    dependent_variable: str = "requests_per_second"  #: override


class RedisSlicePlotTask(SlicePlotTask):
    """
    Bar plot of Redis benchmark metrics, split by the configured fixed axes.
    """

    task_namespace = "redis"
    task_name = "bar-plot-slice"
    public = True
    task_config_class = RedisPlotConfig

    @dependency
    def data(self):
        for bench in self.slice_benchmarks:
            task = bench.find_exec_task(RedisExecTask)
            yield task.stats.get_loader()

    @output
    def plot(self):
        return PlotTarget(self, "metric")

    def run_plot(self):
        # 1. Collect raw data from all loaders in this slice.
        df = pl.concat(
            [loader.df.get() for loader in self.data],
            how="vertical",
            rechunk=True,
        )

        # 2. Compute statistics (absolute / delta / overhead) using the configured dependent variable.
        depvar = self.get_depvar_column()
        stats = self.compute_overhead(df, depvar, how="median", overhead_scale=100)

        # 3. Optionally drop the baseline row from relative views.
        if self.config.drop_baseline_from_relative:
            view_df = stats.filter(
                (pl.col("_metric_type") == "absolute") | ~pl.col("_is_baseline")
            )
        else:
            view_df = stats

        # 4. Render the plot via PlotGrid.
        with PlotGrid(self.plot, view_df, self.config) as grid:
            grid.map(
                grid_barplot,
                x=self.config.tile_xaxis,
                y=depvar,
                err=[f"{depvar}_low", f"{depvar}_high"],
                config=self.config,
            )
            grid.add_legend()
