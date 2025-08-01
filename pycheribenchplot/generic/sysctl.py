from dataclasses import dataclass
from pathlib import Path

import polars as pl
import polars.selectors as cs

from ..core.artefact import PLDataFrameLoadTask, RemoteBenchmarkIterationTarget
from ..core.config import Config, config_field
from ..core.plot import PlotTarget, PlotTask, SlicePlotTask
from ..core.plot_util import (DisplayGrid, DisplayGridConfig, grid_barplot, grid_pointplot)
from ..core.task import ExecutionTask, dependency, output


@dataclass
class SysctlConfig(Config):
    names: list[str] = config_field(list, desc="List of sysctl names to collect")


class IngestSysctl(PLDataFrameLoadTask):
    task_namespace = "sysctl"
    task_name = "sysctl-ingest"

    def _load_one(self, path: Path) -> pl.DataFrame:
        """
        Load sysctl data from a file.

        This assumes that the file contains one sysctl output per line, in the form
        <sysctl_name>=<value>

        The resulting dataframe has the following columns:
         - sysctl_name: the name of the syctl from the file.
         - sysctl_value: the value associated to the sysctl.
        Note that the sysctl_value will consistently be a string, because we can't
        predict the type of each sysctl. We also don't want to preclude sampling
        of non-numeric sysctls in case further processing is done downstream.
        """
        df = pl.read_csv(path,
                         has_header=False,
                         separator="=",
                         schema={
                             "sysctl_name": pl.String,
                             "sysctl_value": pl.String
                         })
        return df


class SysctlExecTask(ExecutionTask):
    """
    Base task that hooks the benchmark execution to sample the given
    set of sysctl values before and after each benchmark iteration.

    This can either be used as a base class to inherit the sysctl sampling or
    as an auxiliary generator in the pipelien configuration.
    """
    task_namespace = "generic"
    task_name = "sysctl"
    public = True

    @output
    def sample_before(self):
        return RemoteBenchmarkIterationTarget(self, "before", loader=IngestSysctl, ext="txt")

    @output
    def sample_after(self):
        return RemoteBenchmarkIterationTarget(self, "after", loader=IngestSysctl, ext="txt")

    def run(self):
        super().run()
        self.script.setup_iteration("sysctl", template="sysctl.hook.jinja")
        self.script.teardown_iteration("sysctl", template="sysctl.hook.jinja")
        self.script.extend_context({
            "sysctl_config": self.config,
            "sysctl_gen_out_before": self.sample_before.shell_path_builder(),
            "sysctl_gen_out_after": self.sample_after.shell_path_builder(),
        })


@dataclass
class SysctlPlotConfig(DisplayGridConfig):
    """
    Base configuration for the SysctlPlotTask.

    Configure plotting of data sampled via sysctl.
    """
    tile_xaxis: str = config_field("target", desc="Parameter to use for the X axis of each tile")
    sysctl_filter: list[str] = config_field(list, desc="Display only the given sysctl prefixes")
    suppress_baseline_relative_metrics: bool = config_field(
        False,
        desc="Drop baseline data from relative plots, so that it does not appear "
        "in the plot. Note that depending on the grid configuration, this may result in "
        "a non-aligned dataframe and cause errors.")


class SysctlSlicePlotTask(SlicePlotTask):
    """
    Display a barplot of sampled sysctl data.
    """
    task_namespace = "sysctl"
    task_name = "plot-slice"
    public = True
    task_config_class = SysctlPlotConfig

    # XXX should be able to name a specific handler in config in case
    # multiple tasks inherit from TimingExecTask?
    # XXX this should really be overridden by gobal target definitions
    exec_task_class = SysctlExecTask

    #: Describe allowed data axes for output
    data_axes = ["sysctl_value"]
    #: Describe overridable auxiliary axes for plot output
    synthetic_axes = ["sysctl_name", "_metric_type"]

    @dependency
    def sysctl_before_data(self):
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(self.exec_task_class)
            yield task.sample_before.get_loader()

    @dependency
    def sysctl_after_data(self):
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(self.exec_task_class)
            yield task.sample_after.get_loader()

    @output
    def sysctl_absolute(self):
        return PlotTarget(self, "abs")

    @output
    def sysctl_relative(self):
        return PlotTarget(self, "rel")

    @output
    def sysctl_overhead(self):
        return PlotTarget(self, "ovh")

    def _is_sysctl_name_aligned(self, df) -> bool:
        return df.group_by("dataset_id").count().n_unique("count") == 1

    def _collect_sysctl_data(self):
        """
        Collect sysctl samples into a single dataframe and compute the
        relative sysctl value for each iteration.
        """
        iteration_key = [*self.param_columns_with_iter, "sysctl_name"]
        df_before = pl.concat((t.df.get() for t in self.sysctl_before_data), how="vertical")
        df_after = pl.concat((t.df.get() for t in self.sysctl_after_data), how="vertical")
        df = (df_after.join(df_before, on=iteration_key, suffix="_before").with_columns(
            pl.col("sysctl_value").cast(pl.Int64) - pl.col("sysctl_value_before").cast(pl.Int64)).select(
                cs.exclude("sysctl_value_before")))

        # Note that at this point the dataframe may not be aligned on the
        # sysctl_name column, meaning that some runs may have a different set of
        # sysctls reported.
        # We deal with this by computing all the combinations of parameter keys
        # and sysctl_name.
        if not self._is_sysctl_name_aligned(df):
            skeleton = df.select(self.key_columns_with_iter).unique()
            sysctl_set = df.select("sysctl_name").unique()
            skeleton = skeleton.join(sysctl_set, how="cross")
            df = skeleton.join(df, on=skeleton.columns, how="left")
            assert self._is_sysctl_name_aligned(df), "sysctl_name alignment failed"

        if self.config.sysctl_filter:
            df = df.filter(pl.col("sysctl_name").is_in(self.config.sysctl_filter))

        return df

    def _do_plot(self, target, view_df, default_display_name):
        name_mapping_defaults = {"sysctl_value": default_display_name}
        if self.config.hue:
            name_mapping_defaults.update({self.config.hue: self.config.hue.capitalize()})

        grid_config = self.config.set_display_defaults(param_names=name_mapping_defaults)
        with DisplayGrid(target, view_df, grid_config) as grid:
            grid.map(grid_barplot,
                     x=self.config.tile_xaxis,
                     y="sysctl_value",
                     err=["sysctl_value_low", "sysctl_value_high"])
            grid.add_legend()

    def run_plot(self):
        df = self._collect_sysctl_data()

        self.logger.info("Compute sysctl statistics")
        stats = self.compute_overhead(df,
                                      "sysctl_value",
                                      how="median",
                                      extra_groupby=["sysctl_name"],
                                      overhead_scale=100)

        self.logger.info("Plot absolute sysctl measurements")
        self._do_plot(self.sysctl_absolute, stats.filter(_metric_type="absolute"), "Value")

        self.logger.info("Plot relative sysctl measurements")
        delta_stats = stats.filter((pl.col("_metric_type") == "delta"))
        if self.config.suppress_baseline_relative_metrics:
            delta_stats = delta_stats.filter(~pl.col("_is_baseline"))
        self._do_plot(self.sysctl_relative, delta_stats, "∆ Value (s)")

        self.logger.info("Plot sysctl overhead measurements")
        ovh_stats = stats.filter((pl.col("_metric_type") == "overhead"))
        if self.config.suppress_baseline_relative_metrics:
            ovh_stats = ovh_stats.filter(~pl.col("_is_baseline"))
        self._do_plot(self.sysctl_overhead, ovh_stats, "% Overhead")
