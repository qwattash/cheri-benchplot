from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import polars as pl

from ..core.artefact import PLDataFrameLoadTask, RemoteBenchmarkIterationTarget
from ..core.config import Config, config_field
from ..core.plot import PlotTarget, PlotTask, SlicePlotTask
from ..core.plot_util import (PlotGrid, PlotGridConfig, grid_barplot, grid_pointplot)
from ..core.task import ExecutionTask, dependency, output


class TimingTool(Enum):
    #: Do not collect timing
    DISABLED = "disabled"
    #: Use the UNIX time utility
    TIME = "time"
    #: Use the hyperfine tool (dependency for the benchmark host)
    HYPERFINE = "hyperfine"


@dataclass
class TimingConfig(Config):
    timing_mode: TimingTool = config_field(TimingTool.HYPERFINE, desc="Timing tool to use")


class IngestTimingStats(PLDataFrameLoadTask):
    """
    Loader for stats data that produces a standard polars dataframe.
    """
    task_namespace = "timing"
    task_name = "ingest-stats"

    def _load_one(self, path: Path) -> pl.DataFrame:
        """
        Load data for a benchmark run from the given target file.

        We only care about the actual time sample. Since we run individual
        iterations under time/hyperfine.
        In hyperfine, this means that we only use the "times" list.
        If applicable, we also check the exit codes, to make sure there were no issues.
        """
        match self.target.task.timing_config.timing_mode:
            case TimingTool.TIME:
                return self._load_time_sample(path)
            case TimingTool.HYPERFINE:
                return self._load_hyperfine_sample(path)
            case _:
                raise RuntimeError("Unexpected timing_config mode")

    def _load_time_sample(self, path: Path) -> pl.DataFrame:
        data = {}
        with open(path, "r") as sample_file:
            for line in sample_file:
                entry = line.split(" ")
                if len(entry) != 2:
                    raise RuntimeError(f"Invalid `time` output file {path}")
                key, value = entry
                data[key] = [float(value)]
        return pl.DataFrame(data).rename({"real": "times"})

    def _load_hyperfine_sample(self, path: Path) -> pl.DataFrame:
        in_df = pl.read_json(path).with_columns(
            pl.col("results").list.first().struct.field("times"),
            pl.col("results").list.first().struct.field("exit_codes"))
        df = in_df.select(
            pl.col("results").list.first().struct.field("user"),
            pl.col("results").list.first().struct.field("system"),
            pl.col("times").list.first(),
            pl.col("times").list.len().alias("check_ntimes"),
            pl.col("exit_codes").list.eval(pl.element() != 0).list.any().alias("check_errs"))
        if (df["check_ntimes"] > 1).any():
            self.logger.error("Unexpected number of hyperfine iterations")
            raise RuntimeError("Input data error")
        if df["check_errs"].any():
            self.logger.error("Found non-zero exit codes for %s at %s", self.benchmark, path)
            raise RuntimeError("Input data error")
        return df.select("user", "system", "times")


class TimingExecTask(ExecutionTask):
    """
    This task can either be used as a base task by inheriting both the configuration and the
    task, or as a separate generator in the same was as PMC.
    """
    @property
    def timing_config(self):
        return self.config

    @output
    def timing(self):
        return RemoteBenchmarkIterationTarget(self, "timing", loader=IngestTimingStats, ext="txt")

    def run(self):
        super().run()
        self.script.extend_context({
            "timing_config": self.timing_config,
            "timing_gen_output_path": self.timing.shell_path_builder()
        })
        self.script.register_global("TimingTool", TimingTool)


@dataclass
class TimingPlotConfig(PlotGridConfig):
    """
    Base configuration for the TimingPlotTask.

    This allows a wide degree of customization of the plot, locking only
    the Y axis of the tiles to be the time metric.
    """
    tile_xaxis: str = config_field("target", desc="Parameter to use for the X axis of each tile")


class TimingSlicePlotTask(SlicePlotTask):
    """
    Simple overview of the collected timing data from a generic.exec task run.
    """
    task_namespace = "timing"
    task_name = "plot-slice"
    public = True
    task_config_class = TimingPlotConfig

    # XXX should be able to name a specific handler in config in case
    # multiple tasks inherit from TimingExecTask?
    exec_task_class = TimingExecTask

    #: Describe allowed data axes for output
    data_axes = ["times"]
    #: Describe overridable axes for plot output
    synthetic_axes = ["_metric_type"]

    @dependency
    def timing(self):
        for b in self.slice_benchmarks():
            task = b.find_exec_task(self.exec_task_class, include_subclass=True)
            yield task.timing.get_loader()

    @output
    def absolute_time(self):
        return PlotTarget(self, "abstime")

    @output
    def relative_time(self):
        return PlotTarget(self, "reltime")

    @output
    def time_overhead(self):
        return PlotTarget(self, "ovhtime")

    def _collect_timing(self):
        df = pl.concat((t.df.get() for t in self.timing), how="vertical", rechunk=True)
        return df

    def _do_plot(self, target, view_df, default_display_name):
        name_mapping_defaults = {"times": default_display_name}
        if self.config.hue:
            name_defaults.update({self.config.hue: self.config.hue.capitalize()})

        grid_config = self.config.set_display_defaults(param_names=name_mapping_defaults)
        with PlotGrid(target, view_df, grid_config) as grid:
            grid.map(grid_barplot, x=self.config.tile_xaxis, y="times", err=["times_low", "times_high"])
            grid.add_legend()

    def run_plot(self):
        df = self._collect_timing()

        self.logger.info("Compute timing statistics")
        stats = self.compute_overhead(df, "times", how="median", overhead_scale=100)

        self.logger.info("Plot absolute time measurements")
        self._do_plot(self.absolute_time, stats.filter(_metric_type="absolute"), "Time (s)")

        self.logger.info("Plot relative time measurements")
        delta_stats = stats.filter((pl.col("_metric_type") == "delta") & ~pl.col("_is_baseline"))
        self._do_plot(self.relative_time, delta_stats, "∆ Time (s)")

        self.logger.info("Plot time overhead measurements")
        ovh_stats = stats.filter((pl.col("_metric_type") == "overhead") & ~pl.col("_is_baseline"))
        self._do_plot(self.time_overhead, ovh_stats, "% Overhead")


class TimingPlotTask(PlotTask):
    """
    Base task to generate a set of plots with the extracted timing information.

    This supports execution tasks that are subclasses of TimingExecTask.

    XXX DEPRECATED, use the slice task instead
    """
    task_config_class = TimingPlotConfig
    exec_task_class = None

    @property
    def timing_config(self):
        return self.config

    @dependency
    def timing(self):
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(self.exec_task_class)
            yield task.timing.get_loader()

    @output
    def absolute_time(self):
        return PlotTarget(self, "abstime")

    @output
    def relative_time(self):
        return PlotTarget(self, "reltime")

    @output
    def time_overhead(self):
        return PlotTarget(self, "ovhtime")

    def _collect_timing(self):
        df = pl.concat((t.df.get() for t in self.timing), how="vertical", rechunk=True)
        return df

    def _do_plot(self, target, view_df, default_display_name):
        grid_config = self.config.set_display_defaults(param_names={
            self.config.hue: self.config.hue.capitalize(),
            "times": default_display_name
        })
        with PlotGrid(target, view_df, grid_config) as grid:
            grid.map(grid_barplot, x=self.config.tile_xaxis, y="times", err=["times_low", "times_high"])
            grid.add_legend()

    def run_plot(self):
        df = self._collect_timing()

        self.logger.info("Compute timing statistics")
        stats = self.compute_overhead(df, "times", how="median", overhead_scale=100)

        self.logger.info("Plot absolute time measurements")
        self._do_plot(self.absolute_time, stats.filter(_metric_type="absolute"), "Time (s)")

        self.logger.info("Plot relative time measurements")
        delta_stats = stats.filter((pl.col("_metric_type") == "delta") & ~pl.col("_is_baseline"))
        self._do_plot(self.relative_time, delta_stats, "∆ Time (s)")

        self.logger.info("Plot time overhead measurements")
        ovh_stats = stats.filter((pl.col("_metric_type") == "overhead") & ~pl.col("_is_baseline"))
        self._do_plot(self.time_overhead, ovh_stats, "% Overhead")
