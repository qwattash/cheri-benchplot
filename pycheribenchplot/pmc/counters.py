from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from ..core.artefact import Target, ValueTarget
from ..core.config import Config, config_field
from ..core.plot import DatasetPlotTask, PlotTarget, PlotTask, new_facet
from ..core.plot_util import DisplayGrid, DisplayGridConfig, grid_barplot
from ..core.task import dependency, output
from ..core.tvrs import TVRSParamsMixin, TVRSPlotConfig
from .pmc_exec import PMCExec, PMCExecConfig


@dataclass
class PMCPlotConfig(DisplayGridConfig):
    pmc_filter: Optional[List[str]] = config_field(None, desc="Show only the given subset of counters")
    cpu_filter: Optional[List[int]] = config_field(None, desc="Filter system-mode counters by CPU index")
    counter_group_name: Optional[str] = config_field(
        None, desc="Parameter to use to identify runs with different sets of counters")
    tile_xaxis: str = config_field("target", desc="Parameter to use for the X axis of each tile")

    def __post_init__(self):
        super().__post_init__()
        self.setdefault(tile_column="scenario")

    def resolve_counter_group(self, task: PMCExec) -> str:
        if self.counter_group_name:
            return task.benchmark.parameters[self.counter_group_name]
        else:
            return task.get_counters_loader().counter_group


class PMCStatDistribution(TVRSParamsMixin, DatasetPlotTask):
    """
    Generate a summary showing the distribution of counters data within a single dataset.
    """
    task_namespace = "pmc"
    task_name = "cnt-distribution"
    public = True

    @dependency
    def counters(self):
        task = self.benchmark.find_exec_task(PMCExec)
        return task.get_counters_loader()

    @output
    def distribution_plot(self):
        return PlotTarget(self, "distribution")

    def run_plot(self):
        exec_task = self.benchmark.find_exec_task(PMCExec)
        exec_config = exec_task.config

        df = self.counters.df.get()
        df = df.unpivot(on=[c.lower() for c in exec_config.counters_list], variable_name="counter", value_name="value")
        hist = df.group_by("counter").agg(pl.col("value").hist(bin_count=10, include_breakpoint=True))
        hist = hist.with_columns(
            pl.col("value").list.eval(pl.element().struct.field("breakpoint")).alias("hist_breakpoint"),
            pl.col("value").list.eval(pl.element().struct.field("count")).alias("hist_count"))

        grid_config = DisplayGridConfig(param_names={
            "hist_breakpoint": "Counter value",
            "hist_count": "Frequency"
        },
                                        tile_sharex=False,
                                        tile_sharey=False,
                                        tile_row="counter",
                                        tile_col="target")
        self.logger.info("Generate PMC distribution for %s: %s", self.benchmark.config.name,
                         ", ".join([f"{k}={v}" for k, v in self.benchmark.parameters.items()]))

        def format_tick(x, pos=None):
            sign = np.sign(x)
            value = abs(x)
            if value == 0:
                return "0"
            xmag = int(np.log10(value))
            xnew = sign * value / 10**xmag
            return f"${xnew:.02f}^{{{xmag}}}$"

        with DisplayGrid(self.distribution_plot, df, grid_config) as grid:

            def simple_barplot(tile, chunk):
                tile.ax.bar(chunk[tile.d.hist_breakpoint], chunk[tile.d.hist_count])
                tile.ax.set_xlabel(tile.d.hist_breakpoint)
                tile.ax.set_ylabel(tile.d.hist_count)
                tile.ax.xaxis.set_major_formatter(FuncFormatter(format_tick))

            grid.map(simple_barplot)


class PMCStatSummary(PlotTask):
    """
    Generate a summary of PMC counters for every group of counters that has been run.

    The counter set is controlled using the `PMCPlotConfig.counter_group_name` config
    option.
    """
    task_namespace = "pmc"
    task_name = "generic-summary"
    task_config_class = PMCPlotConfig
    public = True

    @dependency
    def counters(self):
        # First resolve the counters groups we have
        groups = set()
        for bench in self.session.all_benchmarks():
            task = bench.find_exec_task(PMCExec)
            groups.add(self.config.resolve_counter_group(task))

        # Now schedule a summary for each group
        for name in groups:
            yield PMCGroupSummary(self.session, self.analysis_config, task_config=self.config, pmc_group=name)

    def run(self):
        pass


class PMCGroupSummary(TVRSParamsMixin, PlotTask):
    """
    Generate a summary showing the variation of PMC counters across the dataset
    parameterisation.
    """
    task_namespace = "pmc"
    task_name = "group-summary"
    task_config_class = PMCPlotConfig

    rc_params = {"axes.labelsize": "large", "font.size": 9, "xtick.labelsize": 9}

    derived_metrics = {
        "cycles_per_insn": {
            "requires": ["inst_retired", "cpu_cycles"],
            "column": (pl.col("cpu_cycles") / pl.col("inst_retired"))
        },
        "ipc": {
            "requires": ["inst_retired", "cpu_cycles"],
            "column": (pl.col("inst_retired") / pl.col("cpu_cycles"))
        },
        "br_pred_miss_rate": {
            "requires": ["br_pred", "br_mis_pred"],
            "column": (pl.col("br_mis_pred") / pl.col("br_pred"))
        },
        "insn_spec_rate": {
            "requires": ["inst_retired", "inst_spec"],
            "column": (pl.col("inst_retired") / pl.col("inst_spec"))
        },
        "eff_backend_ipc": {
            "requires": ["inst_retired", "cpu_cycles", "stall_backend"],
            "column": pl.col("inst_retired") / (pl.col("cpu_cycles") - pl.col("stall_backend"))
        },
        "l1i_hit_rate": {
            "requires": ["l1i_cache", "l1i_cache_refill"],
            "column": (pl.col("l1i_cache") - pl.col("l1i_cache_refill")) / pl.col("l1i_cache")
        },
        "l1d_hit_rate": {
            "requires": ["l1d_cache", "l1d_cache_refill"],
            "column": (pl.col("l1d_cache") - pl.col("l1d_cache_refill")) / pl.col("l1d_cache")
        }
    }

    def __init__(self, session, analysis_config, task_config: PMCPlotConfig, pmc_group: str):
        self.pmc_group = pmc_group
        super().__init__(session, analysis_config, task_config)

    @property
    def task_id(self):
        return f"{super().task_id}-{self.pmc_group}"

    @dependency
    def counters(self):
        for bench in self.session.all_benchmarks():
            task = bench.find_exec_task(PMCExec)
            if self.config.resolve_counter_group(task) == self.pmc_group:
                yield task.get_counters_loader()

    @output
    def summary_plot(self):
        return PlotTarget(self, "summary")

    @output
    def summary_delta_plot(self):
        return PlotTarget(self, "delta")

    @output
    def summary_ovh_plot(self):
        return PlotTarget(self, "overhead")

    @output
    def summary_combined(self):
        return PlotTarget(self, "abs-with-overhead")

    @output
    def summary_tbl(self):
        return Target(self, "tbl", ext="csv")

    def gen_derived_metrics(self, df: pl.DataFrame) -> (pl.DataFrame, list[str]):
        """
        Generate derived metrics for the dataset.

        Returns a new dataframe with the available metric columns and
        a list of new column names
        """
        metrics = []
        for name, spec in self.derived_metrics.items():
            has_cols = True
            for required_column in spec["requires"]:
                if required_column not in df.columns:
                    self.logger.debug("Skip derived metric %s: requires missing column %s", name, required_column)
                    has_cols = False
                    break
            if not has_cols:
                continue
            metrics.append(name)
            df = df.with_columns((spec["column"]).alias(name))
        return df, metrics

    def run_plot(self):
        all_df = []
        metrics = set()
        for cnt_loader in self.counters:
            all_df.append(cnt_loader.df.get())
            if not metrics:
                metrics = set(cnt_loader.counter_names)
            # If this fails, we have some issue when filtering the dependencies
            assert metrics == set(cnt_loader.counter_names)

        df = pl.concat(all_df, how="vertical", rechunk=True)
        df, derived_metrics = self.gen_derived_metrics(df)
        metric_cols = [*metrics, *derived_metrics]

        self.logger.info("Handle counters group: %s", self.pmc_group)

        index_cols = [*self.param_columns, "iteration"]
        if "_cpu" in df.columns:
            index_cols = [*index_cols, "_cpu"]
        df = df.unpivot(index=index_cols, on=metric_cols, variable_name="counter", value_name="value")

        # Filter the counters based on configuration
        if self.config.pmc_filter:
            self.logger.info("Filter by PMC: %s", self.config.pmc_filter)
            df = df.filter(pl.col("counter").is_in(self.config.pmc_filter))
        if self.config.cpu_filter and "_cpu" in df.columns:
            self.logger.info("Filter by CPU index: %s", self.config.cpu_filter)
            df = df.filter(pl.col("_cpu").is_in(self.config.cpu_filter))

        if "_cpu" in df.columns:
            # Sum metrics from different CPUs, if the cpu filter is in effect,
            # we will only account for a subset of the CPUs.
            self.logger.info("Combine per-CPU counters")
            grouped = df.group_by([*self.param_columns, "counter", "iteration"])
            df = grouped.agg(pl.col("value").sum(), cs.string().first()).select(cs.exclude("_cpu"))

        self.logger.info("Bootstrap overhead confidence intervals")
        stats = self.compute_overhead(df, "value", extra_groupby=["counter"], how="median", overhead_scale=100)

        self.logger.info("Plot absolute data summary")
        median_df = stats.filter(_metric_type="absolute")
        grid_config = self.config.set_display_defaults(param_names={
            self.config.hue: self.config.hue.capitalize(),
            "value": "Counter value"
        }).set_fixed(tile_row="counter", tile_col=None)
        with DisplayGrid(self.summary_plot, median_df, grid_config) as grid:
            grid.map(grid_barplot, x=self.config.tile_xaxis, y="value", err=["value_low", "value_high"])
            grid.add_legend()

        self.logger.info("Plot delta summary")
        delta_df = stats.filter(_metric_type="delta")
        grid_config = self.config.set_display_defaults(param_names={
            self.config.hue: self.config.hue.capitalize(),
            "value": "∆ Counter value"
        }).set_fixed(tile_row="counter", tile_col=None)
        with DisplayGrid(self.summary_delta_plot, delta_df, grid_config) as grid:
            grid.map(grid_barplot, x=self.config.tile_xaxis, y="value", err=["value_low", "value_high"])
            grid.add_legend()

        self.logger.info("Plot overhead summary")
        # Note: filter out the baseline data, as it doesn't make sense to have it here
        overhead_df = stats.filter(_metric_type="overhead", _is_baseline=False)
        grid_config = self.config.set_display_defaults(param_names={
            self.config.hue: self.config.hue.capitalize(),
            "value": "% Overhead"
        }).set_fixed(tile_row="counter", tile_col=None)
        with DisplayGrid(self.summary_ovh_plot, overhead_df, grid_config) as grid:
            grid.map(grid_barplot, x=self.config.tile_xaxis, y="value", err=["value_low", "value_high"])
            grid.add_legend()

        self.logger.info("Plot combined summary")
        grid_config = self.config.set_display_defaults(param_names={
            self.config.hue: self.config.hue.capitalize(),
            "_metric_type": "Measurement"
        },
                                                       param_values={
                                                           "_metric_type": {
                                                               "absolute": "Counter value",
                                                               "delta": "∆ Counter value",
                                                               "overhead": "% Overhead"
                                                           }
                                                       }).set_fixed(tile_sharey=False,
                                                                    tile_sharex=False,
                                                                    tile_row="counter",
                                                                    tile_col="_metric_type")
        with DisplayGrid(self.summary_combined, stats, grid_config) as grid:
            grid.map(grid_barplot, x=self.config.tile_xaxis, y="value", err=["value_low", "value_high"])

            def _adjust_ylabel(tile, chunk):
                tile.ax.set_ylabel(chunk[tile.d._metric_type][0])

            grid.map(_adjust_ylabel)
            grid.add_legend()

        # Pivot back for readability
        self.logger.info("Generate tabular data")
        tbl_data = stats.pivot(columns="_metric_type", index=[*self._param_columns, "counter"], values="value")
        tbl_data.write_csv(self.summary_tbl.single_path())
