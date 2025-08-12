from dataclasses import dataclass

import polars as pl
import polars.selectors as cs

from ..core.artefact import Target
from ..core.config import Config, config_field
from ..core.plot import PlotTarget, SlicePlotTask
from ..core.plot_util import DisplayGrid, DisplayGridConfig, grid_barplot
from ..core.task import dependency, output
from .pmc_exec import IngestPMCCounters, PMCExec


@dataclass
class PMCPlotConfig(DisplayGridConfig):
    pmc_filter: list[str] | None = config_field(None, desc="Show only the given subset of counters")
    cpu_filter: list[int] | None = config_field(None, desc="Filter system-mode counters by CPU index")
    tile_xaxis: str = config_field(Config.REQUIRED, desc="Parameter to use for the X axis of each tile")

    def resolve_counter_group(self, task: PMCExec) -> str:
        return task.get_counters_loader().counter_group


class PMCSliceSummary(SlicePlotTask):
    """
    Generate a summary showing the variation of PMC counters across an arbitrary
    parameterisation.

    This will combine all the benchmark runs within the slice into a single frame.
    If counters group is parameterised by a free axis in the slice, there may be
    duplicate counter columns in the dataframe. In order to avoid this, we will
    prefix the columns with the counter group name.
    The first group we encounter is treated as the default group and the counter
    names will be unprefixed.

    Derived metrics are computed only once, for the default counter group.
    If the goal is comparing derived metrics across runs that use different counter
    groups, it is advised to use the counter group parameterisation axis as a
    fixed axis in the dynamic analysis top-level task.
    """
    task_namespace = "pmc"
    task_name = "summary"
    task_config_class = PMCPlotConfig
    public = True

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

    @dependency
    def counters(self):
        for bench in self.slice_benchmarks:
            task = bench.find_exec_task(PMCExec)
            yield task.get_counters_loader()

    @output
    def summary_combined(self):
        return PlotTarget(self, "combined")

    @output
    def summary_tbl(self):
        return Target(self, "table", ext="csv")

    def _gen_derived_metrics(self, df: pl.DataFrame) -> (pl.DataFrame, list[str]):
        """
        Generate derived metrics for the slice.

        Expec the input dataframe to be in wide form and have a constant _counter_group axis.
        Note that we avoid computing derived metrics across different _counter_group values,
        because these are captured in different executions and have weaker correlation
        depending on the overall probe effect.
        We generate a new column for each derived metric and the list of generated columns
        is returned together with the dataframe.
        """
        derived = []
        for name, spec in self.derived_metrics.items():
            has_cols = True
            for required_column in spec["requires"]:
                if required_column not in df.columns:
                    self.logger.debug("Skip derived metric %s: requires missing column %s", name, required_column)
                    has_cols = False
                    break
            if not has_cols:
                continue
            derived.append(name)
            df = df.with_columns((spec["column"]).alias(name))
        return df, derived

    def _collect_metrics(self, loader: IngestPMCCounters) -> pl.DataFrame:
        """
        The loader produces a dataframe in wide form; we convert this in long
        form to make it easier to merge counter data from etherogeneous
        counter groups.

        We compute derived metrics here with the assumption that the loader
        provides data that is associated to a single _counter_group.
        Wide form data makes this easier to do.
        """
        df = loader.df.get()
        assert df.n_unique("_counter_group") == 1, "Loader has data from multiple counter groups"
        df, derived_metrics = self._gen_derived_metrics(df)
        metrics = [*loader.counter_names, *derived_metrics]
        index_cols = [*self.key_columns_with_iter, "_cpu"]
        df = df.unpivot(on=metrics, index=index_cols, variable_name="_counter", value_name="value")
        return df

    def setup_plot(self):
        all_df = []
        for loader in self.counters:
            all_df.append(self._collect_metrics(loader))

        self.logger.info("Prepare plot data for slice %s", self.slice_info)

        # Note that when merging we may have repeating _counter values for different
        # runs, belonging to different _counter_group slices. We must consider the
        # tuple (_counter_group, _counter) to be the real identifier for a metric.
        df = pl.concat(all_df, how="vertical", rechunk=True)

        # Filter the counters based on configuration
        if self.config.pmc_filter:
            self.logger.info("Filter by PMC: %s", self.config.pmc_filter)
            df = df.filter(pl.col("_counter").is_in(self.config.pmc_filter))
        if self.config.cpu_filter:
            self.logger.info("Filter by CPU index: %s", self.config.cpu_filter)
            df = df.filter(pl.col("_cpu").is_in(self.config.cpu_filter))

        # Sum metrics from different CPUs, if the cpu filter is in effect,
        # we will only account for a subset of the CPUs.
        self.logger.info("Combine per-CPU counters %s", df["_cpu"].unique().to_list())
        df = df.group_by(["dataset_id", "_counter", "iteration"]).agg(
            pl.col("value").sum(),
            cs.exclude("value").first(),
        ).select(cs.exclude("_cpu"))
        # XXX Would be nice to check the integrity of this, i.e. we are not aggregating
        # unrelated things.

        self.logger.info("Bootstrap overhead confidence intervals")
        self.stats = self.compute_overhead(df, "value", extra_groupby=["_counter"], how="median", overhead_scale=100)

    def run_plot(self):
        self.logger.info("Plot combined PMC summary for slice %s", self.slice_info)
        # Setup default grid configuration
        default_axis_names = {
            "_metric_type": "Measurement",
            "_counter": "Counter",
        }
        if self.config.hue:
            default_axis_names[self.config.hue] = self.config.hue.capitalize()
        default_metric_labels = {"absolute": "Counter value", "delta": "∆ Counter value", "overhead": "% Overhead"}

        grid_config = self.config.with_default_axis_rename(default_axis_names)
        grid_config = grid_config.with_default_axis_remap({"_metric_type": default_metric_labels})
        grid_config = grid_config.with_config_default(tile_sharey=False,
                                                      tile_sharex=False,
                                                      tile_row="_counter",
                                                      tile_col="_metric_type")
        # Draw the barplot but adjust the Y label to reflect the tile metric type
        with DisplayGrid(self.summary_combined, self.stats, grid_config) as grid:
            grid.map(grid_barplot, x=self.config.tile_xaxis, y="value", err=["value_low", "value_high"])

            # XXX this seems useful, integrate it in the base plot grid
            def _adjust_ylabel(tile, chunk):
                tile.ax.set_ylabel(chunk[tile.d._metric_type][0])

            grid.map(_adjust_ylabel)
            grid.add_legend()

        # Pivot back for readability
        self.logger.info("Generate tabular data for slice %s", self.slice_info)
        tbl_data = self.stats.pivot(columns="_metric_type", index=[*self._param_columns, "_counter"], values="value")
        tbl_data.write_csv(self.summary_tbl.single_path())


class PMCSliceAbsSummary(PMCSliceSummary):
    """
    Generate a summary showing the absolute value of the counters across an arbitrary
    parameterisation.

    By default this will tile the rows with the _counter axis.
    """
    task_namespace = "pmc"
    task_name = "summary-abs"
    task_config_class = PMCPlotConfig
    public = True

    @output
    def summary_plot(self):
        return PlotTarget(self, "summary")

    def run_plot(self):
        self.logger.info("Plot absolute counters summary for slice %s", self.slice_info)
        median_df = self.stats.filter(_metric_type="absolute")
        default_axis_names = {
            "_counter": "Counter",
            "value": "Counter value",
        }
        if self.config.hue:
            default_axis_names[self.config.hue] = self.config.hue.capitalize()
        grid_config = self.config.with_default_axis_rename(default_axis_names).with_config_default(tile_row="_counter")

        with DisplayGrid(self.summary_plot, median_df, grid_config) as grid:
            grid.map(grid_barplot, x=self.config.tile_xaxis, y="value", err=["value_low", "value_high"])
            grid.add_legend()


class PMCSliceRelSummary(PMCSliceSummary):
    """
    Generate a summary showing the relative delta value of the counters across an arbitrary
    parameterisation.

    By default this will tile the rows with the _counter axis.
    """
    task_namespace = "pmc"
    task_name = "summary-delta"
    task_config_class = PMCPlotConfig
    public = True

    @output
    def summary_delta_plot(self):
        return PlotTarget(self, "delta")

    def run_plot(self):
        self.logger.info("Plot counters delta summary for slice %s", self.slice_info)
        delta_df = self.stats.filter(_metric_type="delta")
        default_axis_names = {
            "_counter": "Counter",
            "value": "∆ Counter value",
        }
        if self.config.hue:
            default_axis_names[self.config.hue] = self.config.hue.capitalize()
        grid_config = self.config.with_default_axis_rename(default_axis_names).with_config_default(tile_row="_counter")

        with DisplayGrid(self.summary_delta_plot, delta_df, grid_config) as grid:
            grid.map(grid_barplot, x=self.config.tile_xaxis, y="value", err=["value_low", "value_high"])
            grid.add_legend()


class PMCSliceOverheadSummary(PMCSliceSummary):
    """
    Generate a summary showing the relative overhead value of the counters across an arbitrary
    parameterisation.

    By default this will tile the rows with the _counter axis.
    """
    task_namespace = "pmc"
    task_name = "summary-overhead"
    task_config_class = PMCPlotConfig
    public = True

    @output
    def summary_ovh_plot(self):
        return PlotTarget(self, "overhead")

    def run_plot(self):
        self.logger.info("Plot counters overhead summary for slice %s", self.slice_info)
        # Note: filter out the baseline data, as it doesn't make sense to have it here
        overhead_df = self.stats.filter(_metric_type="overhead", _is_baseline=False)
        default_axis_names = {
            "_counter": "Counter",
            "value": "% Overhead",
        }
        if self.config.hue:
            default_axis_names[self.config.hue] = self.config.hue.capitalize()
        grid_config = self.config.with_default_axis_rename(default_axis_names).with_config_default(tile_row="_counter")

        with DisplayGrid(self.summary_ovh_plot, overhead_df, grid_config) as grid:
            grid.map(grid_barplot, x=self.config.tile_xaxis, y="value", err=["value_low", "value_high"])
            grid.add_legend()
