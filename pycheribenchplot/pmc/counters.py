from dataclasses import dataclass

import polars as pl
import polars.selectors as cs

from ..core.artefact import Target
from ..core.config import config_field
from ..core.plot import PlotTarget, SlicePlotTask
from ..core.plot_grid import (
    BarPlotConfig,
    OptColRef,
    PlotGrid,
    PlotGridConfig,
    grid_barplot,
)
from ..core.task import dependency, output
from .pmc_exec import IngestPMCCounters, PMCExec


@dataclass
class PMCPlotConfig(PlotGridConfig, BarPlotConfig):
    """
    Configure a performance counter plot.

    The pmc_filter property is used to select only a subset of the recorded
    counters (including derived metrics).
    Note that a similar result can be accomplished by using the :attr:`GenericAnalysisConfig.param_filter`
    at the top level, however this is more flexible as it allows filtering when
    the workload configuration uses counter groups.

    The cpu_filter property is used to filter system-wide counters by CPU.
    This is useful to further isolate the data if processes are pinned.
    """

    tile_sharex: str | bool = False  #: override
    tile_sharey: str | bool = False  #: override
    tile_row: OptColRef = "<_counter>"  #: override
    tile_col: OptColRef = "<_metric_type>"  #: override

    # XXX it would be nice to have a clean default configuration, it is
    # unclear how much value there is into it.
    # default_axis_names = {
    #     "_metric_type": "Measurement",
    #     "_counter": "Counter",
    # }
    # if self.config.hue:
    #     default_axis_names[self.config.hue] = self.config.hue.capitalize()
    # default_metric_labels = {
    #     "absolute": "Counter value",
    #     "delta": "∆ Counter value",
    #     "overhead": "% Overhead",
    # }
    # grid_config = self.config.with_default_axis_rename(default_axis_names)
    # grid_config = grid_config.with_default_axis_remap(
    #     {"_metric_type": default_metric_labels}
    # )
    # grid_config = grid_config.with_config_default(
    #     tile_sharey=False,
    #     tile_sharex=False,
    #     tile_row="_counter",
    #     tile_col="_metric_type",
    # )

    pmc_filter: list[str] | None = config_field(
        None, desc="Show only the given subset of counters"
    )
    cpu_filter: list[int] | None = config_field(
        None, desc="Filter system-mode counters by CPU index"
    )

    drop_relative_baseline: bool = config_field(
        True, desc="Drop the baseline data point from the normalized plots"
    )

    def resolve_counter_group(self, task: PMCExec) -> str:
        return task.pmc_data.get_loader().counter_group


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

    Generated columns:
    - _counter: The name of the counter
    - value: The counter value
    - _metric_type: The statistic computed in value, can be one of absolute, delta,
      overhead
    - _cpu: The CPU on which the counters are obserte
    """

    task_namespace = "pmc"
    task_name = "summary"
    task_config_class = PMCPlotConfig
    public = True

    rc_params = {"axes.labelsize": "large", "font.size": 9, "xtick.labelsize": 9}

    derived_metrics = {
        "cycles_per_insn": pl.col("cpu_cycles") / pl.col("inst_retired"),
        "ipc": pl.col("inst_retired") / pl.col("cpu_cycles"),
        "br_pred_miss_rate": pl.col("br_mis_pred") / pl.col("br_pred"),
        "insn_spec_rate": pl.col("inst_retired") / pl.col("inst_spec"),
        "eff_backend_ipc": pl.col("inst_retired")
        / (pl.col("cpu_cycles") - pl.col("stall_backend")),
        "l1i_hit_rate": (pl.col("l1i_cache") - pl.col("l1i_cache_refill"))
        / pl.col("l1i_cache"),
        "l1d_hit_rate": (pl.col("l1d_cache") - pl.col("l1d_cache_refill"))
        / pl.col("l1d_cache"),
    }

    @dependency
    def counters(self):
        for bench in self.slice_benchmarks:
            task = bench.find_exec_task(PMCExec)
            yield task.pmc_data.get_loader()

    @output
    def summary_combined(self):
        return PlotTarget(self, "combined")

    @output
    def summary_tbl(self):
        return Target(self, "table", ext="csv")

    def _gen_derived_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate derived metrics for the slice.

        Expec the input dataframe to be in long form and have a constant _counter_group axis.
        Note that we avoid computing derived metrics across different _counter_group values,
        because these are captured in different executions and have weaker correlation
        depending on the overall probe effect.
        We generate a set of counter rows for each derived metric.
        """
        avail_counters = set(df["_counter"].unique())
        computable_metrics = {
            name: expr
            for name, expr in self.derived_metrics.items()
            if not set(expr.meta.root_names()).difference(avail_counters)
        }
        required_counters = set().union(
            *[expr.meta.root_names() for expr in computable_metrics.values()]
        )
        assert required_counters or len(computable_metrics) == 0

        index = [*self.key_columns_with_iter, "_counter_group", "_cpu"]
        result_df = (
            df.filter(pl.col("_counter").is_in(required_counters))
            .pivot(on="_counter", index=index, values="counter_value")
            .with_columns(
                [expr.alias(name) for name, expr in computable_metrics.items()]
            )
            .unpivot(
                on=computable_metrics.keys(),
                index=index,
                variable_name="_counter",
                value_name="counter_value",
            )
        )
        return pl.concat([df, result_df])

    def _collect_metrics(self, loader: IngestPMCCounters) -> pl.DataFrame:
        """
        The loader produces a dataframe in wide form; we convert this in long
        form to make it easier to merge counter data from etherogeneous
        counter groups.

        We compute derived metrics here with the assumption that the loader
        provides data that is associated to a single _counter_group.
        Wide form data makes this easier to do.
        """
        pmc_columns = ["_counter_group", "_cpu", "_counter", "counter_value"]
        df = loader.df.get()
        # Verify and normalize expected columns
        df = df.select(self.key_columns_with_iter + pmc_columns)
        assert df.n_unique("_counter_group") == 1, (
            "Loader has data from multiple counter groups"
        )
        df = self._gen_derived_metrics(df)
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
        df = (
            df.group_by(["dataset_id", "_counter", "iteration"])
            .agg(
                pl.col("counter_value").sum(),
                cs.exclude("counter_value").first(),
            )
            .select(cs.exclude("_cpu"))
        )
        # XXX Would be nice to check the integrity of this, i.e. we are not aggregating
        # unrelated things.

        self.logger.info("Bootstrap overhead confidence intervals")
        self.stats = self.compute_overhead(
            df,
            "counter_value",
            extra_groupby=["_counter"],
            how="median",
            overhead_scale=100,
        )

        if self.config.drop_relative_baseline:
            self.stats = self.stats.filter(
                (pl.col("_metric_type") == "absolute") | ~pl.col("_is_baseline")
            )

    def run_plot(self):
        self.logger.info("Plot combined PMC summary for slice %s", self.slice_info)

        # Draw the barplot but adjust the Y label to reflect the tile metric type
        with PlotGrid(self.summary_combined, self.stats, self.config) as grid:
            grid.map(
                grid_barplot,
                x=self.config.tile_xaxis,
                y="counter_value",
                err=["value_low", "value_high"],
                config=self.config,
            )
            grid.add_legend()

        # Pivot back for readability
        self.logger.info("Generate tabular data for slice %s", self.slice_info)
        tbl_data = self.stats.pivot(
            columns="_metric_type",
            index=[*self._param_columns, "_counter"],
            values="counter_value",
        )
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
        # default_axis_names = {
        #     "_counter": "Counter",
        #     "value": "Counter value",
        # }
        # if self.config.hue:
        #     default_axis_names[self.config.hue] = self.config.hue.capitalize()
        # grid_config = self.config.with_default_axis_rename(
        #     default_axis_names
        # ).with_config_default(tile_row="_counter")

        with PlotGrid(self.summary_plot, median_df, self.config) as grid:
            grid.map(
                grid_barplot,
                x=self.config.tile_xaxis,
                y="value",
                err=["value_low", "value_high"],
                config=self.config,
            )
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
        # default_axis_names = {
        #     "_counter": "Counter",
        #     "value": "∆ Counter value",
        # }
        # if self.config.hue:
        #     default_axis_names[self.config.hue] = self.config.hue.capitalize()
        # grid_config = self.config.with_default_axis_rename(
        #     default_axis_names
        # ).with_config_default(tile_row="_counter")

        with PlotGrid(self.summary_delta_plot, delta_df, self.config) as grid:
            grid.map(
                grid_barplot,
                x=self.config.tile_xaxis,
                y="value",
                err=["value_low", "value_high"],
                config=self.config,
            )
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
        # default_axis_names = {
        #     "_counter": "Counter",
        #     "value": "% Overhead",
        # }
        # if self.config.hue:
        #     default_axis_names[self.config.hue] = self.config.hue.capitalize()
        # grid_config = self.config.with_default_axis_rename(
        #     default_axis_names
        # ).with_config_default(tile_row="_counter")

        with PlotGrid(self.summary_ovh_plot, overhead_df, self.config) as grid:
            grid.map(
                grid_barplot,
                x=self.config.tile_xaxis,
                y="value",
                err=["value_low", "value_high"],
                config=self.config,
            )
            grid.add_legend()
