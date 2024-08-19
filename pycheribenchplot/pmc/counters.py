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
from ..core.plot_util import extended_barplot
from ..core.task import dependency, output
from ..core.tvrs import TVRSParamsMixin, TVRSPlotConfig
from .pmc_exec import PMCExec, PMCExecConfig


@dataclass
class PMCPlotConfig(TVRSPlotConfig):
    pmc_filter: Optional[List[str]] = config_field(None, desc="Show only the given subset of counters")
    cpu_filter: Optional[List[int]] = config_field(None, desc="Filter system-mode counters by CPU index")
    lock_y_axis: bool = config_field(False, desc="Lock Y axis")
    share_x_axis: bool = config_field(False, desc="Share the X axis among rows")
    counter_group_name: Optional[str] = config_field(
        None, desc="Parameter to use to identify runs with different sets of counters")
    tile_column_name: Optional[str] = config_field("scenario", desc="Parameter to use for the facet grid column axis")
    tile_x_name: str = config_field("target", desc="Parameter to use for the X axis of each subplot")
    tile_aspect: float = config_field(1.0, desc="Aspect ratio of the facet tiles")

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
        ctx = self.make_param_context(df)
        ctx.melt(value_vars=[c.lower() for c in exec_config.counters_list], variable_name="counter", value_name="value")
        ctx.relabel()

        def format_tick(x, pos=None):
            sign = np.sign(x)
            value = abs(x)
            if value == 0:
                return "0"
            xmag = int(np.log10(value))
            xnew = sign * value / 10**xmag
            return f"${xnew:.02f}^{{{xmag}}}$"

        self.logger.info("Generate PMC distribution for %s: %s", self.benchmark.config.name,
                         ", ".join([f"{k}={v}" for k, v in self.benchmark.parameters.items()]))
        with new_facet(self.distribution_plot.paths(), ctx.df, col="counter", col_wrap=4, sharex=False) as facet:
            facet.map_dataframe(sns.histplot, "value")
            facet.add_legend()
            for ax in facet.axes.flat:
                ax.xaxis.set_major_formatter(FuncFormatter(format_tick))


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
    def summary_box_plot(self):
        return PlotTarget(self, "summary-box")

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

    def gen_summary(self, ctx):
        """
        Generate the summary stripplot
        """
        palette = ctx.build_palette_for("_hue", allow_empty=False)
        sharey = False
        if self.config.lock_y_axis:
            sharey = "row"
        sharex = self.config.share_x_axis

        # Rename customisable parameters for the tile grid and X axis
        tile_col = ctx.r[self.config.tile_column_name]
        tile_x = ctx.r[self.config.tile_x_name]

        self.logger.info("Generate PMC counters summary for group '%s'", self.pmc_group)
        with new_facet(self.summary_plot.paths(),
                       ctx.df,
                       row=ctx.r.counter,
                       col=tile_col,
                       sharex=sharex,
                       sharey=sharey,
                       margin_titles=True,
                       aspect=self.config.tile_aspect) as facet:
            facet.map_dataframe(sns.stripplot, x=tile_x, y=ctx.r.value, hue=ctx.r._hue, dodge=True, palette=palette)
            if palette:
                facet.add_legend()
                self.adjust_legend_on_top(facet.figure)

        with new_facet(self.summary_box_plot.paths(),
                       ctx.df,
                       row=ctx.r.counter,
                       col=tile_col,
                       sharex=sharex,
                       sharey=sharey,
                       margin_titles=True,
                       aspect=self.config.tile_aspect) as facet:
            facet.map_dataframe(sns.boxplot, x=tile_x, y=ctx.r.value, hue=ctx.r._hue, dodge=True, palette=palette)
            if palette:
                facet.add_legend()
                self.adjust_legend_on_top(facet.figure)

    def gen_delta(self, ctx):
        """
        Generate delta plots
        """
        palette = ctx.build_palette_for("_hue", allow_empty=False)
        sharey = False
        if self.config.lock_y_axis:
            sharey = "row"
        sharex = self.config.share_x_axis

        # Rename customisable parameters for the tile grid and X axis
        tile_col = ctx.r[self.config.tile_column_name]
        tile_x = ctx.r[self.config.tile_x_name]

        self.logger.info("Generate PMC delta summary for group '%s'", self.pmc_group)
        with new_facet(self.summary_delta_plot.paths(),
                       ctx.df,
                       row=ctx.r.counter,
                       col=tile_col,
                       sharex=sharex,
                       sharey=sharey,
                       margin_titles=True,
                       aspect=self.config.tile_aspect) as facet:
            facet.map_dataframe(sns.barplot,
                                x=tile_x,
                                y=ctx.r.value_delta,
                                hue=ctx.r._hue,
                                dodge=True,
                                palette=palette,
                                capsize=0.3)
            if palette:
                facet.add_legend()
                self.adjust_legend_on_top(facet.figure)

        self.logger.info("Generate PMC overhead summary for group '%s'", self.pmc_group)
        with new_facet(self.summary_ovh_plot.paths(),
                       ctx.df,
                       row=ctx.r.counter,
                       col=tile_col,
                       sharex=sharex,
                       sharey=sharey,
                       margin_titles=True,
                       aspect=self.config.tile_aspect) as facet:
            facet.map_dataframe(sns.barplot,
                                x=tile_x,
                                y=ctx.r.value_overhead,
                                hue=ctx.r._hue,
                                dodge=True,
                                palette=palette,
                                capsize=0.3)
            if palette:
                facet.add_legend()
                self.adjust_legend_on_top(facet.figure)

    def gen_combined(self, ctx):
        """
        Generate combined plot with the absolute value, absolue diff and relative
        overhead for each counter.
        The data is shown as the mean and standard deviation of the iteration samples.
        Delta and relative overhead are computed by propagating uncertainty as standard deviation.
        The hue and tiling are controlled by task configuration options.
        """
        palette = ctx.build_palette_for("_hue", allow_empty=False)

        with new_facet(self.summary_combined.paths(),
                       ctx.df,
                       col=ctx.r._metric_type,
                       row=ctx.r.counter,
                       sharex=self.config.share_x_axis,
                       sharey=False,
                       margin_titles=True,
                       aspect=self.config.tile_aspect) as facet:
            x_col = ctx.r[self.config.tile_x_name]

            facet.map_dataframe(extended_barplot,
                                errorbar=("custom", {
                                    "yerr": [ctx.r.value_q25, ctx.r.value_q75],
                                    "color": "black"
                                }),
                                x=x_col,
                                y=ctx.r.value,
                                hue=ctx.r._hue,
                                dodge=True,
                                palette=palette)
            # Ensure that we have labels on the each of the grid Y axes
            for ijk, data in facet.facet_data():
                row, col, hue = ijk
                # Note that the metric type is supposed to be unique for each column
                # and data is a pandas dataframe.
                label = data[ctx.r._metric_type].iloc[0]
                facet.axes[row, col].set_ylabel(label)

            if palette:
                facet.add_legend()
                self.adjust_legend_on_top(facet.figure)
            facet.tight_layout()

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

        ctx = self.make_param_context(df)
        extra_index_cols = ["iteration"]
        if "_cpu" in df.columns:
            extra_index_cols = [*extra_index_cols, "_cpu"]
        ctx.melt(extra_id_vars=extra_index_cols, value_vars=metric_cols, variable_name="counter", value_name="value")
        # Filter the counters based on configuration
        if self.config.pmc_filter:
            ctx.df = ctx.df.filter(pl.col("counter").is_in(self.config.pmc_filter))
        if self.config.cpu_filter:
            ctx.df = ctx.df.filter(pl.col("_cpu").is_in(self.config.cpu_filter))

        if "_cpu" in df.columns:
            # Sum metrics from different CPUs
            ctx.df = ctx.df.group_by([*ctx.params, "counter",
                                      "iteration"]).agg(cs.numeric().sum(),
                                                        cs.string().first()).select(cs.all().exclude("_cpu"))

        ovh_ctx = ctx.compute_overhead(["value"], extra_groupby=["counter"])

        lf_ctx = ctx.compute_lf_overhead("value", extra_groupby=["counter"], overhead_scale=100, how="median")

        ctx.relabel()
        ctx.derived_hue_param(default=["variant", "runtime"])
        ctx.sort()
        self.gen_summary(ctx)

        ovh_ctx.relabel()
        ovh_ctx.derived_hue_param(default=["variant", "runtime"])
        ovh_ctx.sort()
        self.gen_delta(ovh_ctx)

        lf_ctx.relabel(default=dict(_metric_type="Measurement"))
        lf_ctx.derived_hue_param(default=["variant", "runtime"])
        lf_ctx.sort()
        self.gen_combined(lf_ctx)

        # Pivot back for readability
        tbl_data = lf_ctx.df.pivot(columns=lf_ctx.r._metric_type,
                                   index=[*lf_ctx.base_params, lf_ctx.r.counter],
                                   values=lf_ctx.r.value)
        tbl_data.write_csv(self.summary_tbl.single_path())
