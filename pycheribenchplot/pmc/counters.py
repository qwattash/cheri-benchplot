from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from ..core.artefact import ValueTarget
from ..core.config import Config, config_field
from ..core.plot import DatasetPlotTask, PlotTarget, PlotTask, new_facet
from ..core.task import dependency, output
from ..core.tvrs import TVRSParamsMixin, TVRSTaskConfig
from .pmc_exec import PMCExec, PMCExecConfig


@dataclass
class PMCPlotConfig(TVRSTaskConfig):
    pmc_filter: Optional[List[str]] = config_field(None, desc="Show only the given subset of counters")
    lock_y_axis: bool = config_field(False, "Lock Y axis")
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

    def gen_summary(self, ctx):
        """
        Generate the summary stripplot
        """
        palette = ctx.build_palette_for("_hue", allow_empty=False)
        sharey = False
        if self.config.lock_y_axis:
            sharey = "row"

        # Rename customisable parameters for the tile grid and X axis
        tile_col = ctx.r[self.config.tile_column_name]
        tile_x = ctx.r[self.config.tile_x_name]

        self.logger.info("Generate PMC counters summary for group '%s'", self.pmc_group)
        with new_facet(self.summary_plot.paths(),
                       ctx.df,
                       row=ctx.r.counter,
                       col=tile_col,
                       sharex="col",
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
                       sharex="col",
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

        # Rename customisable parameters for the tile grid and X axis
        tile_col = ctx.r[self.config.tile_column_name]
        tile_x = ctx.r[self.config.tile_x_name]

        self.logger.info("Generate PMC delta summary for group '%s'", self.pmc_group)
        with new_facet(self.summary_delta_plot.paths(),
                       ctx.df,
                       row=ctx.r.counter,
                       col=tile_col,
                       sharex="col",
                       sharey=sharey,
                       margin_titles=True,
                       aspect=self.config.tile_aspect) as facet:
            facet.map_dataframe(sns.barplot, x=tile_x, y=ctx.r.value_delta, hue=ctx.r._hue, dodge=True, palette=palette)
            if palette:
                facet.add_legend()
                self.adjust_legend_on_top(facet.figure)

        self.logger.info("Generate PMC overhead summary for group '%s'", self.pmc_group)
        with new_facet(self.summary_ovh_plot.paths(),
                       ctx.df,
                       row=ctx.r.counter,
                       col=tile_col,
                       sharex="col",
                       sharey=sharey,
                       margin_titles=True,
                       aspect=self.config.tile_aspect) as facet:
            facet.map_dataframe(sns.barplot,
                                x=tile_x,
                                y=ctx.r.value_overhead,
                                hue=ctx.r._hue,
                                dodge=True,
                                palette=palette)
            if palette:
                facet.add_legend()
                self.adjust_legend_on_top(facet.figure)

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
        ctx.melt(extra_id_vars=["iteration"], value_vars=metric_cols, variable_name="counter", value_name="value")
        # Filter the counters based on configuration
        if self.config.pmc_filter:
            ctx.df = ctx.df.filter(pl.col("counter").is_in(self.config.pmc_filter))
        # Generate hue parameter level
        ctx.derived_hue_param(default=["variant", "runtime"])
        ovh_ctx = ctx.compute_overhead(["value"], extra_groupby=["counter"])

        ctx.relabel()
        self.gen_summary(ctx)

        ovh_ctx.relabel()
        self.gen_delta(ovh_ctx)
