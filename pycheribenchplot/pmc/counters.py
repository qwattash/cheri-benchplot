from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from ..core.artefact import ValueTarget
from ..core.config import Config
from ..core.plot import DatasetPlotTask, PlotTask, PlotTarget, new_facet
from ..core.task import dependency, output
from .ingest import IngestPMCStatCounters

@dataclass
class PMCPlotConfig(Config):
    show_errorbars: bool = True
    #: Weigth for determining the order of labels based on the parameters
    parameter_weight: Optional[Dict[str, Dict[str, int]]] = None
    #: Relabel parameter axes
    parameter_names: Optional[Dict[str, str]] = None
    #: Relabel parameter axes values
    parameter_labels: Optional[Dict[str, Dict[str, str]]] = None
    #: Filter only the given subset of counters
    pmc_filter: Optional[List[str]] = None
    #: Parameterization axes for the combined plot hue, use all remaining by default
    hue_parameters: Optional[List[str]] = None
    #: Lock Y axis
    lock_y_axis: bool = False


class PMCStatDistribution(DatasetPlotTask):
    """
    Generate a summary showing the distribution of counters data within a single dataset.
    """
    task_namespace = "pmc"
    task_name = "cnt-distribution"
    public = True

    @dependency
    def counters(self):
        task = self.benchmark.find_exec_task(IngestPMCStatCounters)
        return task.counter_data.get_loader()

    @output
    def distribution_plot(self):
        return PlotTarget(self, "distribution")

    def run_plot(self):
        df = self.counters.df.get()

        df = df.with_columns(
            pl.col("dataset_gid").map_elements(self.g_uuid_to_label).alias("target")
        ).melt(
            id_vars="target",
            value_vars=~cs.by_name(self.benchmark.metadata_columns + ["target"]),
            variable_name="counter",
            value_name="value"
        )

        def format_tick(x, pos=None):
            sign = np.sign(x)
            value = abs(x)
            if value == 0:
                return "0"
            xmag = int(np.log10(value))
            xnew = sign * value / 10**xmag
            return f"${xnew:.02f}^{{{xmag}}}$"

        self.logger.info("Generate PMC distribution for %s", self.benchmark.config.name)
        with new_facet(self.distribution_plot.paths(), df, col="counter", col_wrap=4, sharex=False) as facet:
            facet.map_dataframe(sns.histplot, "value")
            facet.add_legend()
            for ax in facet.axes.flat:
                ax.xaxis.set_major_formatter(FuncFormatter(format_tick))


class PMCStatSummary(PlotTask):
    """
    Generate a summary showing the variation of PMC counters across the dataset
    parameterisation.
    """
    task_config_class = PMCPlotConfig
    task_namespace = "pmc"
    task_name = "generic-summary"
    public = True

    rc_params = {
        "axes.labelsize": "large",
        "font.size": 9,
        "xtick.labelsize": 9
    }

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
        }
    }

    @dependency
    def counters(self):
        for bench in self.session.all_benchmarks():
            task = bench.find_exec_task(IngestPMCStatCounters)
            yield task.counter_data.get_loader()

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
    def summary_metrics_plot(self):
        return PlotTarget(self, "metrics")

    def get_parameter_columns(self):
        all_bench = self.session.all_benchmarks()
        assert len(all_bench) > 0
        return list(all_bench[0].parameters.keys())

    def get_metadata_columns(self):
        # Additional generated columns that are not counters
        extra_metadata_columns = ["target"]

        all_bench = self.session.all_benchmarks()
        assert len(all_bench) > 0
        return all_bench[0].metadata_columns + extra_metadata_columns

    def gen_summary(self, df, palette):
        """
        Generate the summary stripplot
        """
        sharey = False
        if self.config.lock_y_axis:
            sharey = "row"

        self.logger.info("Generate PMC counters summary")
        with new_facet(self.summary_plot.paths(), df, row="counter",
                       col="scenario", sharex="col", sharey=sharey,
                       margin_titles=True, aspect=0.65) as facet:
            facet.map_dataframe(sns.stripplot, x="target", y="value",
                                hue="flavor/protection", dodge=True,
                                palette=palette)
            facet.add_legend()

        with new_facet(self.summary_box_plot.paths(), df, row="counter",
                       col="scenario", sharex="col", sharey=sharey,
                       margin_titles=True, aspect=0.65) as facet:
            facet.map_dataframe(sns.boxplot, x="target", y="value",
                                hue="flavor/protection", dodge=True,
                                palette=palette)
            facet.add_legend()

    def gen_delta(self, df, baseline, palette):
        """
        Generate delta plots
        """
        sharey = False
        if self.config.lock_y_axis:
            sharey = "row"

        self.logger.info("Generate PMC delta summary")
        with new_facet(self.summary_delta_plot.paths(), df, row="counter",
                       col="scenario", sharex="col", sharey=sharey,
                       margin_titles=True) as facet:
            facet.map_dataframe(sns.barplot, x="target", y="delta",
                                hue="flavor/protection", dodge=True,
                                palette=palette)
            facet.add_legend()

        # Drop the baseline data from the dataframe, we don't want it in the overhead plot
        df = df.join(baseline, on="dataset_id", how="anti")
        # Update the palette
        prot_combinations = len(df["flavor/protection"].unique())
        palette = sns.color_palette(n_colors=prot_combinations)

        self.logger.info("Generate PMC counters overhead summary")
        with new_facet(self.summary_ovh_plot.paths(), df, row="counter",
                       col="scenario", sharex="col", sharey=sharey,
                       margin_titles=True) as facet:
            facet.map_dataframe(sns.barplot, x="target", y="overhead",
                                hue="flavor/protection", dodge=True,
                                palette=palette)
            facet.add_legend()

    def apply_display_transforms(self, df):
        """
        Transform the dataframe to adjust displayed properties.

        This applies the plot configuration to rename parameter levels, axes and
        filters.
        """
        params = self.get_parameter_columns()
        # Parameter renames
        if self.config.parameter_labels:
            relabeling = []
            for name, mapping in self.config.parameter_labels.items():
                if name not in df.columns:
                    self.logger.warning("Skipping re-labeling of parameter '%s', does not exist", name)
                    continue
                relabeling.append(pl.col(name).replace(mapping))
            df = df.with_columns(*relabeling)

        if self.config.parameter_names:
            df = df.rename(self.config.parameter_names)
            params = [self.config.parameter_names.get(p, p) for p in params]

        # Hide unnecessary parameter axes
        hide_params = ["scenario"]
        for p in params:
            if len(df[p].unique()) == 1:
                hide_params.append(p)
        params = [p for p in params if p not in hide_params]
        # Generate the combined hue column flavor/protection
        if self.config.hue_parameters:
            hue_params = [p for p in params if p in self.config.hue_parameters]
        else:
            hue_params = params
        df = df.with_columns(
            pl.concat_str([pl.col(p) for p in hue_params], separator=" ").alias("flavor/protection")
        )

        # Filter the counters based on configuration
        if self.config.pmc_filter:
            df = df.filter(pl.col("counter").is_in(self.config.pmc_filter))

        return df

    def run_plot(self):
        df = pl.concat([dep.df.get() for dep in self.counters])

        # Generate derived metrics
        for name, spec in self.derived_metrics.items():
            has_cols = True
            for required_column in spec["requires"]:
                if required_column not in df.columns:
                    self.logger.debug("Skip derived metric %s: requires missing column %s",
                                      name, required_column)
                    has_cols = False
                    break
            if not has_cols:
                continue
            df = df.with_columns((spec["column"]).alias(name))

        metadata_cols = cs.by_name(self.get_metadata_columns())
        df = df.with_columns(
            pl.col("dataset_gid").map_elements(self.g_uuid_to_label).alias("target")
        ).melt(
            id_vars=metadata_cols,
            value_vars=~metadata_cols,
            variable_name="counter",
            value_name="value"
        )

        # Generate the overhead relative column as
        # delta = (metric - baseline)
        # overhead = 100 * delta / baseline
        baseline = self.baseline_slice(df)
        baseline_mean = (
            baseline.group_by(self.get_parameter_columns() + ["counter"])
            .mean()
            .select(["scenario", "counter", "value"])
            .rename(dict(value="baseline_mean"))
        )
        stat_df = df.join(baseline_mean, on=["scenario", "counter"]).with_columns(
            (pl.col("value") - pl.col("baseline_mean")).alias("delta"),
            (100 * (pl.col("value") - pl.col("baseline_mean")) / pl.col("baseline_mean")).alias("overhead")
        )
        # This join should preserve the number of rows, otherwise something weird is going on
        assert stat_df.shape[0] == df.shape[0], "Unexpected dataframe shape change"
        df = stat_df
        # XXX Handle error propagation

        df = self.apply_display_transforms(df)
        prot_combinations = len(df["flavor/protection"].unique())
        if prot_combinations == 2:
            # Hack, allow in config
            sns.set_palette(sns.color_palette())
            palette = ["b", "r"]
        else:
            palette = sns.color_palette(n_colors=prot_combinations)

        self.gen_summary(df, palette)
        self.gen_delta(df, baseline, palette)
