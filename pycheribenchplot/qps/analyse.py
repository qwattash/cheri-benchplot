from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import polars as pl
import seaborn as sns

from ..core.analysis import AnalysisTask
from ..core.artefact import Target, ValueTarget
from ..core.config import Config
from ..core.plot import new_facet, new_figure, PlotTask, PlotTarget
from ..core.task import dependency, output
from ..pmc.ingest import IngestPMCStatCounters
from .ingest import IngestQPSData

@dataclass
class QPSPlotConfig(Config):
    show_errorbars: bool = True
    #: Weigth for determining the order of labels based on the parameters
    parameter_weight: Optional[Dict[str, Dict[str, int]]] = None
    #: Relabel parameter axes
    parameter_names: Optional[Dict[str, str]] = None
    #: Relabel parameter axes values
    parameter_labels: Optional[Dict[str, Dict[str, str]]] = None


class LoadQPSData(AnalysisTask):
    """
    Load and merge data from all QPS runs.
    """
    task_namespace = "qps"
    task_name = "merge"

    @dependency
    def all_runs(self):
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(IngestQPSData)
            yield task.data.get_loader()

    @output
    def merged_df(self):
        return ValueTarget(self, "merged")

    def run(self):
        merged = pl.concat((loader.df.get() for loader in self.all_runs), how="vertical",
                           rechunk=True)
        self.merged_df.assign(merged)

    def get_parameter_columns(self):
        # Note that all benchmarks must have the same set of parameter keys.
        # This is enforced during configuration
        all_bench = self.session.all_benchmarks()
        assert len(all_bench) > 0
        return list(all_bench[0].parameters.keys())


class QPSPlot(PlotTask):
    """
    Simple QPS plot that shows the QPS metric on the Y axis and the target ABI
    on the X axis.
    The distribution among iterations is shown with a box plot.

    This generates a plot for each different scenario plus a summary plot containing
    all scenarios.
    """
    task_namespace = "qps"
    task_name = "qps-plot"
    public = True

    @dependency
    def data(self):
        return LoadQPSData(self.session, self.analysis_config)

    @output
    def summary_plot(self):
        return PlotTarget(self, "summary")

    def outputs(self):
        # Dynamically generate the output map
        yield from super().outputs()
        for scenario in self.get_scenarios():
            yield (scenario, PlotTarget(self, scenario))

    def get_scenarios(self):
        df = self.data.merged_df.get()
        if "scenario" not in df.columns:
            self.logger.error("Invalid parameterization, missing scenario parameter key")
            raise RuntimeError("Invalid configuration")
        return df["scenario"].unique()

    def run_plot(self):
        df = self.data.merged_df.get()
        params = self.data.get_parameter_columns()
        # Drop the scenario column from params as we use it as a facet axis
        params.remove("scenario")

        # Merge dataset description from all parameters
        df = df.with_columns(
            pl.col("dataset_gid").map_elements(self.g_uuid_to_label).alias("target"),
            pl.concat_str([pl.col(p) for p in params], separator=" ").alias("flavor/protection")
        )

        for scenario, s_df in df.groupby("scenario"):
            target = self.output_map[scenario]
            self.logger.info("Generate QPS plot for %s", scenario)
            with new_figure(target.paths()) as fig:
                ax = fig.subplots()
                sns.stripplot(s_df, x="target", y="qps", hue="flavor/protection",
                              dodge=True, ax=ax)
                ax.set_title(scenario)
                ax.set_ylabel("QPS")
                ax.set_xlabel("Target")

        # Make the summary plot
        self.logger.info("Generate QPS summary plot")
        prot_combinations = len(df["flavor/protection"].unique())
        palette = sns.color_palette(n_colors=prot_combinations)
        with sns.plotting_context(font_scale=0.4):
            with new_facet(self.summary_plot.paths(), df, col="scenario", col_wrap=4) as facet:
                facet.map_dataframe(sns.stripplot, x="target", y="qps", dodge=True,
                                    hue="flavor/protection", palette=palette)
                facet.add_legend()


class LatencyPlot(PlotTask):
    """
    Simple QPS plot that shows the Latency summary distribution on the Y axis and
    target ABI on the X axis.
    The distribution among iterations is shown with a box plot.

    This generates a plot for each different scenario.
    """
    task_namespace = "qps"
    task_name = "latency-plot"
    public = True

    rc_params = {
        "axes.labelsize": "large",
        "font.size": 6,
        "xtick.labelsize": 5
    }

    @dependency
    def data(self):
        return LoadQPSData(self.session, self.analysis_config)

    def outputs(self):
        # Dynamically generate the output map
        yield from super().outputs()
        for scenario in self.get_scenarios():
            yield (scenario, PlotTarget(self, scenario))

    def get_scenarios(self):
        df = self.data.merged_df.get()
        if "scenario" not in df.columns:
            self.logger.error("Invalid parameterization, missing scenario parameter key")
            raise RuntimeError("Invalid configuration")
        return df["scenario"].unique()

    def run_plot(self):
        df = self.data.merged_df.get()
        params = self.data.get_parameter_columns()
        # Drop the scenario column from params as we use it as a facet axis
        params.remove("scenario")

        # Merge dataset description from all parameters
        df = df.with_columns(
            pl.col("dataset_gid").map_elements(self.g_uuid_to_label).alias("target"),
            pl.concat_str([pl.col(p) for p in params], separator=" ").alias("flavor/protection"),
            pl.col("latency50") / 1000,
            pl.col("latency90") / 1000,
            pl.col("latency95") / 1000,
            pl.col("latency99") / 1000
        )
        df = df.melt(id_vars=["target", "flavor/protection", "scenario"],
                     value_vars=["latency50", "latency90", "latency95", "latency99"],
                     variable_name="percentile",
                     value_name="latency")

        prot_combinations = len(df["flavor/protection"].unique())
        palette = sns.color_palette(n_colors=prot_combinations)
        for scenario, s_df in df.groupby("scenario"):
            target = self.output_map[scenario]
            self.logger.info("Generate Latency plot for %s", scenario)
            with new_facet(target.paths(), df, col="percentile", col_wrap=2) as facet:
                facet.map_dataframe(sns.boxplot, x="target", y="latency",
                                    hue="flavor/protection", palette=palette)
                facet.add_legend()
                facet.fig.subplots_adjust(top=0.9)
                facet.fig.suptitle(scenario)


class QPSByMsgSizePlot(PlotTask):
    """
    Plot the QPS metric across the different message sizes to show how the overhead scales.

    Note that the overhead plot is normalized to the mean of the baseline benchmark run.
    The overhead is inverted here, because we want positive % numbers to indicate
    loss in performance.
    """
    task_namespace = "qps"
    task_name = "qps-by-msgsize-plot"
    task_config_class = QPSPlotConfig
    public = True

    # Plot styling overrides
    rc_params = {
        "axes.labelsize": "large",
        "font.size": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    }

    @dependency
    def data(self):
        return LoadQPSData(self.session, self.analysis_config)

    @output
    def qps_plot(self):
        return PlotTarget(self, "abs")

    @output
    def overhead_plot(self):
        return PlotTarget(self, "rel")

    @output
    def overhead_box_plot(self):
        return PlotTarget(self, "rel-box")

    @output
    def overhead_tbl(self):
        return Target(self, "tbl", ext="csv")

    def run_plot(self):
        df = self.data.merged_df.get()
        params = self.data.get_parameter_columns()
        # Drop the scenario column from params as we use the message size axis instead
        params.remove("scenario")

        # If the 'variant' parameter has only one alternative, we skip it in the legend
        if len(df["variant"].unique()) == 1:
            params.remove("variant")

        df = df.with_columns(
            pl.col("dataset_gid").map_elements(self.g_uuid_to_label).alias("target"),
            pl.concat_str([pl.col(p) for p in params], separator=" ").alias("flavor/protection")
        )

        # Honor error bars configuration
        if self.config.show_errorbars:
            err_conf = ("pi", 90)
        else:
            err_conf = None

        self.logger.info("Generate QPSByMsgSize plot")
        with new_figure(self.qps_plot.paths()) as fig:
            ax = fig.subplots()
            sns.lineplot(df, x="req_size", y="qps", hue="target", style="flavor/protection",
                         markers=True, dashes=True, ax=ax, estimator="mean",
                         err_style="band", errorbar=err_conf)
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(df["req_size"].unique()))
            ax.set_xlabel("Request size")
            ax.set_ylabel("QPS")

        self.logger.info("Generate QPS overhead plot")
        baseline = self.baseline_slice(df)
        baseline_qps = baseline.select(["qps", "scenario"]).group_by("scenario").mean().rename(dict(qps="qps_baseline"))
        # Create the overhead column and drop the baseline data
        df = df.join(baseline_qps, on="scenario").with_columns(
            ((pl.col("qps_baseline") - pl.col("qps")) * 100 / pl.col("qps_baseline")).alias("overhead")
        ).join(baseline, on="dataset_id", how="anti")
        with new_figure(self.overhead_plot.paths()) as fig:
            ax = fig.subplots()
            sns.lineplot(df, x="req_size", y="overhead", hue="target", style="flavor/protection",
                         markers=True, dashes=True, ax=ax, estimator="mean",
                         err_style="band", errorbar=err_conf)
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(df["req_size"].unique()))
            ax.set_xlabel("Request size")
            ax.set_ylabel("% Overhead")

        with new_facet(self.overhead_box_plot.paths(), df, col="target") as facet:
            prot_combinations = len(df["flavor/protection"].unique())
            palette = sns.color_palette(n_colors=prot_combinations)

            facet.map_dataframe(sns.boxplot, x="req_size", y="overhead",
                                hue="flavor/protection", log_scale=(2, None),
                                palette=palette, native_scale=True)
            facet.set_axis_labels(x_var="Request size", y_var="% Overhead")
            for ax in facet.axes.flatten():
                ax.set_xticks(sorted(df["req_size"].unique()))
            facet.add_legend()
            self.adjust_legend_on_top(facet.fig, loc="lower left")


        # Print out the mean overhead in tabular form
        idcols = ["req_size", "target", "flavor/protection"]
        tbl = df.group_by(idcols).mean().select(idcols + ["overhead"]).sort(idcols)
        with open(self.overhead_tbl.single_path(), "w+") as fp:
            tbl.write_csv(fp)


@dataclass
class QPSOverheadPlotConfig(QPSPlotConfig):
    #: Control which parameterization axis is used for the facet grid columns
    facet_columns: Optional[str] = None


class QPSOverheadPlot(PlotTask):
    """
    Simple QPS plot that shows the QPS metric on the Y axis and the target ABI
    on the X axis.
    The distribution among iterations is shown with a box plot.

    This generates a plot for each different scenario plus a summary plot containing
    all scenarios.
    """
    task_config_class = QPSOverheadPlotConfig
    task_namespace = "qps"
    task_name = "qps-overhead-plot"
    public = True

    # Plot styling overrides
    rc_params = {
        "axes.labelsize": "large",
        "font.size": 8,
        "xtick.labelsize": 8,
    }

    @dependency
    def data(self):
        return LoadQPSData(self.session, self.analysis_config)

    def outputs(self):
        # Dynamically generate the output map
        yield from super().outputs()
        for scenario in self.get_scenarios():
            yield (scenario, PlotTarget(self, scenario))

    def get_scenarios(self):
        df = self.data.merged_df.get()
        if "scenario" not in df.columns:
            self.logger.error("Invalid parameterization, missing scenario parameter key")
            raise RuntimeError("Invalid configuration")
        return df["scenario"].unique()

    def run_plot(self):
        df = self.data.merged_df.get()
        params = self.data.get_parameter_columns()
        # Drop the scenario column from params as we use it as a facet axis
        params.remove("scenario")

        # Compute the overhead
        # IMPORTANT: this must occur before we do any of the plot-specific renaming
        baseline = self.baseline_slice(df)
        baseline_qps = baseline.select(["qps", "scenario"]).group_by("scenario").mean().rename(dict(qps="qps_baseline"))
        # Create the overhead column and optionally drop the baseline data
        df = df.join(baseline_qps, on="scenario").with_columns(
            (pl.col("qps") * 100 / pl.col("qps_baseline")).alias("overhead"),
        ).join(baseline, on="dataset_id", how="anti")

        # Proceeed with plot-specific dataframe transforms

        df = df.with_columns(
            pl.col("dataset_gid").map_elements(self.g_uuid_to_label).alias("target"),
        )

        # Compute the parameterization weight for consistent label ordering
        if self.config.parameter_weight:
            df = df.with_columns(pl.lit(0).alias("param_weight"))
            for name, mapping in self.config.parameter_weight.items():
                self.logger.debug("Set weight for %s => %s", name, mapping)
                df = df.with_columns(
                    pl.col("param_weight") + pl.col(name).replace(mapping, default=0)
                )

        # Relabel parameterization axes according to the task configuration overrides
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

        # If we have an axis to use for the facet grid, remove it from the parameters set
        if self.config.facet_columns is not None:
            if self.config.facet_columns not in params:
                self.logger.warning("Configured facet columns key '%s' is missing, "
                                    "maybe it was renamed by QPSPlotConfig.parameter_names. "
                                    "Note that column renaming occurs before this step, so the "
                                    "QPSOverheadPlotConfig.facet_columns key must contain the "
                                    "new parameter name.", self.config.facet_columns)
            params.remove(self.config.facet_columns)

        # Check if there is variation on the flavor/protection axes, otherwise drop them
        drop_params = []
        for p in params:
            if len(df.select(p).unique()) == 1:
                self.logger.debug("No variation on the '%s' axis detected, drop parameterization axis in plot", p)
                drop_params.append(p)
        params = [p for p in params if p not in drop_params]

        # Honor error bars configuration
        if self.config.show_errorbars:
            assert False, "TODO, need to implement"
            err_conf = ("pi", 90)
        else:
            err_conf = None

        self.logger.info("Generate QPS overhead plot with facet_columns=%s, params=%s",
                         self.config.facet_columns, params)

        # Now merge all the parameter columns into the target label
        df = df.with_columns(
            pl.concat_str([pl.col(p) for p in sorted(params)], separator="\n").alias("target")
        )

        for scenario, s_df in df.groupby("scenario"):
            target = self.output_map[scenario]
            self.logger.info("Generate QPS overhead plot for %s", scenario)
            with new_figure(target.paths()) as fig:
                ax = fig.subplots()
                axr = ax.twinx()
                axr.grid(False)

                mean_df = s_df.group_by("target").mean().sort("param_weight")
                x_labels = mean_df["target"]
                x_pos = np.arange(len(x_labels))

                abs_bars = ax.bar(x_pos - 0.125, mean_df["qps"], color="r", width=0.25)
                pct_bars = axr.bar(x_pos + 0.125, mean_df["overhead"], color="b", width=0.25)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels)

                ax.set_xlabel("Target")
                ax.set_ylabel("QPS (messages/sec)")
                axr.set_ylabel("% Relative to no memory safety")
                axr.set_ylim(0, 110)
                ax.legend(handles=[abs_bars, pct_bars], labels=["QPS (msg/s)", "% of no memory safety"],
                          bbox_to_anchor=(0, 1.), loc="lower left", ncols=4)


class QPSPerfCountersPlot(PlotTask):
    """
    Generate mixed qps/pmc metrics when hardware performance counters data is available.
    """
    task_config_class = QPSPlotConfig
    task_namespace = "qps"
    task_name = "pmc-metrics"
    public = True

    derived_metrics = {
        "ex_entry_per_msg": {
            "requires": ["executive_entry", "message_count"],
            "column": (pl.col("executive_entry") / pl.col("message_count"))
        },
        "ex_entry_per_byte": {
            "requires": ["executive_entry", "message_count", "resp_size"],
            "column": (pl.col("executive_entry") / (pl.col("message_count") * pl.col("resp_size")))
        }
    }

    @dependency(optional=True)
    def pmc(self):
        for bench in self.session.all_benchmarks():
            task = bench.find_exec_task(IngestPMCStatCounters)
            yield task.counter_data.get_loader()

    @dependency
    def qps_data(self):
        return LoadQPSData(self.session, self.analysis_config)

    @output
    def qps_metrics(self):
        return PlotTarget(self, "metrics")

    def get_metadata_columns(self):
        # Note that all benchmarks must have the same set of parameter keys.
        # This is enforced during configuration
        return self.qps_data.get_parameter_columns() + ["iteration", "target"]

    def apply_display_transforms(self, df):
        """
        Transform the dataframe to adjust displayed properties.

        This applies the plot configuration to rename parameter levels, axes and
        filters.
        """
        params = self.qps_data.get_parameter_columns()
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
        df = df.with_columns(
            pl.concat_str([pl.col(p) for p in params], separator=" ").alias("flavor/protection")
        )
        return df

    def run_plot(self):
        if self.pmc is None:
            self.logger.info("Skip task %s: missing PMC data", self)
            return

        # Merge the counter data from everywhere
        pmc_df = pl.concat([loader.df.get() for loader in self.pmc]).with_columns(
            pl.col("iteration").cast(pl.Int32)
        )
        assert len(pmc_df.filter(iteration=-1)) == 0, "Missing iteration number on PMC frame"
        # Join with the QPS data
        qps_df = self.qps_data.merged_df.get().with_columns(
            pl.col("iteration").cast(pl.Int32)
        )
        assert len(qps_df.filter(iteration=-1)) == 0, "Missing iteration number on QPS frame"

        df = qps_df.join(pmc_df, on=["dataset_id", "iteration"])
        assert df.shape[0] == pmc_df.shape[0], "Unexpected dataframe shape change"
        assert df.shape[0] == qps_df.shape[0], "Unexpected dataframe shape change"

        # Generate derived metrics
        found_metrics = []
        for name, spec in self.derived_metrics.items():
            has_cols = True
            for required_column in spec["requires"]:
                if required_column not in df.columns:
                    self.logger.info("Skip derived metric %s: requires missing column %s",
                                     name, required_column)
                    has_cols = False
                    break
            if not has_cols:
                continue
            df = df.with_columns((spec["column"]).alias(name))
            found_metrics.append(name)

        if not found_metrics:
            self.logger.info("Skipping plot, no data")
            return

        # Now produce a plot for each derived metric we found
        df = df.with_columns(
            pl.col("dataset_gid").map_elements(self.g_uuid_to_label).alias("target")
        ).melt(
            id_vars=self.get_metadata_columns(),
            value_vars=found_metrics,
            variable_name="metric",
            value_name="value"
        )
        df = self.apply_display_transforms(df)
        prot_combinations = len(df["flavor/protection"].unique())
        palette = sns.color_palette(n_colors=prot_combinations)

        self.logger.info("Generate QPS PMC metrics")
        with new_facet(self.qps_metrics.paths(), df, col="scenario", row="metric",
                       sharex="col", sharey="row", margin_titles=True,
                       aspect=0.85) as facet:
            facet.map_dataframe(sns.barplot, x="target", y="value",
                                hue="flavor/protection", dodge=True,
                                palette=palette)
            facet.add_legend()
