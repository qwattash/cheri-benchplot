from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from ..core.analysis import AnalysisTask
from ..core.artefact import Target, ValueTarget
from ..core.config import Config
from ..core.plot import PlotTarget, PlotTask, new_facet, new_figure
from ..core.task import dependency, output
from ..core.tvrs import TVRSParamsMixin, TVRSTaskConfig
from ..pmc.ingest import IngestPMCStatCounters
from .ingest import IngestQPSData


@dataclass
class QPSPlotConfig(Config):
    show_errorbars: bool = True
    #: Common plot parameterization options
    parameterize_options: TVRSTaskConfig = field(default_factory=TVRSTaskConfig)


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
        merged = pl.concat((loader.df.get() for loader in self.all_runs), how="vertical", rechunk=True)
        # Backward compatible with missing message_count column
        if "message_count" in merged.columns:
            merged = merged.with_columns(pl.col("message_count").fill_null(0))
        self.merged_df.assign(merged)

    def get_parameter_columns(self):
        # Note that all benchmarks must have the same set of parameter keys.
        # This is enforced during configuration
        all_bench = self.session.all_benchmarks()
        assert len(all_bench) > 0
        return list(all_bench[0].parameters.keys())


class QPSPlot(TVRSParamsMixin, PlotTask):
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

    @output
    def summary_table(self):
        return Target(self, f"tbl-summary")

    def outputs(self):
        # Dynamically generate the output map
        yield from super().outputs()
        for scenario in self.get_param_axis("scenario"):
            yield (scenario, PlotTarget(self, scenario))

    def run_plot(self):
        df = self.data.merged_df.get()

        ctx = self.make_param_context(df)
        ctx.suppress_const_params(keep=["target", "scenario"])
        ctx.derived_param_strcat("_flavor", ["variant", "runtime"], sep="/")
        ctx.relabel(default={"_flavor": "flavor/protection"})

        def plot_one_scenario(scenario, s_df):
            target = self.output_map[scenario]
            self.logger.info("Generate QPS plot for %s", scenario)
            with new_figure(target.paths()) as fig:
                ax = fig.subplots()
                sns.stripplot(s_df, x=ctx.r.target, y="qps", hue=ctx.r._flavor, dodge=True, ax=ax)
                ax.set_title(scenario)
                ax.set_ylabel("QPS")
                ax.set_xlabel("Target")

        with self.config_plotting_context():
            ctx.map_by_param("scenario", plot_one_scenario)

        self.logger.info("Generate QPS summary plot")
        flavor_combinations = len(ctx.df[ctx.r._flavor].unique())
        palette = sns.color_palette(n_colors=flavor_combinations)
        with self.config_plotting_context(font_scale=0.4):
            with new_facet(self.summary_plot.paths(), ctx.df, col=ctx.r.scenario, col_wrap=4) as facet:
                facet.map_dataframe(sns.stripplot,
                                    x=ctx.r.target,
                                    y="qps",
                                    dodge=True,
                                    hue=ctx.r._flavor,
                                    palette=palette)
                facet.add_legend()

        # Generate a summary of the results in CSV with mean/std
        (ctx.df.group_by([ctx.r.target, ctx.r._flavor, ctx.r.scenario
                          ]).agg(pl.col("qps").mean().alias("qps-mean"),
                                 pl.col("qps").std().alias("qps-std")).write_csv(self.summary_table.single_path()))


class LatencyPlot(TVRSParamsMixin, PlotTask):
    """
    Simple QPS plot that shows the Latency summary distribution on the Y axis and
    target ABI on the X axis.
    The distribution among iterations is shown with a box plot.

    This generates a plot for each different scenario.
    """
    task_namespace = "qps"
    task_name = "latency-plot"
    public = True

    rc_params = {"axes.labelsize": "large", "font.size": 6, "xtick.labelsize": 5}

    @dependency
    def data(self):
        return LoadQPSData(self.session, self.analysis_config)

    def outputs(self):
        # Dynamically generate the output map
        yield from super().outputs()
        for scenario in self.get_param_axis("scenario"):
            yield (scenario, PlotTarget(self, scenario))

    def run_plot(self):
        df = self.data.merged_df.get()
        # nsec to msec
        df = df.with_columns(
            pl.col("latency50") / 1000,
            pl.col("latency90") / 1000,
            pl.col("latency95") / 1000,
            pl.col("latency99") / 1000)

        ctx = self.make_param_context(df)
        ctx.suppress_const_params(keep=["target", "scenario"])
        ctx.derived_param_strcat("_flavor", ["variant", "runtime"], sep="/")
        ctx.relabel(default={"_flavor": "flavor/protection"})
        ctx.melt(value_vars=["latency50", "latency90", "latency95", "latency99"],
                 variable_name="percentile",
                 value_name="latency")
        palette = ctx.build_palette_for("_flavor")

        def plot_one_scenario(scenario, s_df):
            target = self.output_map[scenario]
            self.logger.info("Generate Latency plot for %s", scenario)
            with new_facet(target.paths(), s_df, col="percentile", col_wrap=2) as facet:
                facet.map_dataframe(sns.boxplot, x=ctx.r.target, y="latency", hue=ctx.r._flavor, palette=palette)
                facet.add_legend()
                facet.fig.subplots_adjust(top=0.9)
                facet.fig.suptitle(scenario)

        with self.config_plotting_context():
            ctx.map_by_param("scenario", plot_one_scenario)


class QPSByMsgSizePlot(TVRSParamsMixin, PlotTask):
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
    rc_params = {"axes.labelsize": "large", "font.size": 8, "xtick.labelsize": 10, "ytick.labelsize": 10}

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

    def tvrs_config(self):
        return self.config.parameterize_options

    def gen_msgsize_plot(self, ctx, err_conf):
        self.logger.info("Generate QPS vs MsgSize plot")
        with new_figure(self.qps_plot.paths()) as fig:
            ax = fig.subplots()
            sns.lineplot(ctx.df,
                         x="req_size",
                         y="qps",
                         hue=ctx.r.target,
                         style=ctx.r._flavor,
                         markers=True,
                         dashes=True,
                         ax=ax,
                         estimator="mean",
                         err_style="band",
                         errorbar=err_conf)
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(ctx.df["req_size"].unique()))
            ax.set_xlabel("Request size")
            ax.set_ylabel("QPS")

    def gen_overhead_plot(self, ctx, err_conf):
        self.logger.info("Generate QPS vs MsgSize overhead plot")
        with new_figure(self.overhead_plot.paths()) as fig:
            ax = fig.subplots()
            sns.lineplot(ctx.df,
                         x="req_size",
                         y="qps_overhead",
                         hue=ctx.r.target,
                         style=ctx.r._flavor,
                         markers=True,
                         dashes=True,
                         ax=ax,
                         estimator="mean",
                         err_style="band",
                         errorbar=err_conf)
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(ctx.df["req_size"].unique()))
            ax.set_xlabel("Request size")
            ax.set_ylabel("% Overhead")

    def gen_overhead_boxes(self, ctx, err_conf):
        self.logger.info("Generate QPS vs MsgSize overhead box plot")
        palette = ctx.build_palette_for("_flavor")
        with new_facet(self.overhead_box_plot.paths(), ctx.df, col=ctx.r.target) as facet:
            facet.map_dataframe(sns.boxplot,
                                x="req_size",
                                y="qps_overhead",
                                hue=ctx.r._flavor,
                                log_scale=(2, None),
                                palette=palette,
                                native_scale=True)
            facet.set_axis_labels(x_var="Request size", y_var="% Overhead")
            for ax in facet.axes.flatten():
                ax.set_xticks(sorted(ctx.df["req_size"].unique()))
            facet.add_legend()
            self.adjust_legend_on_top(facet.fig, loc="lower left")

    def run_plot(self):
        df = self.data.merged_df.get()

        ctx = self.make_param_context(df)
        ctx.suppress_const_params(keep=["target", "scenario"])
        ctx.derived_param_strcat("_flavor", ["variant", "runtime"], sep="/")
        ctx.compute_overhead(metrics=["qps"])
        ctx.relabel(default={"_flavor": "flavor/protection"})

        # Honor error bars configuration
        err_conf = None
        if self.config.show_errorbars:
            err_conf = ("pi", 90)
        with self.config_plotting_context():
            self.gen_msgsize_plot(ctx, err_conf)
            self.gen_overhead_plot(ctx, err_conf)
            self.gen_overhead_boxes(ctx, err_conf)

        # Generate summary of the overheads in CSV with mean/std
        (ctx.df.group_by(["req_size", ctx.r.target, ctx.r._flavor]).agg(
            pl.col("qps_overhead").mean().alias("qps-overhead-mean"),
            pl.col("qps_overhead").std().alias("qps-overhead-std")).sort(["req_size", ctx.r.target, ctx.r._flavor
                                                                          ]).write_csv(self.overhead_tbl.single_path()))


@dataclass
class QPSOverheadPlotConfig(QPSPlotConfig):
    #: Control which parameterization axis is used for the L/R Y axis selection
    facet_column: Optional[str] = None
    #: Label for the baseline combination
    baseline_label: Optional[str] = "baseline"


class QPSOverheadPlot(TVRSParamsMixin, PlotTask):
    """
    QPS plot that shows the QPS metric and the relative % overhead respectively on
    on the left and right Y axes, the target ABI on the X axis.

    This generates a bar plot for each different scenario.
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
        for scenario in self.get_param_axis("scenario"):
            yield (scenario, PlotTarget(self, scenario))
            yield (f"tbl-{scenario}", Target(self, f"tbl-{scenario}"))

    def tvrs_config(self):
        return self.config.parameterize_options

    def plot_one_scenario(self, ctx, scenario, s_df):
        # Honor error bars configuration
        if self.config.show_errorbars:
            err_conf = ("sd", )
        else:
            err_conf = None
        target = self.output_map[scenario]
        self.logger.info("Generate condensed QPS overhead plot for %s", scenario)
        with new_figure(target.paths()) as fig:
            ax = fig.subplots()
            axr = ax.twinx()
            axr.grid(False)

            mean_df = s_df.group_by(ctx.r.target).mean().sort("param_weight")
            x_labels = mean_df[ctx.r.target]
            x_pos = np.arange(len(x_labels))

            abs_bars = ax.bar(x_pos - 0.125, mean_df["qps"], color="r", width=0.25)
            pct_bars = axr.bar(x_pos + 0.125, mean_df["qps_overhead"], color="b", width=0.25)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)

            if self.config.facet_column:
                facet_labels = s_df[self.config.facet_column].unique()
                assert len(facet_labels) == 1
                facet_label = facet_labels[0]
                ax.set_title(f"{self.config.facet_column} = {facet_label}")

            ax.set_xlabel("Target")
            ax.set_ylabel("QPS (messages/sec)")
            axr.set_ylabel(f"% Overhead relative to {self.config.baseline_label}")
            ax.legend(handles=[abs_bars, pct_bars],
                      labels=["QPS (msg/s)", "% Overhead"],
                      bbox_to_anchor=(0, 1.1),
                      loc="lower left",
                      ncols=4)

        table_target = self.output_map[f"tbl-{scenario}"]
        mean_df.write_csv(table_target.single_path())

    def run_plot(self):
        df = self.data.merged_df.get()

        ctx = self.make_param_context(df)
        # If we have an axis to use for the facet grid, we need to keep it
        keep_params = ["target", "scenario"]
        if self.config.facet_column:
            if self.config.facet_column not in ctx.base_params:
                self.logger.error("Configured facet columns parameter '%s' is missing", self.config.facet_column)
                raise RuntimeError("Invalid config")
            keep_params.append(self.config.facet_column)
        ctx.suppress_const_params(keep=keep_params)
        # Now merge all the parameter columns, except facet_column into
        # the derived _label param
        other_params = list(ctx.params)
        other_params.remove("scenario")
        if self.config.facet_column in other_params:
            other_params.remove(self.config.facet_column)
        ctx.derived_param_strcat("_label", other_params, sep="\n")
        ctx.compute_overhead(metrics=["qps"], inverted=True)
        ctx.relabel(default={"_label": "label"})

        self.logger.info("Generate condensed QPS overhead plots with "
                         "facet_columns=%s, params=%s", self.config.facet_column, other_params)
        with self.config_plotting_context():
            ctx.map_by_param("scenario", lambda s_id, s_df: self.plot_one_scenario(ctx, s_id, s_df))


@dataclass
class QPSPmcConfig(QPSPlotConfig):
    #: Filter only the given subset of columns/counters
    metrics_filter: Optional[List[str]] = None


class QPSPerfCountersPlot(TVRSParamsMixin, PlotTask):
    """
    Generate mixed qps/pmc metrics when hardware performance counters data is available.
    """
    task_config_class = QPSPmcConfig
    task_namespace = "qps"
    task_name = "pmc-metrics"
    public = True

    derived_metrics = {
        "ex_entry_per_msg": {
            "requires": ["executive_entry", "message_count"],
            "expr": (pl.col("executive_entry") / pl.col("message_count"))
        },
        "ex_entry_per_byte": {
            "requires": ["executive_entry", "message_count", "resp_size"],
            "expr": (pl.col("executive_entry") / (pl.col("message_count") * pl.col("resp_size")))
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

    @output
    def qps_table(self):
        return PlotTarget(self, "tbl-metrics")

    def tvrs_config(self):
        return self.config.parameterize_options

    def run_plot(self):
        if self.pmc is None:
            self.logger.info("Skip task %s: missing PMC data", self)
            return

        # Merge the counter data from everywhere
        pmc_df = pl.concat([loader.df.get() for loader in self.pmc]).with_columns(pl.col("iteration").cast(pl.Int32))
        assert len(pmc_df.filter(iteration=-1)) == 0, "Missing iteration number on PMC frame"
        # Join with the QPS data
        qps_df = self.qps_data.merged_df.get().with_columns(pl.col("iteration").cast(pl.Int32))
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
                    self.logger.info("Skip derived metric %s: requires missing column %s", name, required_column)
                    has_cols = False
                    break
            if not has_cols:
                continue
            df = df.with_columns((spec["expr"]).alias(name))
            found_metrics.append(name)

        if not found_metrics:
            self.logger.info("Skipping plot, no data")
            return

        # XXX aggregate over iterations?
        ctx = self.make_param_context(df)
        ctx.derived_param_strcat("_flavor", ["variant", "runtime"], sep="/")
        ctx.melt(value_vars=found_metrics, variable_name="metric", value_name="value")
        ctx.relabel(default={"_flavor": "flavor/protection"})

        # Filter the counters based on configuration
        if self.config.metrics_filter:
            ctx.df = ctx.df.filter(pl.col("metric").is_in(self.config.metrics_filter))

        palette = ctx.build_palette_for("_flavor")
        self.logger.info("Generate QPS PMC metrics")
        with new_facet(self.qps_metrics.paths(),
                       ctx.df,
                       col=ctx.r.scenario,
                       row="metric",
                       sharex="col",
                       sharey="row",
                       margin_titles=True,
                       aspect=0.85) as facet:
            facet.map_dataframe(sns.barplot, x=ctx.r.target, y="value", hue=ctx.r._flavor, dodge=True, palette=palette)
            facet.add_legend()

        group_cols = [ctx.r.target, ctx.r.scenario, ctx.r._flavor, "metric"]
        ctx.df.group_by(group_cols).mean().write_csv(self.qps_table.single_path())
