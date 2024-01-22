from dataclasses import dataclass

import polars as pl
import seaborn as sns

from ..core.analysis import AnalysisTask
from ..core.artefact import Target, ValueTarget
from ..core.config import Config
from ..core.plot import new_facet, new_figure, PlotTask, PlotTarget
from ..core.task import dependency, output
from .ingest import IngestQPSData

@dataclass
class QPSPlotConfig(Config):
    show_errorbars: bool = True


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
            sns.lineplot(df, x="reqSize", y="qps", hue="target", style="flavor/protection",
                         markers=True, dashes=True, ax=ax, estimator="mean",
                         err_style="band", errorbar=err_conf)
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(df["reqSize"].unique()))
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
            sns.lineplot(df, x="reqSize", y="overhead", hue="target", style="flavor/protection",
                         markers=True, dashes=True, ax=ax, estimator="mean",
                         err_style="band", errorbar=err_conf)
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(df["reqSize"].unique()))
            ax.set_xlabel("Request size")
            ax.set_ylabel("% Overhead")

        # Print out the mean overhead in tabular form
        idcols = ["reqSize", "target", "flavor/protection"]
        tbl = df.group_by(idcols).mean().select(idcols + ["overhead"]).sort(idcols)
        with open(self.overhead_tbl.single_path(), "w+") as fp:
            tbl.write_csv(fp)
