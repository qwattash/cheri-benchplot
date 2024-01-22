
from dataclasses import dataclass, field
from typing import Dict

import polars as pl
import seaborn as sns

from ..core.analysis import AnalysisTask
from ..core.artefact import ValueTarget
from ..core.config import Config
from ..core.plot import new_facet, new_figure, PlotTask, PlotTarget
from ..core.task import dependency, output
from .ingest import IngestWRKData

class LoadWRKData(AnalysisTask):
    """
    Load and merge data from all WRK runs.
    """
    task_namespace = "wrk"
    task_name = "merge"

    @dependency
    def all_runs(self):
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(IngestWRKData)
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


class RPSPlot(PlotTask):
    """
    Simple plot that shows the requests per second metric on the Y axis and the target ABI
    on the X axis.
    The distribution among iterations is shown with a stripplot.
    """
    task_namespace = "wrk"
    task_name = "rps-plot"
    public = True

    @dependency
    def data(self):
        return LoadWRKData(self.session, self.analysis_config)

    def outputs(self):
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
            (pl.col("requests") * 10**6 / pl.col("duration")).alias("rps")
        )

        for scenario, s_df in df.groupby("scenario"):
            target = self.output_map[scenario]
            self.logger.info("Generate WRK plot for %s", scenario)
            with new_figure(target.paths()) as fig:
                ax = fig.subplots()
                sns.stripplot(s_df, x="target", y="rps", hue="flavor/protection",
                              dodge=True, ax=ax)
                ax.set_title(scenario)
                ax.set_ylabel("RPS")
                ax.set_xlabel("Target")


class RPSByMsgSizePlot(PlotTask):
    """
    Plot the RPS metric across different message sizes to show how the overhead scales.

    Note that the overhead plot is normalized to the mean of the baseline benchmark run.
    The overhead is inverted here, because we want positive % numbers to indicate
    loss in performance.
    """
    task_namespace = "wrk"
    task_name = "rps-by-msgsize-plot"
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
        return LoadWRKData(self.session, self.analysis_config)

    @output
    def rps_plot(self):
        return PlotTarget(self, "abs")

    @output
    def overhead_plot(self):
        return PlotTarget(self, "rel")

    @output
    def overhead_grid(self):
        return PlotTarget(self, "rel-grid")

    def run_plot(self):
        df = self.data.merged_df.get()
        params = self.data.get_parameter_columns()
        # Drop the scenario column from params as we use the message size axis instead
        params.remove("scenario")

        df = df.with_columns(
            pl.col("dataset_gid").map_elements(self.g_uuid_to_label).alias("target"),
            pl.concat_str([pl.col(p) for p in params], separator=" ").alias("flavor/protection"),
            (pl.col("requests") * 10**6 / pl.col("duration")).alias("rps")
        )

        sizes = sorted(df["request_size"].unique())
        x_labels = [f"{s / 1024:.1f}KiB" for s in sizes]

        self.logger.info("Generate WRK RPS by message size plot")
        with new_figure(self.rps_plot.paths()) as fig:
            ax = fig.subplots()
            sns.lineplot(df, x="request_size", y="rps", hue="target",
                         style="flavor/protection", markers=True, dashes=True, ax=ax,
                         err_style="band", errorbar=("pi", 90))
            ax.set_xscale("symlog", base=2, linthresh=0.01)
            ax.set_xticks(sorted(df["request_size"].unique()), labels=x_labels,
                          rotation=-45, ha="left")
            ax.set_xlim(0, df["request_size"].max())
            ax.set_xlabel("Request size")
            ax.set_ylabel("Requests / sec")

        self.logger.info("Generate WRK overhead plot")
        baseline = self.baseline_slice(df)
        baseline_rps = baseline.select(["rps", "scenario"]).group_by("scenario").mean().rename(dict(rps="rps_baseline"))
        # Create the overhead column and drop the baseline data
        df = df.join(baseline_rps, on="scenario").with_columns(
            ((pl.col("rps_baseline") - pl.col("rps")) * 100 / pl.col("rps_baseline")).alias("overhead")
        ).join(baseline, on="dataset_id", how="anti")
        with new_figure(self.overhead_plot.paths()) as fig:
            ax = fig.subplots()
            sns.lineplot(df, x="request_size", y="overhead", hue="target",
                         style="flavor/protection", markers=True, dashes=True, ax=ax,
                         err_style="band", errorbar=("pi", 90))
            ax.set_xscale("symlog", base=2, linthresh=0.01)
            ax.set_xticks(sorted(df["request_size"].unique()), labels=x_labels,
                          rotation=-45, ha="left")
            ax.set_xlim(0, df["request_size"].max())
            ax.set_xlabel("Request size")
            ax.set_ylabel("Requests / sec")

        # Show the overhead in different grid cells for each flavor/protection combination
        self.logger.info("Generate WRK overhead grid")
        df = df.with_columns(
            pl.col("runtime").alias("protection"),
            pl.col("variant").alias("flavor"))
        with new_facet(self.overhead_grid.paths(), df, row="protection", col="flavor") as facet:
            facet.map_dataframe(sns.lineplot, x="request_size", y="overhead", hue="target",
                                err_style="band", errorbar=("pi", 90), marker="o")
            for ax in facet.axes.flat:
                ax.set_xscale("symlog", base=2, linthresh=512)
                ax.set_xticks(sorted(df["request_size"].unique()), labels=x_labels,
                              rotation=-45, ha="left")
                ax.margins(0.05)
            facet.set_xlabels("Request size")
            facet.set_ylabels("% Overhead")
            facet.add_legend()
