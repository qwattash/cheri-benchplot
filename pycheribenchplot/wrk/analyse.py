
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
            yield (f"tbl-{scenario}", Target(self, f"tbl-{scenario}"))

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
            table_target = self.output_map[f"tbl-{scenario}"]
            s_df = s_df.with_columns(
                (pl.col("bytes") / pl.col("duration")).alias("mbps")
            )
            s_df.group_by(["target", "flavor/protection"]).mean().write_csv(table_target.single_path())


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


@dataclass
class WRKPlotConfig(Config):
    show_errorbars: bool = True
    #: Weigth for determining the order of labels based on the parameters
    parameter_weight: Optional[Dict[str, Dict[str, int]]] = None
    #: Relabel parameter axes
    parameter_names: Optional[Dict[str, str]] = None
    #: Relabel parameter axes values
    parameter_labels: Optional[Dict[str, Dict[str, str]]] = None
    #: Control which parameterization axis is used for the facet grid columns
    facet_column: Optional[str] = None
    #: Label for the baseline combination
    baseline_label: Optional[str] = "baseline"


class WRKOverheadPlot(PlotTask):
    """
    Plot that shows the requests/sec metric and the relative % overhead respectively on
    on the left and right Y axes, the target ABI on the X axis.

    This generates a bar plot for each different scenario.
    """
    task_config_class = WRKPlotConfig
    task_namespace = "wrk"
    task_name = "rps-overhead-plot"
    public = True

    # Plot styling overrides
    rc_params = {
        "axes.labelsize": "large",
        "font.size": 8,
        "xtick.labelsize": 8,
    }

    @dependency
    def data(self):
        return LoadWRKData(self.session, self.analysis_config)

    def outputs(self):
        # Dynamically generate the output map
        yield from super().outputs()
        for scenario in self.get_scenarios():
            yield (scenario, PlotTarget(self, scenario))
            yield (f"tbl-{scenario}", Target(self, f"tbl-{scenario}"))

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

        # Duration is in microseconds
        df = df.with_columns(
            (pl.col("requests") * 10**6 / pl.col("duration")).alias("rps")
        )

        # Compute the overhead
        # IMPORTANT: this must occur before we do any of the plot-specific renaming
        baseline = self.baseline_slice(df)
        baseline_rps = baseline.select(["rps", "scenario"]).group_by("scenario").mean().rename(dict(rps="rps_baseline"))
        # Create the overhead column and optionally drop the baseline data
        df = df.join(baseline_rps, on="scenario").with_columns(
            ((pl.col("rps_baseline") - pl.col("rps")) * 100 / pl.col("rps_baseline")).alias("overhead"),
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
        if self.config.facet_column is not None:
            if self.config.facet_column not in params:
                self.logger.warning("Configured facet columns key '%s' is missing, "
                                    "maybe it was renamed by QPSPlotConfig.parameter_names. "
                                    "Note that column renaming occurs before this step, so the "
                                    "QPSOverheadPlotConfig.facet_columns key must contain the "
                                    "new parameter name.", self.config.facet_column)
            params.remove(self.config.facet_column)

        # Check if there is variation on the flavor/protection axes, otherwise drop them
        drop_params = []
        for p in params:
            if len(df.select(p).unique()) == 1:
                self.logger.debug("No variation on the '%s' axis detected, drop parameterization axis in plot", p)
                drop_params.append(p)
        params = [p for p in params if p not in drop_params]

        # Honor error bars configuration
        if self.config.show_errorbars:
            err_conf = ("sd",)
        else:
            err_conf = None

        self.logger.info("Generate RPS overhead plot with facet_columns=%s, params=%s",
                         self.config.facet_column, params)

        # Now merge all the parameter columns into the target label
        df = df.with_columns(
            pl.concat_str([pl.col(p) for p in sorted(params)], separator="\n").alias("target")
        )

        for scenario, s_df in df.groupby("scenario"):
            target = self.output_map[scenario]
            self.logger.info("Generate RPS overhead plot for %s", scenario)
            with new_figure(target.paths()) as fig:
                ax = fig.subplots()
                axr = ax.twinx()
                axr.grid(False)

                mean_df = s_df.group_by("target").mean().sort("param_weight")
                x_labels = mean_df["target"]
                x_pos = np.arange(len(x_labels))

                abs_bars = ax.bar(x_pos - 0.125, mean_df["rps"], color="r", width=0.25)
                pct_bars = axr.bar(x_pos + 0.125, mean_df["overhead"], color="b", width=0.25)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels)

                if self.config.facet_column:
                    facet_labels = s_df[self.config.facet_column].unique()
                    assert len(facet_labels) == 1
                    facet_label = facet_labels[0]
                    ax.set_title(f"{self.config.facet_column} = {facet_label}")

                ax.set_xlabel("Target")
                ax.set_ylabel("RPS (requests/sec)")
                axr.set_ylabel(f"% Overhead relative to {self.config.baseline_label}")
                ax.legend(handles=[abs_bars, pct_bars], labels=["RPS (req/s)", "% Overhead"],
                          bbox_to_anchor=(0, 1.1), loc="lower left", ncols=4)
            table_target = self.output_map[f"tbl-{scenario}"]
            mean_df.write_csv(table_target.single_path())
