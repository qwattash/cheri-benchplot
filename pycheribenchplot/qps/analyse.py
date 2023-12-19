
import polars as pl
import seaborn as sns

from ..core.analysis import AnalysisTask
from ..core.artefact import ValueTarget
from ..core.plot import new_facet, new_figure, PlotTask, PlotTarget
from ..core.task import dependency, output
from .ingest import IngestQPSData

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
        sns.set_theme()
        params = self.data.get_parameter_columns()
        # Drop the scenario column from params as we use it as a facet axis
        params.remove("scenario")

        # Merge dataset description from all parameters
        df = df.with_columns(
            pl.col("dataset_gid").map_elements(self.g_uuid_to_label).alias("target"),
            pl.concat_str([pl.col(p) for p in params], separator=" ").alias("protection")
        )

        for scenario, s_df in df.groupby("scenario"):
            target = self.output_map[scenario]
            with new_figure(target.paths()) as fig:
                ax = fig.subplots()
                sns.boxplot(s_df, x="target", y="qps", hue="protection", ax=ax)
                ax.set_title(scenario)
                ax.set_ylabel("QPS")
                ax.set_xlabel("Target")

        # Make the summary plot
        sns.set_theme(font_scale=0.4)
        with new_facet(self.summary_plot.paths(), df, col="scenario", col_wrap=4) as facet:
            facet.map_dataframe(sns.boxplot, x="target", y="qps", hue="protection")
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
            pl.concat_str([pl.col(p) for p in params], separator=" ").alias("protection"),
            pl.col("latency50") / 1000,
            pl.col("latency90") / 1000,
            pl.col("latency95") / 1000,
            pl.col("latency99") / 1000
        )
        df = df.melt(id_vars=["target", "protection", "scenario"],
                     value_vars=["latency50", "latency90", "latency95", "latency99"],
                     variable_name="percentile",
                     value_name="latency")

        plot_rc = {
            "axes.labelsize": "large",
            "font.size": 6,
            "xtick.labelsize": 5
        }

        for scenario, s_df in df.groupby("scenario"):
            target = self.output_map[scenario]
            with sns.plotting_context(rc=plot_rc):
                with new_facet(target.paths(), df, col="percentile", col_wrap=2) as facet:
                    facet.map_dataframe(sns.boxplot, x="target", y="latency",
                                        hue="protection", palette=sns.color_palette())
                    facet.add_legend()
                    facet.fig.subplots_adjust(top=0.9)
                    facet.fig.suptitle(scenario)
