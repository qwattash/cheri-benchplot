import re
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from marshmallow.validate import And, Predicate
from matplotlib.ticker import AutoMinorLocator

from pycheribenchplot.core.analysis import PlotTask
from pycheribenchplot.core.config import Config, ConfigPath
from pycheribenchplot.core.plot import new_figure
from pycheribenchplot.core.task import (AnalysisTask, DataFrameTarget, DataGenTask, LocalFileTarget)
from pycheribenchplot.kernel_vuln.model import CheriBSDAdvisories, Marker


@dataclass
class CheriBSDKernelVulnConfig(Config):
    #: Vulnerability classification input csv.
    #: This is required.
    classification_file: ConfigPath = field(
        metadata={
            "validate":
            And(Predicate("exists", error="File does not exist"), Predicate("is_file",
                                                                            error="File is not regular file"))
        })


class CheriBSDKernelVuln(DataGenTask):
    """
    This task interprets and freezes a table from the CSV kernel vulnerability classification.
    The input is compiled separately but this is used to ingest it for plotting in a way that
    is better controlled and automated than google sheets or excel.
    """
    public = True
    task_namespace = "kernel-vuln"
    task_name = "cheribsd-sa-classification"
    task_config_class = CheriBSDKernelVulnConfig

    def _output_target(self) -> LocalFileTarget:
        return LocalFileTarget.from_task(self, use_data_root=True, ext="csv")

    def _normalize_input_colname(self, column_name: str) -> str:
        """
        Cleanup a column name by removing all weird charaters and make lowercase
        """
        cleaned = re.sub(r"[()?]", "", column_name)
        cleaned = re.sub(r"[\n\s/]", "_", cleaned)
        cleaned = re.sub(r"_+", "_", cleaned)
        return cleaned.lower()

    @property
    def task_id(self):
        return f"{self.task_namespace}.{self.task_name}"

    def run(self):
        self.logger.debug("Reading vulnerability classification from %s", self.config.classification_file)
        input_df = pd.read_csv(self.config.classification_file, header=[0, 1])
        # Cleanup column index because google sheets generates a quite mangled one
        # We collapse everything on a single level and normalize the names.
        normalized_names = []
        for col_tuple in input_df.columns:
            if col_tuple[1].startswith("Unnamed"):
                normalized_names.append(self._normalize_input_colname(col_tuple[0]))
            else:
                normalized_names.append("cheri_" + self._normalize_input_colname(col_tuple[1]))
        input_df.columns = pd.Index(normalized_names)
        input_df.set_index(["date", "advisory"], inplace=True)
        df = CheriBSDAdvisories.to_schema(self.session).validate(input_df)
        df.to_csv(self._output_target().path)

    def outputs(self):
        yield "advisories", self._output_target()


class CheriBSDKernelVulnLoad(AnalysisTask):
    """
    Load task that grabs the cleaned-up SA classification file from the datagen task.
    """
    task_namespace = "kernel-vuln"
    task_name = "load-sa"

    def run(self):
        # XXX the datagen task should be made session-scoped so that we don't have to
        # pass a random benchmark there.
        target = CheriBSDKernelVuln(self.session.benchmark_matrix.iloc[0, 0], None).output_map["advisories"]
        df = pd.read_csv(target.path, index_col=["date", "advisory"])
        df = CheriBSDAdvisories.to_schema(self.session).validate(df)

        self._df = df
        # self.output_map["df"].df = df

    def outputs(self):
        yield "df", DataFrameTarget(CheriBSDAdvisories, self._df)


class CheriBSDAdvisoriesHistory(PlotTask):
    """
    Generate plot of the distribution of advisories over time.
    """
    public = True
    task_namespace = "kernel-vuln"
    task_name = "plot-sa-timeline"

    def dependencies(self):
        self._sa_load = CheriBSDKernelVulnLoad(self.session, self.analysis_config)
        yield self._sa_load

    def run(self):
        df = self._sa_load.output_map["df"].df.copy()

        df["year"] = df.index.get_level_values("date").year
        df["advisory"] = df.index.get_level_values("advisory")
        # Some intermediate columns
        df["is_memory_safety"] = df["not_memory_safety_bug"].isna()
        df["is_mitigated_spatial"] = df["cheri_does_not_help"].isna()
        df["is_mitigated_temporal"] = (df["cheri_does_not_help"] == Marker.TEMPORAL.value)
        df["is_mitigated"] = df["is_mitigated_spatial"] | df["is_mitigated_temporal"]

        display_columns = ["advisory", "is_memory_safety", "is_mitigated_spatial", "is_mitigated"]
        plot_df = df.groupby("year").agg({
            "advisory": "size",
            "is_memory_safety": "sum",
            "is_mitigated_spatial": "sum",
            "is_mitigated": "sum",
        })[display_columns].reset_index().melt("year", var_name="hue", value_name="#advisories")
        with new_figure(self._plot_output().path) as fig:
            ax = fig.subplots()
            ax.grid(axis="y", linestyle="--")
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            sns.lineplot(ax=ax, data=plot_df, x="year", y="#advisories", hue="hue", palette="pastel")

    def outputs(self):
        yield "plot-timeline", self._plot_output()
