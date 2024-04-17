from dataclasses import dataclass

import polars as pl

from .analysis import AnalysisTask
from .config import Config


@dataclass
def TVRSTaskConfig(Config):
    #: Weigth for determining the order of labels based on the parameters
    parameter_weight: Optional[Dict[str, Dict[str, int]]] = None
    #: Relabel parameter axes
    parameter_names: Optional[Dict[str, str]] = None
    #: Relabel parameter axes values
    parameter_labels: Optional[Dict[str, Dict[str, str]]] = None


class TVRSParamsTask(AnalysisTask):
    """
    Base class for analysis tasks that use the common parameterization:
    target: the CHERI compilation target, e.g. purecap/hybrid
    variant: benchmark variant, e.g. package built with stack zero-init
    runtime: run-time configuration, e.g. with/without temporal memory safety
    scenario: benchmark-specific parameters
    """
    task_namespace = "analysis"
    task_config_class = TVRSTaskConfig

    TVRS_PARAMETERS = ["target", "variant", "runtime", "scenario"]

    def gen_target_column(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate the "target" parameter column from the dataset_gid.
        This uses the `name` field from the instance configuration as the display name.
        """
        return df.with_columns(pl.col("dataset_gid").map_elements(self.g_uuid_to_label).alias("target"))

    def get_parameter_columns(self):
        """
        Fetch and sanitize parameter columns.
        Note that this includes the target column, the caller must ensure that
        the column exists.
        Note that we allow extra parameterization axes to exist, we only want to
        ensure that the TVRS params exist.

        Note that all benchmarks must have the same set of parameter keys,
        this is enforced during configuration.
        """
        all_bench = self.session.all_benchmarks()
        assert len(all_bench) > 0
        pkeys = list(all_bench[0].parameters.keys())
        if set(self.TVRS_PARAMETERS).intersection(pkeys) != set(self.TVRS_PARAMETERS):
            self.logger.error("Invalid parameterization, the following parameters are required: %s",
                              self.TVRS_PARAMETERS)
            raise RuntimeError("Configuration error")
        return pkeys

    def compute_overhead(self, df: pl.DataFrame, metric_columns: list[str], inverted: bool = False) -> pl.DataFrame:
        baseline_columns = {c: f"{c}_baseline" for c in metric_columns}
        baseline = self.baseline_slice(df)
        baseline_metrics = baseline.select(metric_columns +
                                           ["scenario"]).group_by("scenario").mean().rename(baseline_columns)
        # Create the overhead column and optionally drop the baseline data
        sign = -1 if inverted else 1
        overhead_expr = [
            ((pl.col(c) - pl.col(f"{c}_baseline")) * sign * 100 / pl.col(f"{c}_baseline")).alias(f"{c}_overhead")
            for c in metric_columns
        ]
        df = df.join(baseline_metrics, on="scenario").with_columns(
            *overhead_expr
        ).join(baseline, on="dataset_id", how="anti")
        return df

    def prepare_for_display(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Transform the dataframe to adjust displayed properties.

        This applies the plot configuration to rename parameter levels, axes and
        filters.
        """
        params = self.get_parameter_columns()

        # Compute the parameterization weight for consistent label ordering
        if self.config.parameter_weight:
            df = df.with_columns(pl.lit(0).alias("param_weight"))
            for name, mapping in self.config.parameter_weight.items():
                if name in df.columns:
                    self.logger.debug("Set weight for %s => %s", name, mapping)
                    df = df.with_columns(pl.col("param_weight") + pl.col(name).replace(mapping, default=0))
                else:
                    self.logger.warning("Skipping weight for parameter '%s', does not exist", name)

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

    def suppress_constant_params(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Suppress parameter axes with no variation from the dataframe.

        Note: this should occur before we relabel parameters.
        """
        hide_params = []
        params = self.get_parameter_columns()
        for p in params:
            if len(df[p].unique()) == 1:
                hide_params.append(p)
        return df.drop(hide_params)

    def merge_parameter_axes(self, df: pl.DataFrame, name: str, to_combine: list[str], sep=" ") -> pl.DataFrame:
        """
        Merge two or more parameter levels into a single column.
        This is useful when there are too many dimensions to show and we need to
        assign a column to some plot feature (e.g. hue).
        """
        df = df.with_columns(pl.concat_str([pl.col(p) for p in to_combine], separator=sep).alias(name))
        return df
