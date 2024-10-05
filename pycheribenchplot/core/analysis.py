from dataclasses import dataclass, field
from pathlib import Path
from typing import Type
from uuid import UUID
from warnings import warn

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import scipy.stats as scs

from .artefact import BenchmarkIterationTarget, DataFrameTarget
from .benchmark import Benchmark
from .config import AnalysisConfig, Config, InstanceConfig
from .model import DataModel
from .task import ExecutionTask, SessionTask, dependency


class AnalysisTask(SessionTask):
    """
    Analysis tasks that perform anythin from plotting to data checks and transformations.
    This is the base class for all public analysis steps that are allocated by the session.
    Analysis tasks are not necessarily associated to a single benchmark. In general they reference the
    current session and analysis configuration, subclasses may be associated to a benchmark context.

    Note that this currently assumes that tasks with the same name are not issued
    more than once for each benchmark run UUID. If this is violated, we need to
    change the task ID generation.
    """
    task_namespace = "analysis"

    def __init__(self, session: "Session", analysis_config: AnalysisConfig, task_config: Config | None = None):
        super().__init__(session, task_config=task_config)
        # Analysis configuration for this invocation
        self.analysis_config = analysis_config

        # Collect parameterization axes from the benchmark matrix
        all_bench = self.session.all_benchmarks()
        assert len(all_bench) > 0
        self._param_columns = list(all_bench[0].parameters.keys())

    def _do_mean_overhead(self, df, metric, extra_groupby, overhead_scale):
        """
        Compute the overhead assuming normal distribution of the "metric" data in the
        dataframe.

        The parameter axes from the benchmark configuration are used to group the
        dataframe to compute the statistic metrics.
        Note that this returns data in long-form, see :meth:`AnalysisTask.compute_overhead`.
        """
        if len(df.select(cs.ends_with("_std", "_baseline"))):
            self.logger.error("Can not compute mean overhead, conflicting columns with suffix '_std' or '_baseline'")
            raise RuntimeError("Invalid column names")

        baseline_sel = self.baseline_selector()
        param_columns = self.param_columns
        join_columns = list(set(self.param_columns) - set(baseline_sel.keys()))
        if extra_groupby:
            param_columns += extra_groupby
            join_columns += extra_groupby

        # Generate mean and std values for each group
        agg_selectors = [cs.by_name(metric).mean(), cs.by_name(metric).std().name.suffix("_std")]
        agg_rename = cs.ends_with("_std").name.suffix("_baseline")
        stats = df.group_by(param_columns).agg(*agg_selectors)
        # Find the baseline dataframe slice
        bs = stats.filter(**baseline_sel).with_columns(cs.by_name(metric).name.suffix("_baseline"), agg_rename)

        # Now we align the stats and bs frames to compute delta and overhead
        if join_columns:
            join_bs = bs.select(cs.ends_with("_baseline") | cs.by_name(join_columns))
            join_df = stats.join(join_bs, on=join_columns)
        else:
            # The selector may be empty if the baseline slice selects a specific
            # target/variant/runtime/scenario combination.
            # In this case, baseline is a single row which we replicate for every
            # parameter combination
            join_columns = list(baseline_sel.keys())
            _, right_df = pl.align_frames(stats, bs, on=join_columns)
            aligned_bs = right_df.select(cs.by_name(param_columns)
                                         | cs.ends_with("_baseline")).fill_null(strategy="forward")
            join_df = stats.join(aligned_bs, on=join_columns)
        assert join_df.shape[0] == stats.shape[0], "Unexpected join result"

        # Compute the absolute DELTA for each metric
        bs_val = f"{metric}_baseline"
        m_std = f"{metric}_std"
        bs_std = f"{metric}_std_baseline"
        delta = join_df.with_columns(
            pl.lit(metric + "_delta").alias("_metric_type"),
            pl.col(metric) - pl.col(bs_val), (pl.col(m_std).pow(2) + pl.col(bs_std).pow(2)).sqrt()).with_columns(
                # Mask the baseline slice with NaN
                pl.when(**baseline_sel).then(float("nan")).otherwise(pl.col(metric)).alias(metric),
                pl.when(**baseline_sel).then(float("nan")).otherwise(pl.col(m_std)).alias(m_std))

        # Compute the % Overhead for each metric
        rel_err_sum = (pl.col(m_std) / pl.col(metric)).pow(2) + (pl.col(bs_std) / pl.col(bs_val)).pow(2)
        ovh = delta.with_columns(
            pl.lit(metric + "_overhead").alias("_metric_type"),
            (pl.col(metric) / pl.col(bs_val)).mul(overhead_scale),
            ((pl.col(m_std) / pl.col(metric)).pow(2) +
             (pl.col(bs_std) / pl.col(bs_val)).pow(2)).alias("_tmp_sum_of_err_squared"),
        ).with_columns((pl.col(metric).abs() * pl.col("_tmp_sum_of_err_squared").sqrt()).alias(m_std)).with_columns(
            # Mask the baseline slice with NaN
            pl.when(**baseline_sel).then(float("nan")).otherwise(pl.col(metric)).alias(metric),
            pl.when(**baseline_sel).then(float("nan")).otherwise(pl.col(m_std)).alias(m_std))

        # Combine the stats, delta and ovh frame to have long-form data representation
        stats = stats.with_columns(pl.lit(metric).alias("_metric_type"))
        lf_stats = pl.concat([
            stats,
            delta.select(cs.all().exclude("^.*_baseline$")),
            ovh.select(cs.all().exclude("^.*_baseline$").exclude("^_tmp.*$"))
        ],
                             how="vertical",
                             rechunk=True)
        return lf_stats

    def _do_median_bootstrap(self, df, metric, extra_groupby, overhead_scale):
        """
        Compute the median overhead.

        Note that the median confidence intervals are computed using a multi-sample
        bootstrapping approach.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
        This should help to obtain a better confidence interval in the presence of an
        unknown distribution. The confidence level of the interval is configurable,
        defaulting to 0.95.

        The parameter axes from the benchmark configuration are used to group the
        dataframe to compute the statistic metrics.
        Data is returned in long-form, see :meth:`AnalysisTask.compute_overhead`.
        """
        baseline_sel = self.baseline_selector()
        param_columns = self.param_columns
        if extra_groupby:
            param_columns += extra_groupby
        join_columns = [*param_columns, "iteration"]

        assert not [c for c in df.columns if c.endswith("_right")], "Columns may not have suffix '_right'"

        # The baseline selector must specify a constraint on param_columns.
        # If the constraint does not fix every parameterization axis, we have a number of degrees of
        # freedom on the baseline.
        # Therefore, the baseline chunk must be further aligned with the dataframe in order to correctly
        # associate baseline values to each group.
        # In order to compute the degrees of freedom of the baseline, we filter-out all derived parameter
        # columns that do not vary within the baseline chunk.
        metric_b = f"{metric}_baseline"
        baseline = df.filter(**baseline_sel).with_columns(pl.col(metric).alias(metric_b)).select(
            metric_b, *join_columns)
        dof = list(
            baseline.select(cs.by_name(join_columns).n_unique()).transpose(
                include_header=True, header_name="column",
                column_names=["n_values"]).filter(pl.col("n_values") > 1)["column"])
        jdf = df.join(baseline, on=dof).select(~cs.ends_with("_right"))
        # Note that we expect the shape of the dataframe not to change
        assert jdf.shape[0] == df.shape[0], "Unexpected baseline join result"
        df = jdf

        ci_low_col = f"{metric}_low"
        ci_high_col = f"{metric}_high"

        def _bootstrap(chunk: "DataFrame", selectors, statistic_fn):
            data = tuple(chunk[s] for s in selectors)
            boot = scs.bootstrap(data, statistic_fn, vectorized=True, method="basic")
            stat_chunk = chunk.select(
                cs.by_name(param_columns).first(),
                pl.lit(statistic_fn(*data, axis=0)).alias(metric),
                pl.lit(boot.confidence_interval.low).alias(ci_low_col),
                pl.lit(boot.confidence_interval.high).alias(ci_high_col))
            return stat_chunk

        def _median_diff(left, right, axis=-1):
            return np.median(left, axis=axis) - np.median(right, axis=axis)

        def _median_overhead(left, right, axis=-1):
            # Note that we may have division by zero due to the right median being zero sometimes.
            # We don't care about it, but we need to propagate NaN in these cases.
            right_median = np.median(right, axis=axis)
            ratio = np.where(right_median != 0,
                             np.divide(np.median(left, axis=axis), right_median, where=(right_median != 0)), np.nan)
            return (ratio - 1) * overhead_scale

        grouped = df.group_by(*param_columns)
        # First, bootstrap the metric median
        # Note the "*" unpacking, this works around a limitation of map_groups that doesn't
        # like a list argument.
        stat = grouped.map_groups(lambda chunk: _bootstrap(chunk, [metric], np.median))
        stat = stat.with_columns(pl.lit("absolute").alias("_metric_type"))

        # Bootstrap delta median from the baseline and each other group.
        # Note that we force the baseline to baseline delta to 0.
        delta_stat = grouped.map_groups(lambda chunk: _bootstrap(chunk, [metric, metric_b], _median_diff))
        delta_stat = delta_stat.with_columns(pl.lit("delta").alias("_metric_type"))

        # Bootstrap the median overhead from the baseline and each other group.
        # Note that we force the baseline to baseline overhead to 0.
        ovh_stat = grouped.map_groups(lambda chunk: _bootstrap(chunk, [metric, metric_b], _median_overhead))
        ovh_stat = ovh_stat.with_columns(pl.lit("overhead").alias("_metric_type"))
        # Note: force the baseline-to-baseline overhead to NaN
        # ovh_stat = ovh_stat.with_columns(
        #     pl.when(**baseline_sel).then(np.nan).otherwise(pl.col(metric)).alias(metric),
        #     pl.when(**baseline_sel).then(np.nan).otherwise(pl.col(ci_low_col)).alias(ci_low_col),
        #     pl.when(**baseline_sel).then(np.nan).otherwise(pl.col(ci_high_col)).alias(ci_high_col)
        # )

        out_df = pl.concat([stat, delta_stat, ovh_stat], how="vertical", rechunk=True)
        return out_df

    @property
    def param_columns(self) -> list[str]:
        """
        Return the set of parameter names that are expected to be found in
        a dataframe containing data for this session.
        """
        return list(self._param_columns)

    def get_instance_config(self, g_uuid: str) -> InstanceConfig:
        """
        Helper to retreive an instance configuration for the given g_uuid.
        """
        return self.session.get_instance_configuration(g_uuid)

    def g_uuid_to_label(self, g_uuid: str) -> str:
        """
        Helper that maps group UUIDs to a human-readable label that describes the instance

        Deprecated, use the `target` parameter.
        """
        instance_config = self.get_instance_config(g_uuid)
        return instance_config.name

    def baseline_selector(self) -> dict[str, str]:
        """
        Generate a dictionary of selectors that identify the baseline data slice.

        These are suitable to use with polars dataframe filter().
        The baseline must be specified in the analysis configuration.
        """
        baseline_sel = self.analysis_config.baseline
        if baseline_sel is None:
            self.logger.error("Missing baseline selector in analysis configuration")
            raise ValueError("Invalid Configuration")

        if type(baseline_sel) == dict:
            # If we have the 'instance' parameter, replace it with the corresponding
            # dataset_gid
            if "instance" in baseline_sel:
                name = baseline_sel["instance"]
                for b in self.session.all_benchmarks():
                    if b.config.instance.name == name:
                        baseline_sel["dataset_gid"] = b.config.g_uuid
                        del baseline_sel["instance"]
                        break
                else:
                    self.logger.error("Invalid 'instance' value in baseline configuration")
                    raise ValueError("Invalid configuration")
            selectors = dict(baseline_sel)
        else:
            # Expect a UUID
            selectors = {"dataset_id": baseline_sel}
        return selectors

    def baseline_slice(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Extract the baseline cross section of the dataframe.
        Note that the dataframe may be missing some of the parameterization axes,
        assuming that there is no variation along that axis.

        The baseline must be specified in the analysis configuration.
        """
        baseline_sel = self.baseline_selector()
        selectors = {k: v for k, v in baseline_sel.items() if k in df.columns}
        baseline = df.filter(**selectors)
        if len(baseline["dataset_gid"].unique()) != 1:
            self.logger.error("Invalid baseline specifier %s", baseline_sel)
            raise ValueError("Invalid configuration")
        return baseline

    def compute_overhead(self,
                         df: pl.DataFrame,
                         metric: str,
                         inverted: bool = False,
                         extra_groupby: list[str] | None = None,
                         overhead_scale=1,
                         how="median") -> pl.DataFrame:
        """
        Generate overhead vs the common baseline for long-form data. The result
        is returned as a new dataframe.

        The new dataframe will have an additional column named _metric_type,
        which assumes the values of '<metric>', '<metric>_delta' and '<metric>_overhead'.
        The <metric> column change its meaning depending on the value of _metric_type.
        Note that in the resulting frame, the baseline data is not filtered-out.

        If how = "mean", data is aggregated using mean and standard deviation.
        The standard deviation is propagated for the delta and percent overhead as
        the _std columns.
        This makes assumptions about statistical independence of the measurements and
        normality of the distribution.

        If how = "median", data is aggregated using median and quartile range. The IQR is
        computed for the delta and percent overhead as _q25 and _q75 columns.
        Note that the relative measures are normalised with respect to the median,
        the quartiles will indicate spread but do not account the uncertainty on the
        baseline measure.

        Note that this does not preserve row or column ordering.
        """
        assert how == "mean" or how == "median"
        if inverted:
            overhead_scale *= -1

        if how == "median":
            return self._do_median_bootstrap(df, metric, extra_groupby, overhead_scale)
        else:
            return self._do_mean_overhead(df, metric, extra_groupby, overhead_scale)


class DatasetAnalysisTask(AnalysisTask):
    """
    Base class for analysis tasks that operate on a single dataset context.
    These generally used to perform per-dataset operations such as loading
    benchmark output data, pre-processing and preliminary aggregation.
    """
    task_namespace = "analysis.dataset"

    def __init__(self, benchmark: Benchmark, analysis_config: AnalysisConfig, task_config: Config | None = None):
        #: The associated benchmark context
        self.benchmark = benchmark

        # Borg initialization occurs here
        super().__init__(benchmark.session, analysis_config, task_config=task_config)

    @classmethod
    def is_session_task(cls):
        return False

    @classmethod
    def is_dataset_task(cls):
        return True

    @classmethod
    def is_benchmark_task(cls):
        warn(f"{cls.__name__}.is_benchmark_task has been renamed is_dataset_task", DeprecationWarning, 2)
        return cls.is_dataset_task()

    @property
    def uuid(self):
        return self.benchmark.uuid

    @property
    def g_uuid(self):
        return self.benchmark.g_uuid

    @property
    def task_id(self):
        """
        Note that this currently assumes that tasks with the same name are not issued
        more than once for each benchmark run UUID. If this is violated, we need to
        change the task ID generation.
        """
        return f"{self.task_namespace}.{self.task_name}-{self.benchmark.uuid}"


class DatasetAnalysisTaskGroup(AnalysisTask):
    """
    Synthetic target that schedules a per-dataset analysis task for each dataset in the session.
    """
    task_name = "sched-group"

    def __init__(self,
                 session: "Session",
                 task_class: Type[DatasetAnalysisTask],
                 analysis_config: AnalysisConfig,
                 task_config: Config | None = None):
        self._task_class = task_class

        # Borg initialization occurs here
        super().__init__(session, analysis_config, task_config)

    @dependency
    def children(self):
        """
        Schedule one instance of task_class for each dataset in the session
        """
        for bench in self.session.all_benchmarks():
            yield self._task_class(bench, self.analysis_config, self.config)

    @property
    def task_id(self):
        """
        Note that this must depend on the target task ID, otherwise we have duplicate IDs in
        case of multiple groups.
        """
        return f"{super().task_id}-for-{self._task_class.task_namespace}-{self._task_class.task_name}"

    def run(self):
        # Nothing to do here
        pass
