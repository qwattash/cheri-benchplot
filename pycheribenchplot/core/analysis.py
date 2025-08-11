import copy
from dataclasses import MISSING, dataclass
from functools import cached_property
from typing import List, Optional, Type, TypeAlias
from warnings import deprecated

import numpy as np
import polars as pl
import polars.selectors as cs
import scipy.stats as scs

from .benchmark import Benchmark
from .config import (AnalysisConfig, Config, InstanceConfig, TaskTargetConfig, config_field)
from .error import ConfigurationError
from .task import SessionTask, TaskRegistry, dependency

# Forward definition, XXX remove circular import if possible...
Session: TypeAlias = "Session"


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

    def __init__(self, session: Session, analysis_config: AnalysisConfig, task_config: Config | None = None):
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
            rel_err_sum.alias("_tmp_sum_of_err_squared"),
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

        # Mark baseline rows with _baseline=True to aid filtering
        lf_stats = lf_stats.with_columns(pl.when(**baseline_sel).then(True).otherwise(False).alias("_is_baseline"))

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
        self.logger.debug("Bootstrap statistics with param=%s extra=%s baseline=%s dof=%s", self.param_columns,
                          extra_groupby, baseline_sel, dof)
        if dof:
            jdf = df.join(baseline, on=dof).select(~cs.ends_with("_right"))
            # Note that we expect the shape of the dataframe not to change
            if jdf.shape[0] != df.shape[0]:
                self.logger.fatal(
                    "Unexpected baseline join result. Your data may not be aligned on the "
                    "free baseline axes, consider checking that every parameterisation group "
                    "has all the value combinations for %s", dof)
                raise RuntimeError("Data constraint violation")
        else:
            # No unconstrained param_columns on the baseline, this should be a single row
            # If this is the case, it means we also have a single iteration.
            # There is no point of bootstrapping in this case, but the functions
            # below will take care of this.
            assert baseline.shape[0] == 1, "Unexpected baseline shape"
            jdf = df.join(baseline, how="cross")
        assert metric_b in jdf.columns, "Baseline data column missing"
        df = jdf

        ci_low_col = f"{metric}_low"
        ci_high_col = f"{metric}_high"

        def _bootstrap(chunk: pl.DataFrame, selectors, statistic_fn):
            """
            Compute a bootstrap confidence interval for the given statistic.
            The selectors argument extracts the columns to use as arguments
            for the statistic.
            The first selector MUST be the `metric` column.
            """
            assert selectors[0] == metric
            args = tuple(chunk[s] for s in selectors)

            if chunk.shape[0] > 1:
                boot = scs.bootstrap(args, statistic_fn, vectorized=True, method="basic")
                ci_low, ci_high = boot.confidence_interval.low, boot.confidence_interval.high
            else:
                # Do not generate error  bars, single iteration
                ci_low = ci_high = None
            s_value = statistic_fn(*args, axis=0)
            stats_chunk = chunk.select(
                cs.by_name(param_columns).first(),
                pl.lit(s_value).alias(metric),
                pl.lit(ci_low).alias(ci_low_col),
                pl.lit(ci_high).alias(ci_high_col))
            return stats_chunk

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

        out_df = pl.concat([stat, delta_stat, ovh_stat], how="vertical", rechunk=True)
        # Mark rows that belong to the baseline group for easy filtering
        out_df = out_df.with_columns(pl.when(**baseline_sel).then(True).otherwise(False).alias("_is_baseline"))

        return out_df

    @property
    def param_columns(self) -> list[str]:
        """
        Return the set of parameter names that are expected to be found in
        a dataframe containing data for this session.
        """
        return list(self._param_columns)

    @property
    def param_columns_with_iter(self) -> list[str]:
        """
        Return the set of parameter names that are configured by the parameterisation matrix,
        with the addition of the iteration index column.
        """
        return [*self.param_columns, "iteration"]

    @property
    def key_columns_with_iter(self) -> list[str]:
        """
        Row ID columns including the iteration.
        This includes the parameterisation columns and the dataset ID colum,
        plus the iteration index.
        """
        return ["dataset_id", *self.param_columns, "iteration"]

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

        if type(baseline_sel) is dict:
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

    def histogram(self,
                  df: pl.DataFrame,
                  value: str,
                  bins: list[float] | None = None,
                  bin_count: int = 100,
                  prefix: str = "hist_",
                  extra_groupby: list[str] | None = None) -> pl.DataFrame:
        """
        Compute an histogram along the parameterization axes.
        This produces two columns:
         - <prefix>bin containing bin left edges
         - <prefix>count containing histogram data

        :param df: Input dataframe.
        :param value: Name of the column to build the histogram from.
        :param bins: Custom bin breakpoints, by default determined by data min and max.
        :param bin_count: Number of bins to use when custom bins are not provided.
        :param prefix: Column prefix for output columns.
        :param extra_groupby: Any additional group levels other than the parameterization axes.
        :returns: New dataframe containing the histogram data in long form.
        """
        param_columns = self.param_columns
        if extra_groupby:
            param_columns += extra_groupby

        if not bins:
            if bin_count <= 0:
                raise ValueError(f"Invalid bin_count {bin_count} must be > 0")
            lo, hi = df[value].min(), df[value].max()
            bins = np.arange(lo, hi, (hi - lo) / bin_count)

        hist_column = f"{prefix}_histogram"
        hdf = df.group_by(param_columns).agg(
            pl.col(value).hist(bins=bins, include_breakpoint=True).alias(hist_column),
            # Note: preserve the columns assuming the param_columns form a table key
            cs.by_name(df.columns).exclude(param_columns).first())
        hdf = hdf.explode(hist_column).with_columns(
            pl.col(hist_column).struct[0].alias(f"{prefix}bin"),
            pl.col(hist_column).struct[1].alias(f"{prefix}count"))
        return hdf.select(df.columns + [f"{prefix}bin", f"{prefix}count"])


@deprecated("Use SliceAnalisysTask instead")
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
    @deprecated("is_benchmark_task has been renamed is_dataset_task")
    def is_benchmark_task(cls):
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


@deprecated("Use GenericAnalisysTask instead")
class DatasetAnalysisTaskGroup(AnalysisTask):
    """
    Synthetic target that schedules a per-dataset analysis task for each dataset in the session.
    """
    task_name = "sched-group"

    def __init__(self,
                 session: Session,
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


@dataclass
class ParamSliceInfo:
    """
    Internal helper that identifies a parameterization slice
    """
    #: Map { param_name: param_value } that identifies this slice
    fixed_params: dict[str, any]
    #: Names of free-floating parameters in this slice
    free_axes: list[str]
    #: Rank the slice. This is the number of
    #: free_axes that are actually independent of each other.
    rank: int

    @property
    def fixed_axes(self) -> list[str]:
        return list(self.fixed_params.keys())


class SliceAnalysisTask(AnalysisTask):
    """
    Base class for slice analysis tasks.

    This task is scheduled by a GenericAnalysisTask to generate results based
    on an arbitrary separation of `fixed_axes` and `free_axes` in the
    parameterisation.

    Note that a publicly visible SliceAnalysisTask can only be scheduled
    within a GenericAnalysisTask. When free-floating, the session automatically
    generates a GenericAnalysisTask with a default set of `free_axes`.

    XXX this is not a session task, the current distinction of
    Session vs Dataset does not make sense here.
    I need to inherit AnalysisTask because I want to reuse the overhead computing.
    """
    #: Maximum number of `free_axes` supported
    max_degrees_of_freedom = 4

    def __init__(self,
                 session: Session,
                 slice_info: ParamSliceInfo,
                 analysis_config: AnalysisConfig,
                 task_config: Config | None = None):
        self._slice_info = slice_info
        # Borg state initialization
        super().__init__(session, analysis_config, task_config)

    @property
    def task_id(self):
        """
        Note that this depends on the inner slice task, so we can define multiple
        generic analysis passes within the same analysis configuration.
        """
        param_pairs = map("=".join, self._slice_info.fixed_params.items())
        slice_id = "-".join(param_pairs) or "nofixed"
        return f"{super().task_id}-slice-{slice_id}"

    @property
    def slice_benchmarks(self) -> list[Benchmark]:
        """
        Produce a slice of the session parameterisation matrix with the
        descriptors for the current slice.
        """
        sliced = self.session.parameterization_matrix.filter(**self.slice_info.fixed_params)
        return sliced["descriptor"]

    @property
    def slice_info(self) -> ParamSliceInfo:
        """
        Get the slice parameterisation descriptor.
        """
        return self._slice_info


@dataclass
class GenericAnalysisConfig(Config):
    """
    Base configuration for generic analysis tasks.

    This specifies two groups of parameterisation axes:
     1. The `fixed_axes` are used to slice the input data and broadcast the
        groups to the configured SliceAnalysisTask.
     2. The `free_axes` are the degrees of freedom available to the
        SliceAnalysisTask.
    """
    #: The handler that we broadcast each data slice to, must be a SliceAnalysisConfig.
    #: XXX I think I want to allow multiple task specs here
    #: We should have a special configuration type for SliceTargetConfig
    #: this would allow us to filter for slice targets at configuration time
    broadcast: List[TaskTargetConfig] = config_field(MISSING,
                                                     desc="Analysis task that is run for each fixed_axes grouping")

    #: Name of axes to keep fixed and broadcast. Note that the names are
    #: completely user-defined. We have a reserved set of axes that must
    #: always be present: ["target", "scenario"], but need not be fixed.
    fixed_axes: List[str] = config_field(list, desc="Parameterisation axes to keep fixed")

    #: Override the baseline selector for this specific analysis
    baseline: Optional[dict] = config_field(
        None, desc="Override baseline selector. This must uniquely identify a single benchmark run.")


class GenericAnalysisTask(AnalysisTask):
    """
    A generic analysis task allows to configure the grouping and slicing of
    dependent and independent variables.

    Let's assume that we have a set of parameterization axes A, B, C, D; where
    A = {a_1, a_2 ...} and so forth.
    This task allows to arbitrarily split the set into two groups:
     1. The `fixed_axes` are the set of axes that we fix across the sub-analyses.
     2. The `free_axes` are the set of axes that are allowed to vary within
        each sub-analysis.

    So, for instance, if we set fixed_axes = [A, B] and free_axes = [C, D],
    this task will schedule N dependent SliceAnalysisTask, where N = card(A x B).
    Each SliceAnalysisTask is assigned a tuple (a_i, b_i) that identifies the data slice,
    this tuple is fixed within the slice.
    The SliceAnalysisTask is in charge of analysing a subset of the data; this task will
    produce a result and the user can configure how the free_axes are handled.

    The top-level GenericAnalysisTask collects the input data into a single dataframe
    that is then sliced dynamically for the SliceAnalysisTasks.

    This is the base class that can be extended by specific tasks or can be used directly
    in the analysis configuration to configure the dynamic analysis.
    When used directly, the analysis configuration permits the nested configuration of
    a specific SliceAnalysisTask.
    """
    task_namespace = "analysis"
    task_name = "dynamic"
    task_config_class = GenericAnalysisConfig
    public = True

    def __init__(self, session: Session, analysis_config: AnalysisConfig, task_config: Config | None = None):
        super().__init__(session, analysis_config, task_config)

        if not set(self.param_columns).issuperset(self.config.fixed_axes):
            self.logger.error("Invalid configuration: fixed_axes=%s must be a subset of parameterisation %s",
                              self.config.fixed_axes, self.param_columns)
            raise ConfigurationError(f"Invalid {self.task_config_class} configuration")

        self._free_axes = set(self.param_columns).difference(self.config.fixed_axes)
        # Determine the degrees of freedom for the free axes
        # XXX I need to groupby here
        self._rank = self.session.parameterization_matrix.select(
            pl.sum_horizontal(cs.by_name(self._free_axes).n_unique() > 1)).item()
        self.logger.debug("Using axes fixed=%s free=%s (%d DOF)", self.config.fixed_axes, self._free_axes, self._rank)

    @property
    def task_id(self):
        """
        Note that this depends on the inner slice task, so we can define multiple
        generic analysis passes within the same analysis configuration.
        """
        task_id = f"{super().task_id}-slice"
        for broadcast_class, _ in self.resolved_slice_tasks:
            task_id += "-" + broadcast_class.task_namespace + broadcast_class.task_name

        return task_id

    @cached_property
    def resolved_slice_tasks(self) -> list[tuple[Type[SliceAnalysisTask], Config]]:
        # Note: the config layer guarantees that the handler is valid at this point
        # XXX we need to support multiple broadcast handlers
        task_types = []
        for slice_config in self.config.broadcast:
            for slice_task_type in TaskRegistry.resolve_task(slice_config.handler):
                task_types.append((slice_task_type, slice_config.task_options))
        return task_types

    @dependency
    def children(self):
        """
        Schedule one instance of task_class for each dataset in the session
        """
        if self.config.fixed_axes:
            groups = self.session.parameterization_matrix.group_by(self.config.fixed_axes)
        else:
            groups = [(tuple(), self.session.parameterization_matrix)]
        for slice_task, slice_config in self.resolved_slice_tasks:
            for name, group_slice in groups:
                group_options = copy.deepcopy(slice_config)
                fixed_slice_params = dict(zip(self.config.fixed_axes, name))
                slice_info = ParamSliceInfo(fixed_params=fixed_slice_params,
                                            free_axes=list(self._free_axes),
                                            rank=self._rank)
                # XXX check that the rank is not too large
                yield slice_task(self.session, slice_info, self.analysis_config, group_options)
