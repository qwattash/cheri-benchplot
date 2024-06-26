from dataclasses import dataclass, field
from pathlib import Path
from typing import Type
from uuid import UUID
from warnings import warn

import numpy as np
import pandas as pd
import polars as pl
from pandera import Field

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
        #: Analysis configuration for this invocation
        self.analysis_config = analysis_config

    def get_instance_config(self, g_uuid: str) -> InstanceConfig:
        """
        Helper to retreive an instance configuration for the given g_uuid.
        """
        return self.session.get_instance_configuration(g_uuid)

    def g_uuid_to_label(self, g_uuid: str) -> str:
        """
        Helper that maps group UUIDs to a human-readable label that describes the instance
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


class MachineGroupAnalysisTask(AnalysisTask):
    """
    DEPRECATED
    Base class for analysis tasks that operate on a group of benchmark contexts that
    have the same g_uuid (machine configuration), i.e. columns in the benchmark matrix.
    This is used for operations such as merging multiple data from benchmark parameterizations
    that have run on the same machine
    This is generally used to perform operations such as aggregating along the machine configuration
    axis and compute deltas between different benchmark configurations on the same machine.
    """
    task_namespace = "analysis.mgroup"

    def __init__(self, session: "Session", analysis_config: AnalysisConfig, g_uuid: UUID, task_config: Config = None):
        """
        :param session: The current session
        :param analysis_config: The analysis configuration for this run.
        :param g_uuid: The machine configuration ID for this group.
        :param task_config: Optional task configuration.
        """
        #: The associated group uuid
        self.g_uuid = g_uuid

        # Borg state initialization occurs here
        super().__init__(session, analysis_config, task_config=task_config)

    @property
    def task_id(self):
        return f"{self.task_namespace}.{self.task_name}-{self.g_uuid}"


class ParamGroupAnalysisTask(AnalysisTask):
    """
    DEPRECATED
    Base class for analysis tasks that operate on a group of benchmark contexts that
    have the same set of parameterization values, i.e. rows in the benchmark matrix.
    This is used for operations such as merging multiple data from the same benchmark
    setup running different machines.
    This is generally used to perform operations such as aggregating along parameter axes
    and compute deltas between runs on different machine configurations.
    """
    task_namespace = "analysis.pgroup"

    def __init__(self,
                 session: "Session",
                 analysis_config: AnalysisConfig,
                 parameters: dict[str, any],
                 task_config: Config = None):
        #: The baseline group uuid
        self.baseline = session.baseline_g_uuid
        #: The set of parameters identifying the target benchmark matrix row
        self.parameters = parameters

        # Borg state initialization occurs here
        super().__init__(session, analysis_config, task_config=task_config)

    @property
    def task_id(self):
        parameter_set = ":".join([f"{key}={value}" for key, value in self.parameters.items()])
        return f"{self.task_namespace}.{self.task_name}-{parameter_set}"


class StatsByParamGroupTask(ParamGroupAnalysisTask):
    """
    DEPRECATED
    Base task that computes statistics for a group of benchmarks with the same parameter keys.
    This will produce a dataframe with different benchmark runs and machine g_uuids in the index, with the same parameterization values.
    This task depends on the load tasks for all benchmarks with a given set of parameter keys.
    We merge the output of various load tasks into a single dataframe which is used to generate the aggregated output.
    The output dataframe should conform to the DataModel for this task.
    """
    #: The load task to use for loading dependencies
    load_task = None
    #: The output data model
    model: Type[DataModel] = None
    #: Extra group keys to use, if more complex changes to the set of keys is needed, override
    #: the group_keys property
    extra_group_keys: list[str] = []

    def __init__(self, session, analysis_config, parameters, **kwargs):
        self._merged_df = None
        self._df = None

        # Borg state initialization occurs here
        super().__init__(session, analysis_config, parameters, **kwargs)

    def _make_load_depend(self, benchmark: Benchmark):
        return self.load_task(benchmark, self.analysis_config)

    def _output_df(self) -> pd.DataFrame:
        """
        Produce the output dataframe and validate it with the given model
        """
        if len(self._df) == 0:
            # Bail it is weird to have an empty dataframe
            self.logger.error("Empty stats dataframe")
            raise RuntimeError("Empty stats dataframe")
        schema = self.model.to_schema(self.session)
        result = schema.validate(self._df)
        return result

    def _transform_merged(self, mdf: pd.DataFrame) -> pd.DataFrame:
        """
        Hook to perform transformations on the merged dataframe before aggregation.
        """
        return mdf

    # Add I/O checks for base models?
    def _transform_aggregate(self, mdf: pd.DataFrame) -> pd.DataFrame:
        """
        Hook to perform aggregation on the merged dataframe.
        This should produce an extra column index level which will become the "aggregate" level.
        """

        # Setup aggregation functions map, note that function names will become names in the
        # 'aggregate' index level.
        def q25(v):
            return np.quantile(v, q=0.25)

        def q75(v):
            return np.quantile(v, q=0.75)

        agg_list = ["mean", "median", "std", q25, q75]
        agg_strategy = {c: agg_list for c in mdf.columns}

        # Group and aggregate
        grouped = mdf.groupby(self.group_keys)
        agg_df = grouped.aggregate(agg_strategy)
        agg_df.columns.set_names("aggregate", level=-1, inplace=True)
        return agg_df

    def _transform_delta_normal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute deltas with uncertainty propagation assuming a normal distribution
        """
        baseline_xs = df.xs(self.baseline, level="dataset_gid")
        # replicate the baseline to align it to each of the other datasets
        _, baseline_align = df.align(baseline_xs, axis=0)
        meanslice = (slice(None), "mean")
        stdslice = (slice(None), "std")
        delta = df.loc[:, meanslice] - baseline_align.loc[:, meanslice]
        std_delta = (df.loc[:, stdslice]**2 + baseline_align.loc[:, stdslice]**2)**0.5
        norm_delta = delta / baseline_align.loc[:, meanslice]
        # Approximated by taylor expansion as standard deviation for f(x,y) = x / y
        # If proof needed see https://www.stat.cmu.edu/~hseltman/files/ratio.pdf
        x = std_delta / delta.values
        y = baseline_align.loc[:, stdslice] / baseline_align.loc[:, meanslice].values
        std_norm_delta = norm_delta.abs().values * ((x**2 + y**2)**0.5)
        out_df = pd.concat([delta, std_delta, norm_delta, std_norm_delta],
                           keys=["delta", "delta", "norm_delta", "norm_delta"],
                           names=["delta"],
                           axis=1)
        return out_df.reorder_levels(["metric", "aggregate", "delta"], axis=1)

    def _transform_delta_skewed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute deltas with uncertainty propagation assuming a skewed distribution.
        Note this is more shaky in theory, we subtract directly the quartile values
        so that we get an estimation of the worst case scenario, where a value in the
        lower quartile gets subtracted a value from the high quartile (and reverse).
        This likely overestimates the uncertainty.
        XXX probably want to replace this with something like this
        https://iopscience.iop.org/article/10.1088/1681-7575/ab2a8d/pdf
        new_err_hi = data_err_hi + bs_err_lo
        new_err_lo = data_err_lo + bs_err_hi
        """
        baseline_xs = df.xs(self.baseline, level="dataset_gid")
        # replicate the baseline to align it to each of the other datasets
        _, baseline_align = df.align(baseline_xs, axis=0)
        medianslice = (slice(None), "median")
        q75slice = (slice(None), "q75")
        q25slice = (slice(None), "q25")

        # Median delta
        delta = df.loc[:, medianslice] - baseline_align.loc[:, medianslice]
        norm_delta = delta / baseline_align.loc[:, medianslice]
        # Estimated quartiles propagation
        delta_q25 = df.loc[:, q25slice] - baseline_align.loc[:, q75slice].values
        delta_q75 = df.loc[:, q75slice] - baseline_align.loc[:, q25slice].values
        norm_delta_q25 = delta_q25 / baseline_align.loc[:, q75slice].values
        norm_delta_q75 = delta_q75 / baseline_align.loc[:, q25slice].values

        out_df = pd.concat([delta, delta_q25, delta_q75, norm_delta, norm_delta_q25, norm_delta_q75],
                           keys=["delta", "delta", "delta", "norm_delta", "norm_delta", "norm_delta"],
                           names=["delta"],
                           axis=1)
        return out_df.reorder_levels(["metric", "aggregate", "delta"], axis=1)

    # Add I/O checks for base models?
    def _transform_delta(self, adf: pd.DataFrame) -> pd.DataFrame:
        """
        Hook to compute statistics delta on the merged dataframe.
        This should produce an extra column index level which will become the "delta" level.
        """
        # We rely on the column index ordering here, so double check it first
        assert adf.columns.names == ["metric", "aggregate"]

        delta_normal = self._transform_delta_normal(adf)
        delta_skewed = self._transform_delta_skewed(adf)
        # Add the delta level also to the absolute values of the stats data
        base_stats_df = pd.concat([adf], keys=["sample"], names=["delta"],
                                  axis=1).reorder_levels(["metric", "aggregate", "delta"], axis=1)
        # Join together all the delta columns with the original data columns
        stats_df = pd.concat([base_stats_df, delta_normal, delta_skewed], axis=1)
        return stats_df

    @property
    def group_keys(self) -> list[str]:
        """
        The list of keys to group by for statistics aggregation.
        By default group by machine configuration ID and parameter keys, so we aggregate along the iterations axis.
        """
        # Note that key ordering matters, we have this assumption in the DataModelBase.
        return ["dataset_gid"] + self.session.parameter_keys + self.extra_group_keys

    def dependencies(self):
        # Generate dependencies for our parameter set, this corresponds to a benchmark matrix row
        contexts = self.session.parameterization_matrix.filter(**self.parameters)

        if len(contexts) != 1:
            self.logger.error(
                "Unexpected set of benchmarks resolved: %s. A single benchmark matrix row should be selected by %s",
                contexts.index, self.parameters)
            raise RuntimeError("Invalid benchmark parameters set")
        for ctx in contexts["descriptor"]:
            yield self._make_load_depend(ctx)

    def run(self):
        """
        This task merges all load dependencies output into a single dataframe.
        Subclasses get a chance to transform the merged data before aggregation via
        :meth:`StatsByParamSetTask._transform_merged`.
        The merged dataframe is then aggregated using the declared set of aggregation
        functions.
        This will produce one additional column index level which is named 'aggregate'.
        The original column index level name is changed to 'metric'.

        The final dataframe will have the following properties:
        The column index has 3 levels: ["metric", "aggregate", "delta"]
        The row index will have a varying set of levels, depending on the aggregation group keys.
        """
        to_merge = []
        for loader in self.resolved_dependencies:
            to_merge.append(loader.output_map["df"].get())
        merged_df = pd.concat(to_merge)
        merged_df = self.load_task.model.to_schema(self.session).validate(merged_df)
        merged_df.columns.name = "metric"
        merged_df = self._transform_merged(merged_df)
        agg_df = self._transform_aggregate(merged_df)
        delta_df = self._transform_delta(agg_df)

        self.output_map["merged_df"].assign(merged_df)
        self.output_map["df"].assign(delta_df)

    def outputs(self):
        yield "merged_df", DataFrameTarget(self, self.load_task.model)
        yield "df", DataFrameTarget(self, self.model)


def StatsField(name, **kwargs):
    """
    DEPRECATED
    Pandera model field that matches a column with the expected column multi-index pattern
    from the :class:`StatsByParamGroupTask`.
    Note: this is a function in pandera, we have to keep it this way.
    """
    col = (name, "mean|median|std|q25|q75", "sample|delta|norm_delta")
    return Field(alias=col, regex=True, **kwargs)


class StatsForAllParamSetsTask(AnalysisTask):
    """
    DEPRECATED
    Generate statistics for each set of parameters in the benchmark matrix.
    Merge the statistics in the output dataframe.
    """
    #: The task class to produce the statistics dataframe
    stats_task: Type[StatsByParamGroupTask] = None
    #: Data model for the output dataframe
    model: Type[DataModel] = None

    def __init__(self, session, analysis_config, **kwargs):
        #: The merged dataframe with all unaggregated data.
        self._merged_df = None
        #: The output dataframe, after the task is completed.
        self._df = None

        # Borg state initialization occurs here
        super().__init__(session, analysis_config, **kwargs)

    def _output_df(self) -> pd.DataFrame:
        """
        Produce the output dataframe
        """
        schema = self.model.to_schema(self.session)
        return schema.validate(self._df)

    def dependencies(self):
        # If the index is not parameterized, just schedule one stats_task
        if not self.session.parameter_keys:
            yield self.stats_task(self.session, self.analysis_config, {})
        else:
            param_sets = self.session.parameterization_matrix.with_columns(
                pl.concat_list(self.session.parameter_keys).alias("param_sets")).select("param_sets")
            for entry in param_sets:
                params = dict(zip(self.session.parameter_keys, entry))
                yield self.stats_task(self.session, self.analysis_config, params)

    def run(self):
        unaggregated_frames = []
        stats_frames = []
        for stats_task in self.resolved_dependencies:
            unaggregated_frames.append(stats_task.output_map["merged_df"].get())
            stats_frames.append(stats_task.output_map["df"].get())
        merged_df = pd.concat(unaggregated_frames)
        self.output_map["merged_df"].assign(merged_df)
        self.output_map["df"].assign(stats_frames)

    def outputs(self):
        yield "merged_df", DataFrameTarget(self, self.stats_task.load_task.model)
        yield "df", DataFrameTarget(self, self.model)
