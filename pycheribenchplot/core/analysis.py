import typing
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import pandas as pd

from .benchmark import Benchmark
from .config import AnalysisConfig, Config
from .model import DataModel
from .task import AnalysisTask, DataFrameTarget, ExecutionTask


class BenchmarkAnalysisTask(AnalysisTask):
    """
    Base class for analysis tasks that operate on a single benchmark context.
    These generally used to perform per-benchmark operations such as loading
    benchmark output data, pre-processing and preliminary aggregation.
    """
    task_namespace = "analysis.benchmark"

    def __init__(self, benchmark: Benchmark, analysis_config: AnalysisConfig, task_config: Config = None):
        super().__init__(benchmark.session, analysis_config, task_config=task_config)
        #: The associated benchmark context
        self.benchmark = benchmark

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


class MachineGroupAnalysisTask(AnalysisTask):
    """
    Base class for analysis tasks that operate on a group of benchmark contexts that
    have the same g_uuid (machine configuration), i.e. columns in the benchmark matrix.
    This is used for operations such as merging multiple data from benchmark parameterizations
    that have run on the same machine
    This is generally used to perform operations such as aggregating along the machine configuration
    axis and compute deltas between different benchmark configurations on the same machine.
    """
    task_namespace = "analysis.mgroup"

    def __init__(self,
                 session: "PipelineSession",
                 analysis_config: AnalysisConfig,
                 g_uuid: UUID,
                 task_config: Config = None):
        """
        :param session: The current session
        :param analysis_config: The analysis configuration for this run.
        :param g_uuid: The machine configuration ID for this group.
        :param task_config: Optional task configuration.
        """
        super().__init__(session, analysis_config, task_config=task_config)
        #: The associated group uuid
        self.g_uuid = g_uuid

    @property
    def task_id(self):
        return f"{self.task_namespace}.{self.task_name}-{self.g_uuid}"


class ParamGroupAnalysisTask(AnalysisTask):
    """
    Base class for analysis tasks that operate on a group of benchmark contexts that
    have the same set of parameterization values, i.e. rows in the benchmark matrix.
    This is used for operations such as merging multiple data from the same benchmark
    setup running different machines.
    This is generally used to perform operations such as aggregating along parameter axes
    and compute deltas between runs on different machine configurations.
    """
    task_namespace = "analysis.pgroup"

    def __init__(self, session: "PipelineSession", analysis_config: AnalysisConfig, task_config: Config = None):
        super().__init__(session, analysis_config, task_config=task_config)
        #: The baseline group uuid
        self.baseline = analysis_config.baseline_g_uuid

    @property
    def task_id(self):
        return f"{self.task_namespace}.{self.task_name}-{self.g_uuid}"


class BenchmarkDataLoadTask(BenchmarkAnalysisTask):
    """
    General-purpose data loading and pre-processing task for benchmarks.

    This task will load some data from a target of a benchmark exec task.
    The load task needs to be pointed to the provider of the target, from which it
    can extract the path information.
    The data is loaded to a dataframe, according to a :class:`DataModel`.
    The input data model must be specified so that the input data is validated and
    the columns of interest are filtered.
    This task generates a DataFrameTarget() that identifies the task result.
    """
    #: The exec task from which to fetch the target
    exec_task: ExecutionTask = None
    #: The name of the target file to load
    target_key: str = None
    #: Input data model
    model: DataModel = None

    def __init__(self, benchmark: Benchmark, analysis_config: AnalysisConfig, **kwargs):
        super().__init__(benchmark, analysis_config, **kwargs)
        self._df = []

    def _parameter_index_columns(self):
        if self.benchmark.config.parameters:
            return list(self.benchmark.config.parameters.keys())
        return []

    def _output_df(self) -> pd.DataFrame:
        """
        Produce the output dataframe by joining all the iteration frames
        """
        if len(self._df) == 0:
            # Bail because we can not concatenate and validate an empty frame
            # We could support empty data but there is no use case for it now.
            self.logger.error("No data has been loaded for %s", self)
            raise ValueError("Loader did not find any data")
        schema = self.model.to_schema(self.session)
        df = pd.concat(self._df)
        return schema.validate(df)

    def _append_df(self, df: pd.DataFrame):
        """
        Add a given dataframe to the output dataframe.
        This is used to combine multiple iterations of the same benchmark that
        come from different files.
        Here we also set the index columns based on the benchmark configuration.
        """
        if len(df) == 0:
            self.logger.warning("Appending empty dataframe")
            return

        if "dataset_id" not in df.columns:
            self.logger.debug("No dataset column, using default")
            df["dataset_id"] = self.benchmark.uuid
        if "dataset_gid" not in df.columns:
            self.logger.debug("No dataset group, using default")
            df["dataset_gid"] = self.benchmark.g_uuid
        for pcol in self._parameter_index_columns():
            if pcol in df.columns:
                continue
            param = self.benchmark.config.parameters[pcol]
            self.logger.debug("No parameter %s column, generate from config %s=%s", pcol, pcol, param)
            df[pcol] = param

        # Now set the index based on the data model definition and proceed to validate
        schema = self.model.to_schema(self.session)
        df.set_index(schema.index.names, inplace=True)
        valid_df = schema.validate(df)
        self._df.append(df)

    def _load_one_csv(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)

    def _load_one_json(self, path: Path) -> pd.DataFrame:
        return pd.io.json.read_json(path)

    def _load_one(self, path: Path, iteration: int):
        """
        Load data from the given path. The format is inferred from the extension.
        """
        self.logger.debug("Loading data[i=%d] from %s", iteration, path)
        if path.suffix == ".csv":
            df = self._load_one_csv(path)
        elif path.suffix == ".json":
            df = self._load_one_json(path)
        else:
            self.logger.error(
                "Can not determine how to load %s, add extesion or override BenchmarkDataLoadTask._load_one()", path)
            raise RuntimeError("Can not ifer file type from extension")
        df["iteration"] = iteration
        self._append_df(df)

    def run(self):
        target_task = self.exec_task(self.benchmark, script=None)
        target = target_task.output_map.get(self.target_key)
        if target is None:
            self.logger.error("%s can not load data from task %s, output key %s missing", self, target_task,
                              self.target_key)
            raise KeyError(f"{self.target_key} is not in task output_map")
        if not target.is_file():
            raise NotImplementedError("BenchmarkDataLoadTask only supports loading from files")
        for i, path in enumerate(target.paths):
            if not path.exists():
                self.logger.error("Can not load %s, does not exist", path)
                raise FileNotFoundError(f"{path} does not exist")
            if not target.has_iteration_path:
                i = -1
            self._load_one(path, i)

    def outputs(self):
        """
        Note that the target data will be valid only after the Task.completed
        flag has been set.
        """
        yield "df", DataFrameTarget(self.model, self._output_df())
