import typing
from dataclasses import dataclass, field
from uuid import UUID

from .benchmark import Benchmark
from .config import AnalysisConfig, Config, DatasetName
from .task import AnalysisTask


class BenchmarkAnalysisRegistry(type):
    analysis_steps = {}

    def __init__(self, name, bases, kdict):
        super().__init__(name, bases, kdict)
        if self.name is None:
            return
        if self.name in BenchmarkAnalysisRegistry.analysis_steps:
            raise ValueError(f"Duplicate analysis step {self.name}")
        BenchmarkAnalysisRegistry.analysis_steps[self.name] = self

    def __str__(self):
        # From BenchmarkAnalysis base class
        required = [r.value for r in self.require]
        return f"{self.name}: {self.description} requires {required}"


class BenchmarkAnalysis(metaclass=BenchmarkAnalysisRegistry):
    """
    Base class for analysis steps registered on a benchmark handler.
    Each analysis generates one or more artifacts from one or more datasets
    in the benchmark.
    Each analysis handler is responsible for handling the presentation logic that
    translates the dataframes from one or more datasets into dataframes suitable to
    present certain features or relationships in the datasets.
    This supports plotting and other forms of dataset manipulation by specialization.
    Note that the dataframe manipulation logic may be shared between plot analysis and other
    presentation methods.
    """
    # Required datasets to be present for this analysis step to work correctly
    # The callable provides more granular filtering, it is passed the set of datasets loaded
    # and the analysis configuration.
    require: set[DatasetName] | typing.Callable = []
    # Unique name of the analysis step, to be used to enable it in the configuration
    name: str = None
    # Description of the analysis step
    description: str = None
    #: Cross benchmark variant analysis step
    cross_analysis: bool = False
    #: Extra options from the analysis config are parsed with this configuration
    analysis_options_class: Config = None

    def __init__(self, benchmark: "Benchmark", config: Config, **kwargs):
        """
        Create a new benchmark analysis handler.

        :param benchmark: The parent benchmark
        :param config: The analysis options, the type is specified by :attribute:`analysis_options_class`
        """
        self.benchmark = benchmark
        self.logger = benchmark.logger
        self.config = config

    @property
    def analysis_config(self) -> AnalysisConfig:
        return self.benchmark.session.analysis_config

    def get_dataset(self, dset_id: DatasetName):
        """Helper to access datasets in the benchmark"""
        return self.benchmark.get_dataset(dset_id)

    async def process_datasets(self):
        """
        Process the datasets to generate the intermediate data representation
        used to produce the output artifacts.
        """
        pass

    def __str__(self):
        return self.name


class BenchmarkAnalysisTask(AnalysisTask):
    """
    Base class for analysis tasks that operate on a single benchmark context.
    These generally used to perform per-benchmark operations such as loading
    benchmark output data, pre-processing and preliminary aggregation.
    """
    task_namespace = "analysis.benchmark"

    def __init__(self, benchmark: Benchmark, task_config: Config = None):
        super().__init__(benchmark.session, task_config=task_config)
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
