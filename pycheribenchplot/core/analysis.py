import typing
import uuid
from dataclasses import dataclass, field

from .config import AnalysisConfig, Config, DatasetName


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
        if config:
            self.config = config
        else:
            self.config = benchmark.session.analysis_config

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
