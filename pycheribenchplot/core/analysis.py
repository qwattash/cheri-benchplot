import typing
import uuid
from dataclasses import dataclass, field

from .config import Config
from .dataset import DatasetName


class BenchmarkAnalysisRegistry(type):
    analysis_steps = {}

    def __init__(self, name, bases, kdict):
        super().__init__(name, bases, kdict)
        if self.name is None:
            return
        if self.name in BenchmarkAnalysisRegistry.analysis_steps:
            raise ValueError(f"Duplicate analysis step {self.name}")
        BenchmarkAnalysisRegistry.analysis_steps[self.name] = self


@dataclass
class AnalysisConfig(Config):
    # List of plots/analysis steps to enable
    enable: typing.List[str] = field(default_factory=list)
    # Tags for group enable
    enable_tags: typing.Set[str] = field(default_factory=set)
    # Generate multiple split plots instead of combining
    split_subplots: bool = False
    # Output formats
    plot_output_format: typing.List[str] = field(default_factory=lambda: ["pdf"])
    # Constants to show in various plots, depending on the X and Y axes.
    # The dictionary maps parameters of the benchmark parameterisation to a dict
    # mapping description -> constant value
    parameter_constants: typing.Dict[str, dict] = field(default_factory=dict)
    # Baseline dataset group id, defaults to the baseline instance uuid
    baseline_gid: uuid.UUID = None

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.plot_output_format, list)


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
    # Analysis step tags for group selection
    tags: set[str] = set()
    # Cross benchmark variant analysis step
    cross_analysis: bool = False

    def __init__(self, benchmark: "BenchmarkBase", **kwargs):
        self.benchmark = benchmark
        self.logger = benchmark.logger
        self.config = benchmark.manager.analysis_config

    def get_dataset(self, dset_id: DatasetName):
        """Helper to access datasets in the benchmark"""
        return self.benchmark.get_dataset(dset_id)

    def process_datasets(self):
        """
        Process the datasets to generate the intermediate data representation
        used to produce the output artifacts.
        """
        pass

    def __str__(self):
        return self.name
