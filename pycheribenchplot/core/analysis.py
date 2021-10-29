from .dataset import DatasetID


class BenchmarkAnalysisRegistry(type):
    analysis_steps = set()

    def __init__(self, name, bases, kdict):
        super().__init__(name, bases, kdict)
        BenchmarkAnalysisRegistry.analysis_steps.add(self)


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
    When running the analysis step, we gather the analysis classes we can run based
    on the dataset dependencies they advertise.
    """
    @classmethod
    def check_required_datasets(cls, dsets: list[DatasetID]):
        """Check whether this analysis step can run"""
        return False

    def __init__(self, benchmark: "BenchmarkBase", **kwargs):
        self.benchmark = benchmark
        self.logger = benchmark.logger

    def process_datasets(self):
        """
        Process the datasets to generate the intermediate data representation
        used to produce the output artifacts.
        """
        pass

    def __str__(self):
        return self.__class__.__name__