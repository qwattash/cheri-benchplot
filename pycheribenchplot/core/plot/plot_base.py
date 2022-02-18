from abc import ABC, abstractmethod

from ..analysis import BenchmarkAnalysis
from ..dataset import DatasetArtefact
from ..util import new_logger
from .backend import ColumnLayout, Surface
from .data_view import CellData, LegendInfo
from .excel import SpreadsheetSurface
from .matplotlib import MatplotlibSurface


class BenchmarkPlot(BenchmarkAnalysis):
    """
    Base class for handling all plots.
    This class is associated to a plot surface that is responsible for organising
    the layout of subplots.
    Subplots are then added to the layout if the benchmark supports the required
    datasets for the subplot.
    """

    # List of subplot classes that we attempt to draw
    subplots = []

    @classmethod
    def check_required_datasets(cls, dsets: list[DatasetArtefact]):
        """
        Check if any of the subplots we have can be generated
        """
        dset_avail = set(dsets)
        for plot_klass in cls.subplots:
            dset_req = set(plot_klass.get_required_datasets())
            if dset_req.issubset(dset_avail):
                return True
        return False

    def __init__(self, benchmark: "BenchmarkBase"):
        super().__init__(benchmark)
        self.surfaces = [SpreadsheetSurface(), MatplotlibSurface()]
        self.logger = new_logger(self.get_plot_name(), benchmark.logger)
        self.active_subplots = self._make_subplots()

    def get_plot_name(self):
        """
        Main title for the plot.
        """
        return self.__class__.__name__

    def get_plot_file(self):
        """
        Generate output file for the plot.
        Note that the filename should not have an extension as the surface will
        append it later.
        """
        return self.benchmark.manager.plot_ouput_path / self.__class__.__name__

    def _make_subplots(self) -> list["BenchmarkSubPlot"]:
        """
        By default, we create one instance for each subplot class listed in the class
        """
        dset_avail = set(self.benchmark.datasets.keys())
        active_subplots = []
        for plot_klass in self.subplots:
            dset_req = set(plot_klass.get_required_datasets())
            if dset_req.issubset(dset_avail):
                active_subplots.append(plot_klass(self))
        return active_subplots

    def _setup_surface(self, surface: Surface):
        """
        Setup drawing surface layout and other parameters.
        """
        surface.set_layout(ColumnLayout(len(self.active_subplots)))
        surface.set_config(self.config)

    def _process_surface(self, surface: Surface):
        """
        Add data views to the given surface and draw it.
        """
        self.logger.info("Setup plot on %s", surface)
        self._setup_surface(surface)
        for plot in self.active_subplots:
            cell = surface.make_cell(title=plot.get_cell_title())
            plot.generate(surface, cell)
            surface.next_cell(cell)
        self.logger.debug("Drawing plot '%s'", self.get_plot_name())
        surface.draw(self.get_plot_name(), self.get_plot_file())

    def process_datasets(self):
        for surface in self.surfaces:
            self._process_surface(surface)


class BenchmarkSubPlot(ABC):
    """
    A subplot is responsible for generating the data view for a cell of
    the plot surface layout.
    """
    @classmethod
    def get_required_datasets(cls):
        return []

    def __init__(self, plot: BenchmarkPlot):
        self.plot = plot
        self.logger = plot.logger
        self.benchmark = self.plot.benchmark

    def get_dataset(self, dset_id: DatasetArtefact):
        """Helper to access datasets in the benchmark"""
        dset = self.plot.get_dataset(dset_id)
        assert dset is not None, "Subplot scheduled with missing dependency"
        return dset

    def build_legend_by_dataset(self):
        """
        Build a legend map that allocates colors and labels to the datasets merged
        in the current benchmark instance.
        """
        legend = {uuid: str(bench.instance_config.name) for uuid, bench in self.benchmark.merged_benchmarks.items()}
        legend[self.benchmark.uuid] = f"{self.benchmark.instance_config.name}(*)"
        legend_info = LegendInfo(legend.keys(), labels=legend.values())
        return legend_info

    @abstractmethod
    def generate(self, surface: Surface, cell: CellData):
        """Generate data views to plot for a given cell"""
        ...
