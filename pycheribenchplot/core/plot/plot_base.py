import typing
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pandas as pd

from ..analysis import BenchmarkAnalysis
from ..dataset import DatasetArtefact
from ..util import new_logger
from .backend import FigureManager, Mosaic
from .data_view import CellData, LegendInfo
from .excel import ExcelFigureManager
from .matplotlib import MplFigureManager


class Symbols(Enum):
    DELTA = "\u0394"

    def __add__(self, other):
        if isinstance(other, str):
            return str(self) + other
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, str):
            return other + str(self)
        else:
            return NotImplemented

    def __str__(self):
        return self.value


class BenchmarkPlotBase(BenchmarkAnalysis):
    """
    Base class for handling all plots.
    This class is associated to a plot surface that is responsible for organising
    the layout of subplots.
    Subplots are then added to the layout if the benchmark supports the required
    datasets for the subplot.
    """
    # List of subplot classes that we attempt to draw
    subplots = []

    def __init__(self, benchmark: "BenchmarkBase"):
        super().__init__(benchmark)
        self.logger = new_logger(self.get_plot_name(), benchmark.logger)
        self.mosaic = self._make_subplots_mosaic()
        self.fig_manager = self._make_figure_manager()

    def get_plot_name(self):
        """
        Main title for the plot.
        """
        return self.name

    def get_plot_file(self):
        """
        Generate output file for the plot.
        Note that the filename should not have an extension as the surface will
        append it later.
        """
        return self.benchmark.get_plot_path() / self.name

    def _make_figure_manager(self):
        """
        Build the figure manager backend for this plot.
        """
        raise NotImplementedError("Must override")

    def _make_subplots_mosaic(self) -> Mosaic:
        """
        Mosaic layout for matplotlib-like setup.
        By default generate a single columns with subplots.
        """
        dset_avail = set(self.benchmark.datasets.keys())
        layout = Mosaic()
        for idx, plot_klass in enumerate(self.subplots):
            dset_req = set(plot_klass.get_required_datasets())
            if dset_req.issubset(dset_avail):
                name = f"subplot-{idx}"
                subplot = plot_klass(self)
                layout.subplots[name] = subplot
                nrows, ncols = subplot.get_mosaic_extent()
                layout.allocate(name, nrows, ncols)
        return layout

    def process_datasets(self):
        """Populate the plot axes and draw everything."""
        self.logger.info("Setup plot %s on %s", self.get_plot_name(), self.fig_manager)
        self.fig_manager.allocate_cells(self.mosaic)
        for subplot in self.mosaic:
            subplot.generate(self.fig_manager, subplot.cell)
        self.logger.debug("Drawing plot '%s'", self.get_plot_name())
        self.fig_manager.draw(self.mosaic, self.get_plot_name(), self.get_plot_file())


class BenchmarkTable(BenchmarkPlotBase):
    def _make_figure_manager(self):
        return ExcelFigureManager(self.config)


class BenchmarkPlot(BenchmarkPlotBase):
    def _make_figure_manager(self):
        return MplFigureManager(self.config)


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
        # Cell assigned for rendering
        self.cell = None

    def get_mosaic_extent(self) -> typing.Tuple[int, int]:
        """
        Return a tuple (nrows, ncols) that defines how many mosaic rows and columns this
        subplot uses. This is only relevant for the default mosaic layout setup, if more
        exotic layouts are required, override _make_subplots_mosaic() from the PlotBase.
        """
        return (1, 1)

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
        bench_group = self.benchmark.get_benchmark_group()
        legend = {uuid: str(bench.instance_config.name) for uuid, bench in bench_group.items()}
        legend[self.benchmark.uuid] += "(*)"
        index = pd.Index(legend.keys(), name="dataset_id")
        legend_info = LegendInfo.from_index(index, legend.values())
        return legend_info

    @abstractmethod
    def generate(self, fm: FigureManager, cell: CellData):
        """Generate data views to plot for a given cell"""
        ...
