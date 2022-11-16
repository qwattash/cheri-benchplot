import typing
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pandas as pd

from pycheribenchplot.core.config import Config

from ..analysis import BenchmarkAnalysis
from ..dataset import DataSetContainer
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

    def __init__(self, benchmark: "Benchmark", options):
        super().__init__(benchmark, options)
        self.logger = new_logger(self.get_plot_name(), benchmark.logger)
        self.mosaic = self._make_subplots_mosaic()
        self.fig_manager = self._make_figure_manager()

    def get_plot_name(self):
        """
        Main title for the plot.
        """
        if self.description is not None:
            return self.description
        return self.name

    def get_plot_file(self):
        """
        Generate output file for the plot.
        Note that the filename should not have an extension as the surface will
        append it later.
        """
        if self.cross_analysis:
            return self.benchmark.session.get_plot_root_path() / self.name
        else:
            return self.benchmark.get_plot_path() / self.name

    def get_plot_root_path(self):
        return self.benchmark.session.get_plot_root_path()

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
        return self._make_mosaic_from_subplots_list()

    def _make_mosaic_from_subplots_list(self):
        """
        Basic layouting function that generates a single column from all the
        subplots in the declared BenchmarkPlotBase::subplots
        """
        layout = Mosaic()
        for idx, plot_klass in enumerate(self.subplots):
            name = f"subplot-{idx}"
            subplot = plot_klass(self)
            layout.subplots[name] = subplot
            nrows, ncols = subplot.get_mosaic_extent()
            layout.allocate(name, nrows, ncols)
        return layout

    def _make_mosaic_from_dataset_columns(self,
                                          plot_ctor: typing.Callable,
                                          ds: DataSetContainer,
                                          include_overhead=False):
        """
        Layouting strategy that generates a column of plots for each (valid)
        data column in the given dataset frame.

        plot_ctor: subplots constructor, must accept the following (plot, column_name, overhead=<True|False>)
        ds: The dataset to show
        include_overhead: Generate extra column of plots showing the relative overhead
        """
        subplots = {}
        layout = []
        for idx, metric in enumerate(ds.data_columns()):
            name_abs = f"subplot-{idx}-abs"
            name_delta = f"subplot-{idx}-delta"
            if ds.merged_df[metric].isna().all():
                continue
            subplots[name_abs] = plot_ctor(self, metric, overhead=False)
            row = [name_abs]
            if include_overhead:
                subplots[name_delta] = plot_ctor(self, metric, overhead=True)
                row.append(name_delta)
            layout.append(row)
        return Mosaic(layout, subplots)

    def _make_mosaic_from_cross_merged_dataset(self,
                                               plot_ctor: typing.Callable,
                                               ds: DataSetContainer,
                                               include_overhead=False):
        """
        Layouting strategy that generates a column of plots for each valid data column in the cross-benchmark
        merged dataframe. A new column is added for each benchmark parameterisation key. Another extra column
        showing relative overhead is added for each existing one if include_overhead is True.

        plot_ctor: subplots constructor, must accept the following (plot, column_name, param_name, overhead=<True|False>)
        ds: The dataset to show
        include_overhead: Generate extra column of plots showing the relative overhead
        """
        layout = []
        subplots = {}
        for p in ds.parameter_index_columns():
            for j, metric in enumerate(ds.data_columns()):
                if ds.cross_merged_df[(metric, "median", "sample")].isna().all():
                    # Skip missing metrics
                    continue
                name = f"subplot-param-{p}-{metric}"
                name_delta = f"subplot-param-{p}-{metric}-delta"
                subplots[name] = plot_ctor(self, metric, p, overhead=False)
                row = [name]
                if include_overhead:
                    subplots[name_delta] = plot_ctor(self, metric, p, overhead=True)
                    row.append(name_delta)
                if j < len(layout):
                    layout[j].extend(row)
                else:
                    layout.append(row)
        return Mosaic(layout, subplots)

    async def process_datasets(self):
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
        return MplFigureManager(self.analysis_config)


class BenchmarkSubPlot(ABC):
    """
    A subplot is responsible for generating the data view for a cell of
    the plot surface layout.
    """
    description: str = None

    def __init__(self, plot: BenchmarkPlot):
        self.plot = plot
        self.logger = plot.logger
        self.benchmark = self.plot.benchmark
        # Cell assigned for rendering
        self.cell = None

    def get_cell_title(self) -> str:
        """Generate the title for this subplot, uses the description property by default"""
        if self.description is not None:
            return self.description
        else:
            raise ValueError("Missing subplot description")

    def get_mosaic_extent(self) -> typing.Tuple[int, int]:
        """
        Return a tuple (nrows, ncols) that defines how many mosaic rows and columns this
        subplot uses. This is only relevant for the default mosaic layout setup, if more
        exotic layouts are required, override _make_subplots_mosaic() from the PlotBase.
        """
        return (1, 1)

    def get_dataset(self, dset_id: "DatasetArtefact"):
        """Helper to access datasets in the benchmark"""
        dset = self.plot.get_dataset(dset_id)
        assert dset is not None, "Subplot scheduled with missing dependency"
        return dset

    def build_legend_by_dataset(self):
        """
        Build a legend map that allocates colors and labels to the datasets merged
        in the current benchmark instance.
        """
        bench_group = self.benchmark.get_merged_benchmarks()
        legend = {uuid: str(bench.config.instance.name) for uuid, bench in bench_group.items()}
        legend[self.benchmark.uuid] += "(*)"
        index = pd.Index(legend.keys(), name="dataset_id")
        legend_info = LegendInfo.from_index(index, legend.values())
        return legend_info

    def build_legend_by_gid(self):
        """
        Build a legend map that allocates colors and labels by dataset group ID. This operates on
        the datasets merged in the current cross-benchmark accumulator instance.
        """
        bench_groups = self.benchmark.get_benchmark_groups()
        legend = {uuid: str(group[0].config.instance.name) for uuid, group in bench_groups.items()}
        legend[self.benchmark.g_uuid] += "(*)"
        index = pd.Index(legend.keys(), name="dataset_gid")
        legend_info = LegendInfo.from_index(index, legend.values())
        return legend_info

    @abstractmethod
    def generate(self, fm: FigureManager, cell: CellData):
        """Generate data views to plot for a given cell"""
        ...
