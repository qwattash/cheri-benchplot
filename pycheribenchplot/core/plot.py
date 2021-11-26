import typing
import itertools as it
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .util import new_logger
from .dataset import DatasetArtefact
from .analysis import BenchmarkAnalysis


class PlotError(Exception):
    pass


class ColorMap:
    @classmethod
    def from_keys(cls, keys: list, mapname: str = "Pastel1"):
        nkeys = len(keys)
        cmap = plt.get_cmap(mapname)
        return ColorMap(keys, cmap(range(nkeys)), keys)

    @classmethod
    def from_index(cls, df: pd.DataFrame, mapname: str = "Pastel1"):
        values = df.index.unique()
        nvalues = len(values)
        cmap = plt.get_cmap(mapname)
        return ColorMap(values, cmap(range(nvalues)), map(lambda val: str(val), values))

    @classmethod
    def from_column(cls, col: pd.Series, mapname: str = "Pastel1"):
        values = col.unique()
        nvalues = len(values)
        cmap = plt.get_cmap(mapname)
        return ColorMap(values, cmap(range(nvalues)), values)

    @classmethod
    def base8(cls):
        """Hardcoded 8-color map with color names as keys (solarized-light theme)"""
        colors = {
            "black": "#002b36",
            "red": "#dc322f",
            "green": "#859900",
            "orange": "#cb4b16",
            "yellow": "#b58900",
            "blue": "#268bd2",
            "magenta": "#d33682",
            "violet": "#6c71c4",
            "cyan": "#2aa198",
            "white": "#839496"
        }
        return ColorMap(colors.keys(), colors.values(), colors.keys())

    def __init__(self, keys: typing.Iterable[typing.Hashable], colors, labels: typing.Iterable[str]):
        self.cmap = OrderedDict()
        for k, c, l in zip(keys, colors, labels):
            self.cmap[k] = (c, l)

    @property
    def colors(self):
        return map(lambda pair: pair[0], self.cmap.values())

    @property
    def labels(self):
        return map(lambda pair: pair[1], self.cmap.values())

    def get_color(self, key: typing.Hashable):
        if key is None:
            return None
        return self.cmap[key][0]

    def get_label(self, key: typing.Hashable):
        if key is None:
            return None
        return self.cmap[key][1]

    def color_items(self):
        return {k: pair[0] for k, pair in self.cmap.items()}

    def __iter__(self):
        return zip(self.colors, self.labels)


class DataView(ABC):
    """
    Base class for single plot types that are drawn onto a cell.
    A data view encodes the rendering logic for a specific type of plot (e.g. a line plot) from a generic
    dataframe using the given columns as axes (if relevant).
    Individual plots can override concrete DataViews to customize the plot appearence.
    Arguments:
    df: View dataframe
    fmt: Column data formatters map. Accept a dict of column-name => formatter.
    x: name of the column to pull X-axis values from (default 'x')
    yleft: name of the column(s) to pull left Y-axis values from, if any
    yright: name of the column(s) to pull right Y-axis values from, if any
    colormap: colormap to use for individual samples in the dataset. This can be used to map different samples
    to different colors. If none is given, no coloring will be performed
    color_col: column to use as key for the color in the colormap. If a colormap is given but no color column, the
    index will be used.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 fmt: dict = {},
                 x: str = "x",
                 yleft: typing.Union[str, list[str]] = [],
                 yright: typing.Union[str, list[str]] = [],
                 colormap: ColorMap = None,
                 color_col: str = None):
        self.df = df
        self.fmt = fmt
        self.x = x
        self._yleft = yleft
        self._yright = yright
        self.colormap = colormap
        self.color_col = color_col

    @property
    def yleft(self) -> list[str]:
        """Return a normalized list of yleft column names"""
        if isinstance(self._yleft, str):
            return [self._yleft]
        else:
            # Any other iterable is fine
            return self._yleft

    @property
    def yright(self) -> list[str]:
        """Return a normalized list of yrigth column names"""
        if isinstance(self._yright, str):
            return [self._yright]
        else:
            # Any other iterable is fine
            return self._yright

    @abstractmethod
    def render(self, cell: "CellData", surface: "Surface"):
        """
        Render this data view. This function allows customization over specific plot methods.
        The main surface used for rendering is given, but the implementation is specific to the
        surface type.
        """
        ...


class CellData:
    """
    Base class to represent a cell on the surface of the plot (e.g. a matplotlib subplot axes).
    This is used to bundle multiple views onto the cell and render them according to the surface type.
    """
    cell_id_gen = it.count()

    def __init__(self, title="", yleft_text="", yright_text="", x_text="", legend_map={}, legend_col="__dataset_id"):
        self.title = title  # Title for the cell of the plot
        self.yleft_text = yleft_text  # Annotation on the left Y axis
        self.yright_text = yright_text  # Annotation on the right Y axis
        self.x_text = x_text  # Annotation on the X axis
        self.legend_map = legend_map  # map index label values to human-readable names
        self.legend_col = legend_col  # Index label for the legend key of each set of data
        self.views = []
        self.surface = None
        self.cell_id = next(CellData.cell_id_gen)

    def set_surface(self, surface):
        self.surface = surface

    def add_view(self, view: DataView):
        self.views.append(view)


class GridLayout:
    """
    Base class abstracting the layout of a plot surface.
    This manages the indexing of the plot cells and holds the
    data that is to be rendered at a given position.
    Some surfaces may not support some layouts and instead flatten the layout.
    """
    def __init__(self, nrows, ncols):
        self._layout = np.full([nrows, ncols], None)

    @property
    def shape(self):
        return self._layout.shape

    def __iter__(self):
        # r, c = np.meshgrid(self._layout.shape)
        # rc = np.dstack([r.ravel(), c.ravel()])
        for row in self._layout:
            yield row

    def next_cell(self) -> typing.Optional[np.ndarray]:
        """Find next empty position"""
        for i, stride in enumerate(self._layout):
            for j, cell in enumerate(stride):
                if cell is None:
                    return np.array((i, j))
        return None

    def set_cell(self, row: int, col: int, cell: CellData):
        self._layout[(row, col)] = cell


class ColumnLayout(GridLayout):
    def __init__(self, nrows):
        super().__init__(nrows, 1)


class RowLayout(GridLayout):
    def __init__(self, ncols):
        super().__init__(1, ncols)


class Surface(ABC):
    """
    The surface abstraction on which to plot.
    We model a grid layout and we associate dataframes to plot to each grid cell.
    Each dataframe must have a specific layout informing what to plot.
    Each view for the cell can hava a different plot type associated, so we can combine
    scatter and line plots (for example).
    """
    def __init__(self):
        self.logger = new_logger(str(self))
        self._layout = None

    def set_layout(self, layout: GridLayout):
        """
        Set the layout to use for the surface.
        Note: this must be called before any cell is added to the surface.
        """
        self._layout = layout

    def next_cell(self, cell: CellData):
        """
        Add the given cell to the next position available in the layout.
        """
        i, j = self._layout.next_cell()
        cell.set_surface(self)
        self._layout.set_cell(i, j, cell)

    def set_cell(self, row: int, col: int, cell: CellData):
        """
        Set the given data to plot on a cell
        """
        cell.set_surface(self)
        self._layout.set_cell(row, col, cell)

    @abstractmethod
    def make_cell(self, **kwargs) -> CellData:
        """Cell factory function"""
        ...

    @abstractmethod
    def make_view(self, plot_type: str, **kwargs) -> DataView:
        """
        Factory for data views for a given plot type.
        I the plot type is not supported, we may throw and exception
        """
        ...

    @abstractmethod
    def draw(self, title: str, dest: Path):
        """Draw all the registered views into the surface."""
        ...

    def __str__(self):
        return self.__class__.__name__


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

    def __init__(self, benchmark: "BenchmarkBase", surfaces: list[Surface]):
        super().__init__(benchmark)
        assert len(surfaces), "Empty plot surface list"
        self.surfaces = surfaces
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
        return self.benchmark.manager.session_ouput_path / self.__class__.__name__

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
        self.benchmark = self.plot.benchmark

    def get_dataset(self, dset_id: DatasetArtefact):
        """Helper to access datasets in the benchmark"""
        dset = self.plot.get_dataset(dset_id)
        assert dset is not None, "Subplot scheduled with missing dependency"
        return dset

    @abstractmethod
    def generate(self, surface: Surface, cell: CellData):
        """Generate data views to plot for a given cell"""
        ...
