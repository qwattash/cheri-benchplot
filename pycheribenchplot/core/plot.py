import itertools as it
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import BenchmarkAnalysis
from .dataset import DatasetArtefact
from .util import new_logger


class PlotError(Exception):
    pass


class PlotUnsupportedError(PlotError):
    pass


class ColorMap:
    @classmethod
    def from_keys(cls,
                  keys: list,
                  mapname: str = "Pastel1",
                  labels: list = None,
                  color_range: typing.Tuple[float, float] = (0, 1)):
        nkeys = len(keys)
        cmap = plt.get_cmap(mapname)
        if labels is None:
            labels = keys
        cmap_range = np.linspace(color_range[0], color_range[1], nkeys)
        return ColorMap(keys, cmap(cmap_range), labels)

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

    def __init__(self, keys: typing.Iterable[typing.Hashable], colors: typing.Iterable, labels: typing.Iterable[str]):
        self.cmap = pd.DataFrame({"labels": labels, "colors": list(colors)}, index=keys)

    @property
    def colors(self):
        return self.cmap["colors"]

    @property
    def labels(self):
        return self.cmap["labels"]

    def get_color(self, key: typing.Hashable):
        return self.cmap.loc[key, "colors"]

    def get_label(self, key: typing.Hashable):
        return self.cmap.loc[key, "labels"]

    def color_items(self):
        return self.cmap["colors"].items()

    def label_items(self):
        return self.cmap["labels"].items()

    def __iter__(self):
        return self.cmap.iterrows()

    def map_labels_to_level(self, df: pd.DataFrame, level=0, axis=0) -> pd.DataFrame:
        """
        Given a dataframe with a multi-index on the given axis, map the
        given level to the labels of the legend map, using the level values
        as keys into the legend map
        The axis may be 0 => index or 1 => columns, mirroring pandas.
        Returns a copy of the original dataframe with the new index level
        """
        df = df.copy()
        if axis == 0:
            index = df.index.to_frame()
        elif axis == 1:
            index = df.columns.to_frame()
        else:
            raise ValueError("Axis must be 0 or 1")
        _, mapped_level = index[level].align(self.cmap["labels"], level=level)
        index[level] = mapped_level
        new_index = pd.MultiIndex.from_frame(index)
        if axis == 0:
            df.index = new_index
        else:
            df.columns = new_index
        return df

    def with_label_index(self):
        labels = self.cmap["labels"]
        new_df = self.cmap.set_index("labels")
        new_df["labels"] = labels.values
        return new_df


@dataclass
class DataView:
    """
    Base class for single plot types that are drawn onto a cell.
    A data view encodes the parameters needed for rendering a specific type of plot (e.g. a line plot)
    from a generic dataframe using the given columns.
    Each surface will then handle the DataView to render the plot.

    Arguments:
    df: View dataframe
    plot: The renderer to use for the plot (e.g. table, hist, line...)
    colormap: colormap to use for individual samples in the dataset. This can be used to map different samples
    to different colors. If none is given, no coloring will be performed.
    color_col: column to use as key for the color in the colormap. If a colormap is given but no color column, the
    index will be used.

    fmt: Column data formatters map. Accept a dict of column-name => formatter.
    x: name of the column to pull X-axis values from (default 'x')
    yleft: name of the column(s) to pull left Y-axis values from, if any
    yright: name of the column(s) to pull right Y-axis values from, if any
    """
    plot: str
    df: pd.DataFrame
    colormap: ColorMap = None
    color_col: str = None


@dataclass
class TableDataView(DataView):
    """
    Data view for tabular visualization

    Arguments:
    columns: a list of columns to display, the column identifiers
    depend on the dataframe column index.
    """
    columns: list = field(default_factory=list)


@dataclass
class Scale:
    """
    Wrapper for scale information

    Arguments:
    name: the scale name (e.g. log)
    base: log base for logarithmic scales
    """
    name: str
    base: typing.Optional[int] = None


@dataclass
class XYPlotDataView(DataView):
    """
    Data view for X-Y plot visualization.

    Arguments:
    x: X dataframe column (or index level)
    yleft: left Y axis column(s)
    yright: right Y axis column(s)

    For each axis (x, yleft, yright) the following arguments exist:
    <ax>_scale: axis scale, if None use whatever default the surface provides
    """
    x: str = "x"
    yleft: list[str] = field(default_factory=list)
    yright: list[str] = field(default_factory=list)
    x_scale: Scale = None
    yleft_scale: Scale = None
    yright_scale: Scale = None


@dataclass
class BarPlotDataView(XYPlotDataView):
    pass


@dataclass
class HistPlotDataView(BarPlotDataView):
    """
    Parameters for histogram plots

    Arguments:
    buckets: histogram buckets
    bucket_group: column or index level to use to generate histogram groups,
    this will be used to plot multiple histogram columns for each bucket
    """
    buckets: list[float] = field(default_factory=list)
    bucket_group: str = None


class CellData:
    """
    Base class to represent a cell on the surface of the plot (e.g. a matplotlib subplot axes).
    This is used to bundle multiple views onto the cell and render them according to the surface type.
    """
    cell_id_gen = it.count()

    def __init__(self, title="", yleft_label=None, yright_label=None, x_label=""):
        """
        Arguments:
        title: Title for the cell of the plot
        yleft_label: Annotation on the left Y axis
        yright_label: Annotation on the right Y axis
        x_label: Annotation on the X axis

        Properties:
        legend_map: ColorMap mapping index label values to human-readable names and colors
        Note that whether the legend is keyed by column names or index level currently
        depends on the view that renders the data.
        legend_level: Index label for the legend key of each set of data
        legend_axis: Axis in which the legend level is found

        TODO: honor legend_level and add legend_level_axis to map the values in the level to
        legend labels in the view dataframe. Alternatively, remove the legend_level property.
        """
        self.title = title
        self.yleft_label = yleft_label
        self.yright_label = yright_label
        self.x_label = x_label
        self.views = []
        self.surface = None
        self.cell_id = next(CellData.cell_id_gen)
        self.legend_map = None
        self.legend_level = None
        self.legend_axis = None

    def set_surface(self, surface):
        self.surface = surface

    def add_view(self, view: DataView):
        self.views.append(view)

    def draw(self, ctx: "Surface.DrawContext"):
        pass


class ViewRenderer(ABC):
    """
    Base class that renders plots.
    This must be specialized for each supported plot type by each surface
    """
    @abstractmethod
    def render(self, view: DataView, cell: CellData, surface: "Surface", **kwargs):
        ...


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
    @dataclass
    class DrawContext:
        title: str
        dest: Path
        row: int
        col: int

    def __init__(self):
        self.logger = new_logger(str(self))
        self._layout = None
        self._renderers = {}

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

    def _make_draw_context(self, title, dest, **kwargs):
        return self.DrawContext(title, dest, row=0, col=0, **kwargs)

    def _finalize_draw_context(self, ctx):
        pass

    def get_renderer(self, data_view: DataView) -> ViewRenderer:
        try:
            return self._renderers[data_view.plot]()
        except KeyError:
            raise PlotUnsupportedError(f"Plot type {data_view.plot} not supported by {self}")

    def draw(self, title: str, dest: Path):
        """Draw all the registered views into the surface."""
        ctx = self._make_draw_context(title, dest)
        try:
            self.logger.debug("Drawing...")
            for ri, row in enumerate(self._layout):
                for ci, cell in enumerate(row):
                    ctx.row = ri
                    ctx.col = ci
                    cell.draw(ctx)
        finally:
            self._finalize_draw_context(ctx)

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
