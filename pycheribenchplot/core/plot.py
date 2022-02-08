import itertools as it
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import AnalysisConfig, BenchmarkAnalysis
from .config import Config
from .dataset import DatasetArtefact
from .util import new_logger


class PlotError(Exception):
    pass


class PlotUnsupportedError(PlotError):
    pass


class LegendInfo:
    """
    Helper to build legends by associating descriptive labels to dataframe index levels.
    """
    @classmethod
    def multi_axis(cls, left: "LegendInfo", right: "LegendInfo", cmap_name: str = None):
        """
        Build a multi-axis legend for both the left and right axes. The extra "axis" index
        level is added to keys and will be used internally when multiple axes are enabled,
        if it exists.
        """
        if cmap_name is None:
            cmap_name = left.cmap_name
        left.info_df["axis"] = "left"
        right.info_df["axis"] = "right"
        combined = pd.concat((left.info_df, right.info_df)).set_index("axis", append=True).sort_index()
        return LegendInfo(combined.index, labels=combined["labels"], cmap_name=cmap_name)

    def __init__(self,
                 keys: typing.Iterable[typing.Hashable] | pd.Index,
                 labels: typing.Iterable[str] = None,
                 cmap_name: str = "Pastel1",
                 color_range: typing.Tuple[float, float] = (0, 1),
                 colors: list = None):
        """
        Note that the constructor allows the use of keys as a list of values or a more comples
        pandas index. The class factory helpers should be used in most cases.
        If a list of colors is given, override automatic creation.
        """
        self.cmap_name = cmap_name
        if labels is None:
            labels = keys
        if colors is None:
            cmap = plt.get_cmap(cmap_name)
            cmap_range = np.linspace(color_range[0], color_range[1], len(keys))
            colors = cmap(cmap_range)
        self.info_df = pd.DataFrame({"labels": labels, "colors": list(colors)}, index=keys)

    def map_label(self, fn) -> "LegendInfo":
        new_labels = self.info_df["labels"].map(fn)
        new_info = LegendInfo(self.info_df.index, labels=new_labels, colors=self.info_df["colors"])
        return new_info

    @property
    def index(self):
        return self.info_df.index

    def color(self, key: typing.Hashable, axis="left"):
        if "axis" in self.info_df.index.names:
            lookup_df = self.info_df.xs(axis, level="axis")
        else:
            lookup_df = self.info_df
        return lookup_df.loc[key, "colors"]

    def label(self, key: typing.Hashable, axis="left"):
        if "axis" in self.info_df.index.names:
            lookup_df = self.info_df.xs(axis, level="axis")
        else:
            lookup_df = self.info_df
        return lookup_df.loc[key, "labels"]

    def color_items(self):
        return self.info_df["colors"].items()

    def label_items(self):
        return self.info_df["labels"].items()

    def remap_colors(self, mapname: str, color_range: typing.Tuple[float, float] = (0, 1)):
        self.cmap_name = mapname
        cmap = plt.get_cmap(mapname)
        cmap_range = np.linspace(color_range[0], color_range[1], len(self.info_df))
        self.info_df["colors"] = list(cmap(cmap_range))

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
        _, mapped_level = index[level].align(self.info_df["labels"], level=level)
        index[level] = mapped_level
        new_index = pd.MultiIndex.from_frame(index)
        if axis == 0:
            df.index = new_index
        else:
            df.columns = new_index
        return df

    def with_label_index(self):
        labels = self.info_df["labels"]
        new_df = self.info_df.set_index("labels")
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
    legend_info: Override legend_info from the cell, the legends will then be merged.
    legend_level: Same as CellData.legend_level but for the overridden legend_info

    fmt: Column data formatters map. Accept a dict of column-name => formatter.
    x: name of the column to pull X-axis values from (default 'x')
    yleft: name of the column(s) to pull left Y-axis values from, if any
    yright: name of the column(s) to pull right Y-axis values from, if any
    """
    plot: str
    df: pd.DataFrame
    legend_info: LegendInfo = None
    legend_level: str = None

    def __post_init__(self):
        # Make sure that we operate on a fresh dataframe so that renderers can mess
        # it up without risk of weird behaviours
        self.df = self.df.copy()

    def get_col(self, col, df=None):
        """
        Return a dataframe column, which might be a column or index level.
        Note that index levels are normalized to dataframe.
        """
        if df is None:
            df = self.df
        if col in df.index.names:
            return df.index.to_frame()[col]
        else:
            return df[col]


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
    lintresh: linear treshold for symlog scales
    linscale: linear scale factor
    """
    name: str
    base: typing.Optional[int] = None
    lintresh: typing.Optional[int] = None
    linscale: typing.Optional[float] = None


@dataclass
class Style:
    """
    Wrapper for style information
    """
    line_style: str = "solid"
    line_width: int = None


@dataclass
class AALineDataView(DataView):
    """
    Axis-aligned lines (vertical and horizontal) spanning the
    whole axes.

    Arguments:
    horizontal: column names to use for the horizontal lines Y axis values
    vertical: column names to use for the vertical lines X axis value
    """
    horizontal: list[str] = field(default_factory=list)
    vertical: list[str] = field(default_factory=list)
    style: Style = field(default_factory=Style)


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

    @property
    def has_yleft(self):
        return len(self.yleft) != 0

    @property
    def has_yright(self):
        return len(self.yright) != 0

    def get_x(self):
        return self.get_col(self.x)

    def get_yleft(self):
        return self.get_col(self.yleft)

    def get_yright(self):
        return self.get_col(self.yright)


@dataclass
class BarPlotDataView(XYPlotDataView):
    """
    Parameters for bar plots

    Arguments:
    bar_group: column or index level to use to generate bar groups,
    each group is plotted along the given x axis
    stack_group: column or index level to use to group bars for stacking
    bar_width: relative size of bar with respect to the bar maximum size
    can vary in the interval (0, 1)
    bar_group_location: how to align the bar groups with respect to the
    X values. Allowed values are "center", "left".
    """
    bar_group: str = None
    stack_group: str = None
    bar_group_location: str = "center"
    bar_width: float = 0.8


@dataclass
class HistPlotDataView(XYPlotDataView):
    """
    Parameters for histogram plots

    Arguments:
    buckets: histogram buckets
    bucket_group: column or index level to use to generate histogram groups,
    this will be used to plot multiple histogram columns for each bucket
    """
    buckets: list[float] = field(default_factory=list)
    bucket_group: str = None


@dataclass
class AxisConfig:
    """
    Axis parameters wrapper
    """
    label: str
    enable: bool = False
    limits: tuple[float, float] = None
    ticks: list[float] = None
    tick_labels: list[str] = None
    tick_rotation: int = None
    scale: Scale = None

    def __bool__(self):
        return self.enable


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
        Each of the axes (x, yleft and yright) has a separate configuration object

        Properties:
        legend_info: LegendInfo mapping index label values to human-readable names and colors
        Note that whether the legend is keyed by column names or index level currently
        depends on the view that renders the data.
        legend_level: Index label for the legend key of each set of data
        """
        self.title = title
        self.x_config = AxisConfig(x_label, enable=True)
        self.yleft_config = AxisConfig(yleft_label, enable=True)
        self.yright_config = AxisConfig(yright_label)
        self.legend_info = None
        self.legend_level = None

        self.views = []
        self.surface = None
        self.cell_id = next(CellData.cell_id_gen)

    def set_surface(self, surface):
        self.surface = surface

    def add_view(self, view: DataView):
        self.views.append(view)

    def validate(self):
        assert self.legend_info or not self.legend_level, "CellData legend_level set without legend_info"

    def default_legend_info(self, view):
        return LegendInfo.default(view.df.index.unique().values)

    def get_legend_level(self, view):
        if view.legend_level:
            level = view.legend_level
        else:
            level = self.legend_level
        return level

    def get_legend_col(self, view, df=None):
        level = self.get_legend_level(view)
        if level:
            return view.get_col(level, df)
        else:
            if df is None:
                df = view.df
            flat_index = df.index.to_flat_index()
            flat_index.name = ",".join(df.index.names)
            return flat_index

    def get_legend_info(self, view):
        if view.legend_info:
            legend_info = view.legend_info
        elif self.legend_info:
            legend_info = self.legend_info
        else:
            legend_info = self.default_legend_info(view)
        return legend_info

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
    The surface manages the plot rendering.
    Each surface has a configured layout to manage subplot axes.
    Each plot rendering is wrapped by a Cell, which represents a set of subplot axes.
    Each cell can display multiple views that wrap a plot rendering method.
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
        self.config = AnalysisConfig()

    def set_layout(self, layout: GridLayout):
        """
        Set the layout to use for the surface.
        Note: this must be called before any cell is added to the surface.
        """
        self._layout = layout

    def set_config(self, config: AnalysisConfig):
        self.config = config

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

    def _draw_cell(cell, ctx):
        cell.validate()
        cell.draw(ctx)

    def _draw_split(self, title, dest):
        base = dest.parent / "split"
        base.mkdir(exist_ok=True)
        stem = dest.stem

        for ri, row in enumerate(self._layout):
            for ci, cell in enumerate(row):
                cell_dest = base / dest.with_stem(f"{stem}-{ri}-{ci}").name
                ctx = self._make_draw_context(title, cell_dest)
                ctx.row = 0
                ctx.col = 0
                try:
                    cell.validate()
                    cell.draw(ctx)
                finally:
                    self._finalize_draw_context(ctx)

    def _draw_combined(self, title, dest):
        ctx = self._make_draw_context(title, dest)
        try:
            for ri, row in enumerate(self._layout):
                for ci, cell in enumerate(row):
                    ctx.row = ri
                    ctx.col = ci
                    cell.validate()
                    cell.draw(ctx)
        finally:
            self._finalize_draw_context(ctx)

    def draw(self, title: str, dest: Path):
        """Draw all the registered views into the surface."""
        self.logger.debug("Drawing...")

        if self.config.split_subplots:
            self._draw_split(title, dest)
        else:
            self._draw_combined(title, dest)

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
