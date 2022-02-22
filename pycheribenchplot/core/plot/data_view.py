import itertools as it
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    df: pd.DataFrame
    legend_info: LegendInfo = None
    legend_level: str = None
    key: str = field(init=False)

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

    def __post_init__(self):
        self.key = "table"


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

    def __post_init__(self):
        super().__post_init__()
        self.key = "axline"


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

    def __post_init__(self):
        super().__post_init__()
        # Ensure yleft and yright are lists
        if not isinstance(self.yleft, list):
            self.yleft = [self.yleft]
        if not isinstance(self.yright, list):
            self.yright = [self.yright]

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

    def iter_yleft(self):
        if isinstance(self.yleft, list):
            for c in self.yleft:
                yield c
        else:
            yield self.yleft

    def iter_yright(self):
        if isinstance(self.yright, list):
            for c in self.yright:
                yield c
        else:
            yield self.yright


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

    def __post_init__(self):
        super().__post_init__()
        self.key = "bar"


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

    def __post_init__(self):
        super().__post_init__()
        self.key = "hist"


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
