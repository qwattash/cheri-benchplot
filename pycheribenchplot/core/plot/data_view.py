import itertools as it
import typing
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class LegendInfo:
    """
    Helper to build legends by associating descriptive labels to dataframe index levels.
    Legend info can be combined to create multi-index legend info that are selected using
    more than one key for complex plot configurations.
    """
    @classmethod
    def multi_axis(cls, left: "LegendInfo", right: "LegendInfo"):
        """
        Build a multi-axis legend for both the left and right axes. The extra "axis" index
        level is added to keys and will be used internally when multiple axes are enabled,
        if it exists.
        """
        return cls.combine("axis", {"left": left, "right": right})

    @classmethod
    def combine(cls, index_name: str, chunks: dict[str | tuple, "LegendInfo"]) -> "LegendInfo":
        """
        Combine multiple LegendInfo adding a new index level with the given name.
        Chunks is a dictionary mapping the new index key to the corresponding LegendInfo.
        Note that the chunks should share the same index levels.
        Note that if the chunks keys are tuples, we will still generate a single index level
        in the legend_info dataframe. If multiple columns are required, we need to expose the
        "levels" argument of pd.concat()
        """
        unique_levels = np.unique([c.info_df.index.names for c in chunks.values()])
        assert len(unique_levels) == 1, "LegendInfo chunks should share all index levels"
        unique_keys = pd.unique(list(chunks.keys()))

        new_df = pd.concat((c.info_df for c in chunks.values()),
                           keys=chunks.keys(),
                           levels=[unique_keys],
                           names=[index_name])
        return LegendInfo(new_df)

    @classmethod
    def from_index(cls,
                   index: pd.Index | typing.Iterable,
                   labels: typing.Iterable[str],
                   cmap_name: str = "Pastel1",
                   color_range: tuple[float, float] = (0, 1),
                   colors: list = None):
        if not isinstance(index, pd.Index):
            # Normalize index
            index = pd.Index(index, name="primary")
        if colors is None:
            cmap = plt.get_cmap(cmap_name)
            cmap_range = np.linspace(color_range[0], color_range[1], len(index))
            colors = cmap(cmap_range)
        frame = pd.DataFrame({"labels": list(labels), "colors": list(colors)}, index=index)
        return LegendInfo(frame)

    @classmethod
    def gen_colors(cls,
                   values: pd.Series,
                   mapname: str,
                   color_range: tuple[float, float] = (0, 1),
                   groupby=None) -> pd.Series:
        """
        Generates a set of colors with the given colormap and range for a sequence of values
        """
        cmap = plt.get_cmap(mapname)
        if groupby is not None:
            groups = values.groupby(groupby)
            points = np.linspace(color_range[0], color_range[1], len(groups))
            colors = cmap(points)
            return groups.ngroup().map(lambda g: colors[g])
        else:
            points = np.linspace(color_range[0], color_range[1], len(values))
            return pd.Series(list(cmap(points)), index=values.index, dtype=object)

    @classmethod
    def color_hsv_transform(cls):
        """
        Generate a function to alter colors, suitable to use in a dataframe map() or apply()
        """
        pass

    def __init__(self, df: pd.DataFrame):
        """
        Note that the constructor allows the use of keys as a list of values or a more comples
        pandas index. The class factory helpers should be used in most cases.
        If a list of colors is given, override automatic creation.
        """
        assert "labels" in df.columns
        assert "colors" in df.columns
        self.info_df = df

    def remap_colors(self, mapname: str, **kwargs) -> "LegendInfo":
        new_df = self.info_df.copy()
        new_df["colors"] = LegendInfo.gen_colors(self.info_df["labels"], mapname, **kwargs)
        return LegendInfo(new_df)

    def resolve(self, df: pd.DataFrame, levels: list[str]):
        """
        Return a dataframe with the corresponding labels and colors columns joined onto the
        given df over levels.
        Note that since we need to join the levels on the info_df index, only columns with
        a single value on the column multi-index level 0 are supported. We drop any other
        column index level.
        """
        assert len(self.info_df.index.names) == len(levels)
        if len(df.columns.names) > 1:
            tmp_df = df.copy()
            tmp_df.columns = tmp_df.columns.droplevel(tmp_df.columns.names[1:])
        else:
            tmp_df = df
        joined = tmp_df.merge(self.info_df, left_on=levels, right_index=True)
        assert not joined["colors"].isna().any()
        assert not joined["labels"].isna().any()
        return joined[["colors", "labels"]]

    @property
    def labels(self):
        return self.info_df.loc[:, "labels"]

    @property
    def colors(self):
        return self.info_df.loc[:, "colors"]

    def with_label_index(self):
        labels = self.info_df["labels"]
        new_df = self.info_df.set_index("labels")
        new_df["labels"] = labels.values
        return new_df

    def _map_column(self, fn, column, group_by=None) -> "LegendInfo":
        new_df = self.info_df.copy()
        if group_by is None:
            new_df[column] = fn(self.info_df[column])
        else:
            grouped = self.info_df.groupby(group_by)
            new_df[column] = grouped[column].apply(fn)

        return LegendInfo(new_df)

    def map_color(self, fn, group_by=None) -> "LegendInfo":
        if group_by is None:
            group_by = self.info_df.index
        return self._map_column(fn, "colors", group_by=group_by)

    def map_label(self, fn, group_by=None) -> "LegendInfo":
        if group_by is None:
            group_by = self.info_df.index
        return self._map_column(fn, "labels", group_by=group_by)

    def assign_colors(self,
                      base_map: str | mcolors.Colormap,
                      levels: list[str] | str,
                      color_mapper: typing.Callable,
                      color_range: tuple[float, float] = (0, 1)):
        """
        Helper to assign semantically meaningful colors to labels. The level list specifies the
        levels of the legend_info index taken into account when scaling colors.

        base_map: base colormap to use for the outermost level.
        levels: levels to be used to scale colors
        color_range: colormap range to use between (0, 1)
        scale: attribute to scale for each level or callable
        scale_range: interval in which to scale the value given
        """
        df = self.info_df.copy()
        # Assign base colors
        if isinstance(base_map, str):
            cmap = plt.get_cmap(base_map)
        else:
            cmap = base_map
        grouped = df.groupby(levels)
        color_points = np.linspace(color_range[0], color_range[1], len(grouped))
        for c, (group_key, group) in zip(cmap(color_points), grouped):
            color_vec = pd.Series([c] * len(group), dtype=object, index=group.index)
            df.loc[group.index, "colors"] = color_mapper(c, color_vec)
        return LegendInfo(df)

    def assign_colors_luminance(self,
                                base_map: str | mcolors.Colormap,
                                levels: list[str] | str,
                                color_range: tuple[float, float] = (0, 1),
                                lum_range: tuple[float, float] = (-0.3, 0.3)):
        """
        Assign a primary color from the base map to each group, then offset the luminance for each
        element in the group. This works well for monotonically increasing L* colormaps, but it is
        acceptable also for symmetric ones. Does not respond well to greyscale printing.
        """
        def mapper(base_color, color_vec):
            lum_offsets = np.linspace(lum_range[0], lum_range[1], len(color_vec))
            hsv = color_vec.map(mcolors.to_rgb).map(mcolors.rgb_to_hsv)
            for i, (hsv_color, offset) in enumerate(zip(hsv, lum_offsets)):
                hsv_color[2] += offset
                hsv_color = np.clip(hsv_color, 0, 1)
                color_vec[i] = mcolors.to_rgba(mcolors.hsv_to_rgb(hsv_color))
            return color_vec

        return self.assign_colors(base_map, levels, mapper, color_range=color_range)

    def assign_colors_hsv(self,
                          levels: list[str] | str,
                          sub_levels: list[str] | str = None,
                          h: tuple[float, float] = (0, 1),
                          s: tuple[float, float] = (0, 1),
                          v: tuple[float, float] = (0, 1)):
        """
        Allocate the HSV color space to the legend colors.
        The Hue selects the levels group, the saturation is used to select secondary levels and luminance offsets
        the leaf colors.
        """
        if isinstance(levels, str):
            levels = [levels]
        if isinstance(sub_levels, str) and sub_levels is not None:
            sub_levels = [sub_levels]
        df = self.info_df.copy()

        h_group = df.groupby(levels)
        h_values = np.linspace(h[0], h[1], len(h_group))
        h_vector = h_group.ngroup().map(lambda i: h_values[i])

        if sub_levels:
            s_group = df.groupby(sub_levels)
            s_values = np.linspace(s[0], s[1], len(s_group))
            v_group = df.groupby(levels + sub_levels)
        else:
            s_group = h_group
            s_values = pd.Series((s[0] + s[1]) / 2, index=df.index)
            v_group = h_group
        s_vector = s_group.ngroup().map(lambda i: s_values[i])
        v_values = np.linspace(v[0], v[1], len(v_group))
        v_vector = v_group.ngroup().map(lambda i: v_values[i])

        hsv = pd.concat((h_vector, s_vector, v_vector), axis=1)
        hsv.columns = ["h", "s", "v"]
        hsv["hsv"] = hsv.values.tolist()
        df["colors"] = hsv["hsv"].map(mcolors.hsv_to_rgb)
        return LegendInfo(df)

    ## XXX New interface that assumes public access to the info_df
    # this should make things much flexible

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

    def show_samples(self, grayscale=False):
        """
        Debug/preview helper that shows color samples for the legend
        """
        fig, ax = plt.subplots()
        tmp = self.info_df.copy()
        tmp["y"] = [1] * len(tmp)
        if grayscale:
            color = tmp["colors"].map(lambda c: mcolors.rgb_to_hsv(mcolors.to_rgb(c))[-1]).map(lambda c: [c] * 3)
        else:
            color = tmp["colors"]
        tmp.plot.bar(y="y", ax=ax, color=color)
        fig.show()


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
    legend_level: Frame columns/levels to use as legend keys

    fmt: Column data formatters map. Accept a dict of column-name => formatter.
    x: name of the column to pull X-axis values from (default 'x')
    yleft: name of the column(s) to pull left Y-axis values from, if any
    yright: name of the column(s) to pull right Y-axis values from, if any
    """
    df: pd.DataFrame
    legend_info: LegendInfo = None
    legend_level: list[str] = None
    key: str = field(init=False)

    def __post_init__(self):
        # Make sure that we operate on a fresh dataframe so that renderers can mess
        # it up without risk of weird behaviours
        self.df = self.df.copy()
        if self.legend_level is None:
            self.legend_level = self.default_legend_level()
        if self.legend_info is None:
            self.legend_info = self.default_legend_info()
        if not isinstance(self.legend_level, list):
            self.legend_level = [self.legend_level]

    def get_col(self, col: list[str | tuple] | str, df=None):
        """
        Return a dataframe column, which might be a column or index level.
        Note that index levels are normalized to dataframe.
        """
        if df is None:
            df = self.df
        df = df.reset_index().set_index(df.index, drop=False)
        return df[col]

    def default_legend_info(self):
        keys = self.df.index.unique()
        labels = keys.map(str).str.join(",")
        return LegendInfo.from_index(keys, labels)

    def default_legend_level(self):
        return self.df.index.names


@dataclass
class TableDataView(DataView):
    """
    Data view for tabular visualization

    Arguments:
    columns: a list of columns to display, the column identifiers
    depend on the dataframe column index.

    The associated legend_info can have the usual per-row coloring using
    legend_levels. Column indexes can also be used, these should be set in the
    special "column" legend frame index and must contain the column names.
    This allows to implement single-cell coloring if required.
    """
    columns: list = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
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
        return self.get_col(self.x).squeeze()

    def get_yleft(self):
        return self.get_col(self.yleft)

    def get_yright(self):
        return self.get_col(self.yright)


@dataclass
class BarPlotDataView(XYPlotDataView):
    """
    Parameters for bar plots

    Arguments:
    - bar_group: column or index level to use to generate bar groups,
    each group is plotted along the given x axis
    - stack_group: column or index level to use to group bars for stacking
    - bar_width: relative size of bar with respect to the bar maximum size
    can vary in the interval (0, 1)
    - bar_group_location: how to align the bar groups with respect to the
    X values. Allowed values are "center", "left".
    - bar_axes_ordering: when multiple axes are used (yleft and yright),
    determine how the bars are grouped. When the value is "sequential"
    each group contains first the left bars and then the right bars.
    When the value is "interleaved", left and right bars are paired, so that
    the first yleft column is next to the first yright column, and so forth.
    - bar_text: Display the bar value on top of the bar.
    """
    bar_group: str = None
    stack_group: str = None
    bar_width: float = 0.8
    bar_group_location: str = "center"
    bar_axes_ordering: str = "sequential"
    bar_text: bool = False
    bar_text_pad: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        self.key = "bar"


@dataclass
class ArrowPlotDataView(DataView):
    """
    Parameters for the arrow plot.

    Arguments:
    - x: column for X axis values
    - y: column for Y axis labels
    - group_by: Group levels/columns to group data
    - base_group: Baseline group name, this is the set of data from which arrows
    originate
    """
    x: str = None
    y: str = None
    group_by: list[str] = field(default_factory=list)
    base_group: str = None

    def __post_init__(self):
        super().__post_init__()
        self.key = "arrow"


@dataclass
class HistPlotDataView(XYPlotDataView):
    """
    Parameters for histogram plots

    Arguments:
    x: should hold the column/level to use to fill the histogram buckets for
    each group.
    buckets: histogram buckets
    bucket_group: column or index level to use to generate histogram groups,
    this will be used to plot multiple histogram columns for each bucket
    bar_align: where to align histogram bars
    """
    buckets: list[float] = field(default_factory=list)
    bucket_group: list[str] = None
    bar_align: str = "mid"

    def __post_init__(self):
        super().__post_init__()
        self.key = "hist"
        if self.bucket_group is None:
            raise ValueError("bucket_group is required")
        if not isinstance(self.bucket_group, list):
            self.bucket_group = [self.bucket_group]
        # Use the bucket group as legend level
        self.legend_level = self.bucket_group


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
    padding: float = 0.02
    origin_line: bool = True

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
        """
        self.title = title
        self.x_config = AxisConfig(x_label, enable=True)
        self.yleft_config = AxisConfig(yleft_label, enable=True)
        self.yright_config = AxisConfig(yright_label)

        self.views = []
        self.figure_manager = None
        self.cell_id = next(CellData.cell_id_gen)
        self._renderers = {}

    def add_view(self, view: DataView):
        if not isinstance(view.df, pd.DataFrame):
            raise TypeError("DataView has invalid dataframe")
        self.views.append(view)

    def get_renderer(self, data_view: DataView) -> "ViewRenderer":
        try:
            return self._renderers[data_view.key]()
        except KeyError:
            self.logger.debug("Skipping data view %s unsupported", data_view.key)
            return None

    def draw(self):
        raise NotImplementedError("Must override")
