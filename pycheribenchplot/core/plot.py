import typing
import itertools as it
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from .dataset import *



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


class Surface(ABC):
    """
    The surface abstraction on which to plot.
    We model a grid layout and we associate dataframes to plot to each grid cell.
    Each dataframe must have a specific layout informing what to plot.
    Each view for the cell can hava a different plot type associated, so we can combine
    scatter and line plots (for example).
    """
    def __init__(self):
        # The layout is encoded as a matrix indexed by row,column of the figure subplots
        # each cell in the plot is associated with cell data
        self._layout = np.full([1, 1], CellData())
        self._expand_layout = False
        self._expand_direction = "row"
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def layout_shape(self):
        return self._layout.shape

    def output_file_ext(self):
        """Extension for the output file"""
        return ""

    def find_empty_cell(self):
        """
        Find an emtpy cell in the layout, if none is found we return None, otherwise
        return the cell coordinates
        """
        if self._expand_direction == "row":
            axis = 0
        else:
            axis = 1
        layout = np.rollaxis(self._layout, axis)

        for i, stride in enumerate(layout):
            for j, cell in enumerate(stride):
                if len(cell.views) == 0:
                    return np.roll(np.array([i, j]), shift=axis)
        return None

    def set_layout(self, nrows: int, ncols: int, expand: bool = False, how: str = "row"):
        """
        Create a drawing surface with given number of rows and columns of plots.
        Note that this will reset any views that have been added to the current layout.

        Arguments:
        nrows: the number of rows to allocate
        ncols: the number of columns to allocate
        [expand]: automatically expand the layout
        [how]: direction in which to expand the layout with next_cell(). If how="row",
        fill rows until we reach ncols, then create another row. If how="col",
        fill columns until we reach nrows, then create another column.
        """
        self._expand_layout = expand
        self._expand_direction = how
        self._layout = np.full([nrows, ncols], self.make_cell())

    def next_cell(self, cell: CellData):
        """
        Add the given cell to the next position available in the layout.
        """
        index = self.find_empty_cell()
        if index is not None:
            i, j = index
            self.set_cell(i, j, cell)
        elif self._expand_layout:
            stride_shape = list(self._layout.shape)
            if self._expand_direction == "row":
                axis = 0
            else:
                axis = 1
            stride_shape[(axis + 1) % 2] = 1
            new_stride = np.full(stride_shape, self.make_cell())
            new_stride[0, 0] = cell
            self._layout = np.concatenate((self._layout, new_stride), axis=axis)
            cell.set_surface(self)
        else:
            raise IndexError("No empty cell found")
        assert len(self._layout.shape) == 2

    def set_cell(self, row: int, col: int, cell: CellData):
        """
        Set the given data to plot on a cell
        """
        cell.set_surface(self)
        self._layout[(row, col)] = cell

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


class MatplotlibSurface(Surface):
    def _supported_methods(self):
        return ["scatter", "line", "bar-group"]

    def _make_figure(self):
        rows, cols = self.layout_shape
        self.fig, self.axes = plt.subplots(rows, cols, sharex=True, figsize=(10 * cols, 5 * rows), squeeze=False)

    def _plot_view(self, ax: "plt.Axes", cell: CellData, view: DataView):
        self.logger.error("Unsupported plot method %s, skip data view", view.method)

    def _plot_bar_group(self, ax: "plt.Axes", cell: CellData, view: DataView):
        """
        For each value in the view, plot a set of bars corresponding each to a dataset_id we are
        comparing. The height of the bars is taken from the y_left column of the view.
        """
        pass

    def _plot_at(self, index, cell):
        ax = self.axes[index]
        for view in cell.views:
            self._plot_view(ax, cell, view)

    def draw(self, title, dest):
        self.logger.debug("Drawing...")
        self._make_figure()
        for index, cell in np.ndenumerate(self._layout):
            self._plot_at(index, cell)
        self.fig.savefig(dest)


class Plot(ABC):
    """Base class for drawing plots."""
    def __init__(self, benchmark: "BenchmarkBase", surface: Surface):
        """
        Note: the benchmark instance passed here is the baseline instance on which we performed the
        dataset aggregation.
        """
        self.benchmark = benchmark
        self.logger = self.benchmark.logger
        self.surface = surface

    @abstractmethod
    def _get_plot_title(self):
        """Return a human-readable name for the plot"""
        ...

    @abstractmethod
    def _get_plot_file(self):
        """Return the output path of the plot"""
        ...

    def prepare(self):
        """Prepare the elements of the plot to draw"""
        self.logger.debug("Setup plot %s", self._get_plot_title())

    def draw(self):
        """Actually draw the plot."""
        self.logger.debug("Drawing plot %s", self._get_plot_title())
        self.surface.draw(self._get_plot_title(), self._get_plot_file())


#### Old stuff


def align_twin_axes(ax, ax_twin, min_twin, max_twin):
    """
    Alipgn 0 point on twin Y axes
    We first shift the twin axis to align the zero point in figure
    coordinates, then we correct limits on both axes to make sure we
    are fitting the shifted parts of the plot.
    """

    lim = ax.get_ylim()
    twin_lim = ax_twin.get_ylim()
    tx = ax.transData
    ttwin = ax_twin.transData
    tx_inv = ax.transData.inverted()
    ttwin_inv = ax_twin.transData.inverted()

    # Shift twin axis to align zero
    _, dy = ttwin.transform((0, 0)) - tx.transform((0, 0))
    _, shift = ttwin_inv.transform((0, 0)) - ttwin_inv.transform((0, dy))
    ax_twin.set_ylim(*(twin_lim - shift))

    # Correct both axes limits to make sure everything is visible
    # after the shift
    lim = ax.get_ylim()
    twin_lim = ax_twin.get_ylim()
    margin = 0.1  # percentage
    if min_twin < twin_lim[0]:
        # adjust min
        # get size of adjustment in figure space
        min_twin = min_twin - np.absolute(margin * min_twin)
        _, dy = (ttwin.transform((0, 0)) - ttwin.transform((0, twin_lim[0] - min_twin)))
        # convert figure-space shift to axis shift
        _, ax_shift = tx_inv.transform((0, 0)) - tx_inv.transform((0, dy))
        _, twin_shift = ttwin_inv.transform((0, 0)) - ttwin_inv.transform((0, dy))
        ax.set_ylim(lim[0] - ax_shift, lim[1])
        ax_twin.set_ylim(twin_lim[0] - twin_shift, twin_lim[1])
    if max_twin > twin_lim[1]:
        # adjust max
        # get size of adjustment in figure space
        max_twin = max_twin + np.absolute(margin * max_twin)
        _, dy = ttwin.transform((0, 0)) - ttwin.transform((0, twin_lim[1] - max_twin))
        # convert figure-space shift to axis shift
        _, ax_shift = tx_inv.transform((0, 0)) - tx_inv.transform((0, dy))
        _, twin_shift = ttwin_inv.transform((0, 0)) - ttwin_inv.transform((0, dy))
        ax.set_ylim(lim[0], lim[1] - ax_shift)
        ax_twin.set_ylim(twin_lim[0], twin_lim[1] - twin_shift)


class _ColorMap:
    def __init__(self, colors, labels):
        self.colors = colors
        self.label = labels

    def __iter__(self):
        return zip(self.colors, self.label)


def make_colormap2(keys):
    nkeys = len(keys)
    cmap = plt.get_cmap("Pastel1")
    return _ColorMap(cmap(range(nkeys)), keys)


class PlotDataset:
    def __init__(self, col, index):
        self.x = None
        self.x_labels = None
        self.data = None
        self.err_high = None
        self.err_low = None
        self.scale_fn = None


class _Plot:
    """
    Base class for drawing plots
    """
    def __init__(self, options, outfile):
        self.options = options
        self.outfile = outfile
        self.fig, self.axes = self._make_figure()
        if type(self.axes) != np.array:
            self.axes = np.array([self.axes])

    def _make_figure(self):
        return plt.subplots(1, 1, figsize=(10, 5))

    @property
    def output_path(self):
        self.options.output / self.outfile

    def iteraxes(self):
        return np.nditer(self.axes)

    def draw(self):
        plt.savefig(self.output_path)
        if not self.options.silent:
            plt.show()
        plt.close(self.fig)


class MultiPlot(_Plot):
    """Wrapper for multi row/column plots"""
    def __init__(self, options, outfile, rows, cols):
        self.rows = rows
        self.cols = cols
        super().__init__(options, outfile)

    def _make_figure(self):
        return plt.subplots(self.rows, self.cols, figsize=(20, 5))

    def get_axis(self, row, col):
        assert row < self.rows
        assert col < self.cols
        if len(self.axes.shape) == 1:
            return self.axes[max(row, col)]
        return self.axes[row, col]


class MultiBarPlot(MultiPlot):
    """Wrapper for multi row/column bar plots"""
    def __init__(self, options, outfile, rows, cols):
        super().__init__(options, outfile, rows, cols)
        for ax in self.iteraxes():
            ax.xaxis.grid(True, linestyle="--")
            ax.axhline(y=0, linestyle="--", color="gray")
        self.bar_width = 0.8

    def set_title(self, title):
        self.fig.suptitle(title)

    def plot_at(self, row, col, x, y, xlabels=None, ylabels=None, xdesc=None, ydesc=None, **kwargs):
        ax = self.get_axis(row, col)
        ax.set_xticks(x)
        if xlabels is not None:
            ax.set_xticklabels(xlabels, rotation=90)

        ax.bar(x, y, **kwargs)


class _StackedPlot:
    """Base class for stacked plots"""
    def __init__(self, options, title, benchmark, nmetrics, ncmp=2, ygrid="main", outfile=None):
        self.options = options
        # main parameters setup
        self.benchmark_name = benchmark
        if outfile is None:
            outfile = "{}-combined-overhead.pdf".format(benchmark)
        self.outfile = outfile
        # main figure setup
        self.fig, self.axes = plt.subplots(nmetrics, 1, sharex=True, figsize=(10, 5 * nmetrics))
        try:
            self.axes[0]
        except:
            self.axes = [self.axes]
        self.fig.subplots_adjust(hspace=0)
        self.ygrid = ygrid
        for ax in self.axes:
            ax.xaxis.grid(True, linestyle="--")
            ax.axhline(y=0, linestyle="--", color="gray")
        self.twinax = [ax.twinx() for ax in self.axes]
        self.xticks = []
        self.xtick_group = ncmp
        self.fig.suptitle(title, x=0.5, y=0.998)


class StackedBarPlot(_StackedPlot):
    """Stacked bars for combined overhead plotting"""
    def __init__(self, options, title, benchmark, nmetrics, **kwargs):
        """
        options: command line options
        title: plot title
        benchmark: benchmark name
        nmetrics: number of stacked metrics plots
        ncmp: number of configurations to compare for each metric plot
        """
        super().__init__(options, title, benchmark, nmetrics, **kwargs)
        # tunables
        self.bar_width = 0.8

    def setup_xaxis(self, xlabels):
        self.xticks = np.arange(1, len(xlabels) * self.xtick_group + 1, self.xtick_group)
        self.axes[-1].set_xticks(self.xticks)
        self.axes[-1].set_xticklabels(xlabels, rotation=90)
        self.axes[0].tick_params(labelbottom=False, labeltop=True)
        self.axes[0].set_xticks(self.xticks)
        self.axes[0].set_xticklabels(xlabels, rotation=90)

    def plot_main_axis(self, data, err_hi, err_lo, labels, color, group_offset=0):
        """Plot data groups on main Y axis"""
        assert len(data.columns) == len(self.axes)
        check_nan(data, "NaN in data")
        check_nan(err_hi, "NaN in err_hi")
        check_nan(err_lo, "NaN in err_lo")

        for dcol, ehcol, elcol, ax, label in zip(data, err_hi, err_lo, self.axes, labels):
            scale = calc_scale(np.min(data[dcol] - err_lo[elcol]), np.max(data[dcol] + err_hi[ehcol]))
            assert len(data[dcol]) == len(err_lo[elcol])
            assert len(data[dcol]) == len(err_hi[ehcol])
            yerr = [err_lo[elcol] / 10**scale, err_hi[ehcol] / 10**scale]
            bar_unit = self.bar_width / self.xtick_group
            bar_offset = group_offset - int(self.xtick_group / 2)
            ax.bar(self.xticks + bar_offset * bar_unit,
                   data[dcol] / 10**scale,
                   yerr=yerr,
                   width=bar_unit,
                   color=color,
                   capsize=2)
            if scale:
                ax.set_ylabel(r"$\Delta$ {} ($10^{}$)".format(label, scale))
            else:
                ax.set_ylabel(r"$\Delta$ {}".format(label))

            if self.ygrid == "main":
                ax.yaxis.grid(True, linestyle="--", color="gray")

    def plot_twin_axis(self, data, err_hi, err_lo, labels, color, group_offset=1):
        """Plot data group on the twin Y axis"""
        assert len(data.columns) == len(self.axes)
        check_nan(data, "NaN in data")
        check_nan(err_hi, "NaN in err_hi")
        check_nan(err_lo, "NaN in err_lo")

        for dcol, ehcol, elcol, ax, label in zip(data, err_hi, err_lo, self.twinax, labels):
            assert len(data[dcol]) == len(err_lo[elcol])
            assert len(data[dcol]) == len(err_hi[ehcol])
            yerr = [err_lo[elcol], err_hi[ehcol]]
            bar_unit = self.bar_width / self.xtick_group
            bar_offset = group_offset - int(self.xtick_group / 2)
            ax.bar(self.xticks + bar_offset * bar_unit, data[dcol], yerr=yerr, width=bar_unit, color=color, capsize=2)
            ax.set_ylabel(r"% change in {}".format(label))

            if self.ygrid == "twin":
                ax.yaxis.grid(True, linestyle="--", color="gray")

    def draw(self, colormap):
        for ax, twin in zip(self.axes, self.twinax):
            tmin, tmax = twin.get_ylim()
            align_twin_axes(ax, twin, tmin, tmax)
        legend_items = []
        for color, label in colormap:
            legend_items.append(mpatches.Patch(facecolor=color, edgecolor=color, label=label))
        self.axes[0].legend(handles=legend_items,
                            bbox_to_anchor=(0, 1.32, 1, 0.2),
                            loc="lower left",
                            mode="expand",
                            ncol=2)
        outpath = self.options.output / self.outfile
        plt.savefig(outpath)
        if not self.options.silent:
            plt.show()
        plt.close(self.fig)


class StackedLinePlot(_StackedPlot):
    """Stacked line graphs for combined metrics plot over a parameter"""
    def __init__(self, options, title, benchmark, nmetrics, **kwargs):
        super().__init__(options, title, benchmark, nmetrics, **kwargs)
        self.axis_scale = []

    def setup_xaxis(self, xticks, xlabels=None, scale="linear"):
        assert len(set(xticks)) == len(xticks), "Non-unique X axis?"
        if xlabels is None:
            xlabels = map(str, xticks)

        self.xticks = xticks
        self.axes[-1].set_xticks(self.xticks)
        self.axes[-1].set_xticklabels(xlabels, rotation=90)
        self.axes[0].tick_params(labelbottom=False, labeltop=True)
        self.axes[0].set_xticks(self.xticks)
        self.axes[0].set_xticklabels(xlabels, rotation=90)

        if scale == "linear":
            for ax in self.axes:
                ax.set_xscale(scale)
        elif scale == "log2":
            for ax in self.axes:
                ax.set_xscale("log", basex=2)
        else:
            fatal("Invalid StackedLinePlot X scale {}".format(scale))

        for ax in self.twinax:
            ax.axis("off")
        self.axis_scale = [0] * len(self.axes)

    def plot_main_axis2(self, data, err_hi, err_lo, subplot_labels, dataset_index, colormap):
        """
        Plot data on the main axis.
        """
        assert len(data.columns) == len(self.axes)
        check_nan(data, "NaN in data")
        check_nan(err_hi, "NaN in err_hi")
        check_nan(err_lo, "NaN in err_lo")

        data_groups = data.groupby(dataset_index)
        errhi_groups = err_hi.groupby(dataset_index)
        errlo_groups = err_lo.groupby(dataset_index)
        assert len(data_groups) == len(errhi_groups) and len(data_groups) == len(errlo_groups)
        assert data_groups.indices == errhi_groups.indices and data_groups.indices == errlo_groups.indices

        # XXX precompute yscale for now, use a proper Scale in matplotlib
        for idx, group_name, color in zip(data_groups.indices, data_groups.groups, colormap.colors):
            data_group = data_groups.get_group(group_name)
            errhi_group = errhi_groups.get_group(group_name)
            errlo_group = errlo_groups.get_group(group_name)
            self.compute_yaxis_scale(data_group, errhi_group, errlo_group)

        for idx, group_name, color in zip(data_groups.indices, data_groups.groups, colormap.colors):
            data_group = data_groups.get_group(group_name)
            errhi_group = errhi_groups.get_group(group_name)
            errlo_group = errlo_groups.get_group(group_name)
            self.plot_main_axis_dataset(data_group, errhi_group, errlo_group, subplot_labels, color)

    def compute_yaxis_scale(self, data, err_hi, err_lo):
        for col, errhi_col, errlo_col, ax_index in zip(data, err_hi, err_lo, range(len(self.axes))):
            scale = calc_scale(np.min(data[col] - err_lo[errlo_col]), np.max(data[col] + err_hi[errhi_col]))
            self.axis_scale[ax_index] = max(self.axis_scale[ax_index], scale)

    def plot_main_axis_dataset(self, data, err_hi, err_lo, subplot_labels, color):
        assert len(data.columns) == len(self.axes), "Metrics amount does not match axes"
        assert len(data) == len(self.xticks), "Data points amount does not match X axis"
        check_nan(data, "NaN in data")
        check_nan(err_hi, "NaN in err_hi")
        check_nan(err_lo, "NaN in err_lo")

        for col, errhi_col, errlo_col, ax, scale, label in zip(data, err_hi, err_lo, self.axes, self.axis_scale,
                                                               subplot_labels):
            yerr = [err_lo[errlo_col] / 10**scale, err_hi[errhi_col] / 10**scale]

            ax.errorbar(self.xticks, data[col] / 10**scale, yerr=yerr, color=color, capsize=3)
            ax.scatter(self.xticks, data[col] / 10**scale, color=color, s=2, marker='o')
            if scale:
                ax.set_ylabel(r"{} ($10^{}$)".format(label, scale))
            else:
                ax.set_ylabel(label)

            if self.ygrid == "main":
                ax.yaxis.grid(True, linestyle="--", color="gray")

    def plot_main_axis(self, data, err_hi, err_lo, labels, color):
        """Plot data groups on main Y axis"""
        assert len(data.columns) == len(self.axes)
        check_nan(data, "NaN in data")
        check_nan(err_hi, "NaN in err_hi")
        check_nan(err_lo, "NaN in err_lo")

        for dcol, ehcol, elcol, ax, scale, label in zip(data, err_hi, err_lo, self.axes, self.axis_scale, labels):
            assert len(data[dcol]) == len(err_lo[elcol])
            assert len(data[dcol]) == len(err_hi[ehcol])
            yerr = [err_lo[elcol] / 10**scale, err_hi[ehcol] / 10**scale]

            ax.errorbar(self.xticks, data[dcol] / 10**scale, yerr=yerr, color=color, capsize=2)
            if scale:
                ax.set_ylabel(r"{} ($10^{}$)".format(label, scale))
            else:
                ax.set_ylabel(label)

            if self.ygrid == "main":
                ax.yaxis.grid(True, linestyle="--", color="gray")

    def scatter_main_axis(self, data, color):
        """Scatter individual data points on main Y axis"""
        assert len(data.columns) == len(self.axes)
        check_nan(data, "NaN in data")

        for dcol, ax, scale, label in zip(data, self.axes, self.axis_scale, labels):
            ax.scatter(self.xticks, data[dcol] / 10**scale, marker="x", color=color)

    def annotate_points(self, data, labels):
        """Annotate points in the plot with given labels"""
        assert len(data.columns) == len(self.axes)
        check_nan(data, "NaN in data")

        for dcol, ax, scale, lcol in zip(data, self.axes, self.axis_scale, labels):
            for value, x, label in zip(data[dcol], self.xticks, labels[lcol]):
                ax.annotate("{:.1f}%".format(label),
                            xy=(x, value / 10**scale),
                            textcoords="offset points",
                            xytext=(0, 10))

    def draw(self, colormap):
        for ax, twin in zip(self.axes, self.twinax):
            tmin, tmax = twin.get_ylim()
            align_twin_axes(ax, twin, tmin, tmax)
        legend_items = []
        for color, label in colormap:
            legend_items.append(mpatches.Patch(facecolor=color, edgecolor=color, label=label))
        if len(self.axes) == 1:
            self.axes[0].legend(handles=legend_items, ncol=2)
        else:
            self.axes[0].legend(handles=legend_items,
                                bbox_to_anchor=(0, 1.32, 1, 0.2),
                                loc="lower left",
                                mode="expand",
                                ncol=2)
        # Adjust spacing between title and legend based on the legend rows
        self.fig.subplots_adjust(top=1 - 0.01 * len(legend_items) / 2, bottom=0)
        outpath = self.options.output / "{}-combined-lines.pdf".format(self.benchmark_name)
        plt.savefig(outpath)
        outpath = self.options.output / "{}-combined-lines.svg".format(self.benchmark_name)
        plt.savefig(outpath)
        if not self.options.silent:
            plt.show()
        plt.close(self.fig)
