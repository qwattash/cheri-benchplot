import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from .dataset import *


def align_twin_axes(ax, ax_twin, min_twin, max_twin):
    """
    Align 0 point on twin Y axes
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
    margin = 0.1 # percentage
    if min_twin < twin_lim[0]:
        # adjust min
        # get size of adjustment in figure space
        min_twin = min_twin - np.absolute(margin * min_twin)
        _, dy = (ttwin.transform((0, 0)) -
                 ttwin.transform((0, twin_lim[0] - min_twin)))
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

class ColorMap:
    def __init__(self, colors, labels):
        self.colors = colors
        self.label = labels

    def __iter__(self):
        return zip(self.colors, self.label)


def make_colormap2(keys):
    nkeys = len(keys)
    cmap = plt.get_cmap("Pastel1")
    return ColorMap(cmap(range(nkeys)), keys)


class PlotDataset:

    def __init__(self, col, index):
        self.x = None
        self.x_labels = None
        self.data = None
        self.err_high = None
        self.err_low = None
        self.scale_fn = None


class Plot:
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


class MultiPlot(Plot):
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

    def plot_at(self, row, col, x, y, xlabels=None, ylabels=None,
                xdesc=None, ydesc=None, **kwargs):
        ax = self.get_axis(row, col)
        ax.set_xticks(x)
        if xlabels is not None:
            ax.set_xticklabels(xlabels, rotation=90)

        ax.bar(x, y, **kwargs)


class StackedPlot:
    """Base class for stacked plots"""

    def __init__(self, options, title, benchmark, nmetrics, ncmp=2, ygrid="main", outfile=None):
        self.options = options
        # main parameters setup
        self.benchmark_name = benchmark
        if outfile is None:
            outfile = "{}-combined-overhead.pdf".format(benchmark)
        self.outfile = outfile
        # main figure setup
        self.fig, self.axes = plt.subplots(
            nmetrics, 1, sharex=True, figsize=(10, 5 * nmetrics))
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


class StackedBarPlot(StackedPlot):
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
        self.xticks = np.arange(1, len(xlabels)*self.xtick_group + 1,
                                self.xtick_group)
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
            scale = calc_scale(np.min(data[dcol] - err_lo[elcol]),
                               np.max(data[dcol] + err_hi[ehcol]))
            assert len(data[dcol]) == len(err_lo[elcol])
            assert len(data[dcol]) == len(err_hi[ehcol])
            yerr = [err_lo[elcol] / 10**scale, err_hi[ehcol] / 10**scale]
            bar_unit = self.bar_width / self.xtick_group
            bar_offset = group_offset - int(self.xtick_group / 2)
            ax.bar(self.xticks + bar_offset * bar_unit,
                   data[dcol] / 10**scale, yerr=yerr, width=bar_unit,
                   color=color, capsize=2)
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
            ax.bar(self.xticks + bar_offset * bar_unit,
                   data[dcol], yerr=yerr, width=bar_unit,
                   color=color, capsize=2)
            ax.set_ylabel(r"% change in {}".format(label))

            if self.ygrid == "twin":
                ax.yaxis.grid(True, linestyle="--", color="gray")

    def draw(self, colormap):
        for ax, twin in zip(self.axes, self.twinax):
            tmin, tmax = twin.get_ylim()
            align_twin_axes(ax, twin, tmin, tmax)
        legend_items = []
        for color, label in colormap:
            legend_items.append(mpatches.Patch(
                facecolor=color, edgecolor=color, label=label))
        self.axes[0].legend(handles=legend_items,
                            bbox_to_anchor=(0, 1.32, 1, 0.2),
                            loc="lower left", mode="expand", ncol=2)
        outpath = self.options.output / self.outfile
        plt.savefig(outpath)
        if not self.options.silent:
            plt.show()
        plt.close(self.fig)


class StackedLinePlot(StackedPlot):
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
            scale = calc_scale(np.min(data[col] - err_lo[errlo_col]),
                               np.max(data[col] + err_hi[errhi_col]))
            self.axis_scale[ax_index] = max(self.axis_scale[ax_index], scale)

    def plot_main_axis_dataset(self, data, err_hi, err_lo, subplot_labels, color):
        assert len(data.columns) == len(self.axes), "Metrics amount does not match axes"
        assert len(data) == len(self.xticks), "Data points amount does not match X axis"
        check_nan(data, "NaN in data")
        check_nan(err_hi, "NaN in err_hi")
        check_nan(err_lo, "NaN in err_lo")

        for col, errhi_col, errlo_col, ax, scale, label in zip(
                data, err_hi, err_lo, self.axes, self.axis_scale, subplot_labels):
            yerr = [err_lo[errlo_col] / 10**scale, err_hi[errhi_col] / 10**scale]

            ax.errorbar(self.xticks, data[col] / 10**scale, yerr=yerr,
                        color=color, capsize=3)
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

        for dcol, ehcol, elcol, ax, scale, label in zip(
                data, err_hi, err_lo, self.axes, self.axis_scale, labels):
            assert len(data[dcol]) == len(err_lo[elcol])
            assert len(data[dcol]) == len(err_hi[ehcol])
            yerr = [err_lo[elcol] / 10**scale, err_hi[ehcol] / 10**scale]

            ax.errorbar(self.xticks, data[dcol] / 10**scale, yerr=yerr,
                        color=color, capsize=2)
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
                ax.annotate("{:.1f}%".format(label), xy=(x, value / 10**scale),
                            textcoords="offset points", xytext=(0, 10))

    def draw(self, colormap):
        for ax, twin in zip(self.axes, self.twinax):
            tmin, tmax = twin.get_ylim()
            align_twin_axes(ax, twin, tmin, tmax)
        legend_items = []
        for color, label in colormap:
            legend_items.append(mpatches.Patch(
                facecolor=color, edgecolor=color, label=label))
        if len(self.axes) == 1:
            self.axes[0].legend(handles=legend_items, ncol=2)
        else:
            self.axes[0].legend(handles=legend_items,
                                bbox_to_anchor=(0, 1.32, 1, 0.2),
                                loc="lower left", mode="expand", ncol=2)
        # Adjust spacing between title and legend based on the legend rows
        self.fig.subplots_adjust(top=1 - 0.01 * len(legend_items) / 2, bottom=0)
        outpath = self.options.output / "{}-combined-lines.pdf".format(self.benchmark_name)
        plt.savefig(outpath)
        outpath = self.options.output / "{}-combined-lines.svg".format(self.benchmark_name)
        plt.savefig(outpath)
        if not self.options.silent:
            plt.show()
        plt.close(self.fig)
