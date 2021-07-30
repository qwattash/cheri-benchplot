
import logging
from enum import Enum

from ..core.plot import StackedLinePlot, StackedBarPlot, make_colormap2


class NetperfPlot(Enum):
    QEMU_PC_HIST = "qemu-pc-hist"
    ALL_BY_XFER_SIZE = "xfer-size"

    def __str__(self):
        return self.value

class Plotter:
    def __init__(self, benchmark):
        self.benchmark = benchmark
        self.options = benchmark.options
        self.df = benchmark.merged_stats

    def _get_datasets(self):
        """Fetch the datasets (benchmark runs) to compare in each subplot"""
        return self.df.index.get_level_values("__dataset_id").unique()

    def _get_outfile(self):
        runs = self._get_datasets()
        outfile = "netperf-{}-{}".format(self.options.config.name,
                                         "+".join(runs))
        return outfile

    def _get_colormap(self):
        runs = self._get_datasets()
        # self.stats["CHERI Kernel ABI"][runs]
        return make_colormap2(runs)


class NetperfQemuPCHist(Plotter):
    """
    Plot qemu PC histograms. We output N + 1 histograms:
    1. Absolute values from each dataset
    2. Relative diff between datasets and baseline
    """

    def __init__(self, benchmark):
        super().__init__(benchmark)

    def _get_subplot_columns(self):
        """
        Produce list of columns for which we want to create a subplot in
        the stacked plot.
        """
        cols = self.benchmark.pmc.valid_data_columns() + self.netperf.data_columns()
        return cols

    def _get_subplot_data(self):
        """Extract median and error columns"""
        data_cols = col2stat("median", self.benchmark.pmc.valid_data_columns())
        data_cols += self.netperf.data_columns()
        data = self.stats[data_cols]
        err_hi = self.stats[col2stat("errhi", self._get_subplot_columns())]
        err_lo = self.stats[col2stat("errlo", self._get_subplot_columns())]
        return (data, err_hi, err_lo)

    def draw(self):
        datasets = self._get_datasets()
        outfile = self._get_outfile()
        cmap = self._get_colormap()
        logging.info("Generate plot data for %s runs:%s",
                     self.options.config.name, list(datasets))
        cols = self._get_subplot_columns()
        plot = StackedLinePlot(self.options, "Netperf", outfile, len(cols))

        x = self.stats.index.get_level_values(self.x_index).unique()
        plot.setup_xaxis(x)
        data, err_hi, err_lo = self._get_subplot_data()
        plot.plot_main_axis2(data, err_hi, err_lo, cols, "__dataset_id", cmap)
        # plot.plot_main_axis(data, err_hi, err_lo, cols, cmap)

        logging.info("Plot %s -> %s", self.options.config.name,
                     outfile)
        plot.draw(cmap)



class NetperfTXSizeStackPlot:
    """
    Plots the dataset as follows:
    X axis: transaction size
    Y1 axis: overhead with respect to baseline
    Y2 axis: absolute values of both baseline and measures
    """

    x_mapping = {
        # NetperfConfigs.UDP_RR_50K_FIXED: "Request Size Bytes"
    }

    def __init__(self, benchmark):
        self.benchmark = benchmark
        self.options = self.benchmark.options
        self.netperf = self.benchmark.netperf
        self.stats = self.benchmark.merged_stats
        try:
            self.x_index = self.x_mapping[self.options.config]
        except KeyError:
            logging.error("%s does not support transaction_size plot",
                          self.options.config.name)
            exit(1)

    def _get_outfile(self):
        runs = self._get_datasets()
        outfile = "netperf-{}-{}".format(self.options.config.name,
                                         "+".join(runs))
        return outfile

    def _get_colormap(self):
        runs = self._get_datasets()
        # self.stats["CHERI Kernel ABI"][runs]
        return make_colormap2(runs)

    def _get_datasets(self):
        """Fetch the datasets (benchmark runs) to compare in each subplot"""
        return self.stats.index.get_level_values("__dataset_id").unique()

    def _get_subplot_columns(self):
        """
        Produce list of columns for which we want to create a subplot in
        the stacked plot.
        """
        cols = self.benchmark.pmc.valid_data_columns() + self.netperf.data_columns()
        return cols

    def _get_subplot_data(self):
        """Extract median and error columns"""
        data_cols = col2stat("median", self.benchmark.pmc.valid_data_columns())
        data_cols += self.netperf.data_columns()
        data = self.stats[data_cols]
        err_hi = self.stats[col2stat("errhi", self._get_subplot_columns())]
        err_lo = self.stats[col2stat("errlo", self._get_subplot_columns())]
        return (data, err_hi, err_lo)

    def draw(self):
        datasets = self._get_datasets()
        outfile = self._get_outfile()
        cmap = self._get_colormap()
        logging.info("Generate plot data for %s runs:%s",
                     self.options.config.name, list(datasets))
        cols = self._get_subplot_columns()
        plot = StackedLinePlot(self.options, "Netperf", outfile, len(cols))

        x = self.stats.index.get_level_values(self.x_index).unique()
        plot.setup_xaxis(x)
        data, err_hi, err_lo = self._get_subplot_data()
        plot.plot_main_axis2(data, err_hi, err_lo, cols, "__dataset_id", cmap)
        # plot.plot_main_axis(data, err_hi, err_lo, cols, cmap)

        logging.info("Plot %s -> %s", self.options.config.name,
                     outfile)
        plot.draw(cmap)
