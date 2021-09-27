import logging
from enum import Enum

import pandas as pd
import numpy as np

from ..core.plot import (Plot, check_multi_index_aligned, subset_xs, rotate_multi_index_level, StackedLinePlot,
                         StackedBarPlot, make_colormap2, ColorMap)
from ..core.html import HTMLSurface

class NetperfQEMUStatsExplorationTable(Plot):
    """
    Note: this does not support the matplotlib surface
    """
    def __init__(self, benchmark, pmc_dset, qemu_bb_dset, qemu_call_dset, surface=HTMLSurface()):
        super().__init__(benchmark, surface)
        self.pmc_dset = pmc_dset
        self.qemu_bb_dset = qemu_bb_dset
        self.qemu_call_dset = qemu_call_dset

    def _get_plot_title(self):
        return "Netperf PC hit count exploration"

    def _get_plot_file(self):
        path = self.benchmark.manager_config.output_path / "netperf-pc-table.{}".format(self.surface.output_file_ext())
        return path

    def _get_legend_map(self):
        legend = {
            uuid: str(bench.instance_config.kernelabi)
            for uuid, bench in self.benchmark.merged_benchmarks.items()
        }
        legend[self.benchmark.uuid] = f"{self.benchmark.instance_config.kernelabi}(baseline)"
        return legend

    def prepare_xs(self):
        """Preparation step for a table in a cell for the given cross-section of the main dataframe"""
        pass

    def prepare(self):
        """
        For each dataset (including the baseline) we show the dataframes as tables.
        Combine the qemu stats datasets into a single table for ease of inspection
        """
        legend_map = self._get_legend_map()
        baseline = self.benchmark.uuid
        bb_df = self.qemu_bb_dset.agg_df
        call_df = self.qemu_call_dset.agg_df
        # Note: the df here is implicitly a copy and following operations will not modify it
        df = bb_df.join(call_df, how="outer", rsuffix="_call")
        pmc = self.pmc_dset.agg_df
        # make the ratios a percentage
        df["norm_diff_count"] = df["norm_diff_count"] * 100
        df["norm_diff_call_count"] = df["norm_diff_call_count"] * 100
        if not check_multi_index_aligned(df, "__dataset_id"):
            self.logger.error("Unaligned index, skipping plot")
            return
        # PMC data XXX-AM unused for now
        icount_diff = pmc.loc[:, "diff_median_instructions"]

        self.surface.set_layout(1, 1, expand=True, how="row")
        # Table for common functions
        nonzero = df["count"].groupby(["file", "symbol"]).min() != 0
        common_syms = nonzero & (nonzero != np.nan)
        common_df = subset_xs(df, common_syms)
        view_df, colmap = rotate_multi_index_level(common_df, "__dataset_id", legend_map)
        # Decide which columns to show:
        # Showed for both the baseline and measure runs
        common_cols = ["count", "call_count"]
        # Showed only for measure runs
        measure_cols = ["diff_count", "norm_diff_count", "diff_call_count", "norm_diff_call_count"]
        show_cols = np.append(
            colmap.loc[:, common_cols].to_numpy().transpose().ravel(),
            colmap.loc[colmap.index != baseline, measure_cols].to_numpy().transpose().ravel())
        # Decide how to sort
        sort_cols = colmap.loc[colmap.index != baseline, "norm_diff_call_count"].to_numpy().ravel()

        # Build the color index column and the color map.
        # We paint red all rows with invalid symbols
        # We paint orange all rows where the call number is 0 but we appear to hit basic blocks
        valid_syms_cmap = ColorMap.base8()
        unknown_syms = view_df.index.get_level_values("file").map(lambda name: "red" if name == "<unknown>" else None)
        # check which call count numbers make sense at all and map it back to the view_dataframe index
        sensible_call_count = (common_df["count"] != 0) & (common_df["call_count"] != 0)  # Both bb_count and call_count are nonzero
        weird_syms = sensible_call_count.groupby(["file", "symbol"]).all().map(lambda v: None if v else "cyan")
        # Double check we did not mess up and misaligned the indexes
        assert (weird_syms.index.values == view_df.index.values).all()
        view_df["_color"] = weird_syms.where(~weird_syms.isna(), unknown_syms)

        # build the final view df
        view_df2 = view_df.sort_values(list(sort_cols), ascending=False, key=abs)
        cell = self.surface.make_cell(title="QEMU stats for common functions")
        view = self.surface.make_view("table", df=view_df2, yleft=show_cols,
                                      colormap=valid_syms_cmap, color_col="_color")
        cell.add_view(view)
        self.surface.next_cell(cell)

        # Table for functions that are only in one of the runs
        extra_df = subset_xs(df, ~common_syms)
        view_df, colmap = rotate_multi_index_level(extra_df, "__dataset_id", legend_map)
        view_df = view_df[show_cols].sort_values(list(sort_cols), ascending=False, key=abs)
        cell = self.surface.make_cell(title="QEMU stats for extra functions")
        view = self.surface.make_view("table", df=view_df, yleft=show_cols)
        cell.add_view(view)
        self.surface.next_cell(cell)


###################### Old stuff


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
        outfile = "netperf-{}-{}".format(self.options.config.name, "+".join(runs))
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
        logging.info("Generate plot data for %s runs:%s", self.options.config.name, list(datasets))
        cols = self._get_subplot_columns()
        plot = StackedLinePlot(self.options, "Netperf", outfile, len(cols))

        x = self.stats.index.get_level_values(self.x_index).unique()
        plot.setup_xaxis(x)
        data, err_hi, err_lo = self._get_subplot_data()
        plot.plot_main_axis2(data, err_hi, err_lo, cols, "__dataset_id", cmap)
        # plot.plot_main_axis(data, err_hi, err_lo, cols, cmap)

        logging.info("Plot %s -> %s", self.options.config.name, outfile)
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
            logging.error("%s does not support transaction_size plot", self.options.config.name)
            exit(1)

    def _get_outfile(self):
        runs = self._get_datasets()
        outfile = "netperf-{}-{}".format(self.options.config.name, "+".join(runs))
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
        logging.info("Generate plot data for %s runs:%s", self.options.config.name, list(datasets))
        cols = self._get_subplot_columns()
        plot = StackedLinePlot(self.options, "Netperf", outfile, len(cols))

        x = self.stats.index.get_level_values(self.x_index).unique()
        plot.setup_xaxis(x)
        data, err_hi, err_lo = self._get_subplot_data()
        plot.plot_main_axis2(data, err_hi, err_lo, cols, "__dataset_id", cmap)
        # plot.plot_main_axis(data, err_hi, err_lo, cols, cmap)

        logging.info("Plot %s -> %s", self.options.config.name, outfile)
        plot.draw(cmap)
