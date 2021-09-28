import logging
from enum import Enum

import pandas as pd
import numpy as np

from ..core.dataset import subset_xs, rotate_multi_index_level, reorder_columns
from ..core.plot import (Plot, check_multi_index_aligned, StackedLinePlot, StackedBarPlot, make_colormap2, ColorMap)
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

    def _prepare_xs(self, df_xs, cell_title):
        """
        Prepare the given cross-section of the dataframe as a table in a cell of the plot.
        """
        legend_map = self._get_legend_map()
        baseline = self.benchmark.uuid
        view_df, colmap = rotate_multi_index_level(df_xs, "__dataset_id", legend_map)
        # Decide which columns to show:
        # Showed for both the baseline and measure runs
        common_cols = ["bb_count", "call_count", "start", "start_call"]  ## XXX-AM find a way to format columns
        # Showed only for measure runs
        relative_cols = ["delta_bb_count", "norm_delta_bb_count", "delta_call_count", "norm_delta_call_count"]
        show_cols = np.append(colmap.loc[:, common_cols].to_numpy().transpose().ravel(),
                              colmap.loc[colmap.index != baseline, relative_cols].to_numpy().transpose().ravel())
        # Decide how to sort and order columns
        sort_cols = colmap.loc[colmap.index != baseline, "norm_delta_call_count"].to_numpy().ravel()
        view_df = view_df.sort_values(list(sort_cols), ascending=False, key=abs)
        view_df = reorder_columns(view_df, show_cols)
        # Add the view for drawing
        cell = self.surface.make_cell(title=cell_title)
        view = self.surface.make_view("table", df=view_df, yleft=show_cols)
        cell.add_view(view)
        self.surface.next_cell(cell)

    def _get_common_symbols_xs(self, df):
        """
        Return a dataframe containing the cross section of the (joined) qemu stats dataframe
        containing the symbols that are common to all runs (i.e. across all __dataset_id values).
        We consider valid common symbols those for which we were able to resolve the (file, sym_name)
        pair and have sensible BB count and call_count values.
        """
        nonzero = df[["bb_count", "call_count"]].groupby(["file", "symbol"]).min().fillna(0) != 0
        common_syms = nonzero.any(axis=1)
        common_df = subset_xs(df, common_syms)
        # Filter remaining symbols for validity
        valid_syms = common_df.index.get_level_values("file") != "[unknown]"
        valid_call = (common_df["bb_count"] != 0) & (common_df["call_count"] != 0)
        valid_df = common_df[valid_syms & valid_call]
        return valid_df

    def _get_unique_symbols_xs(self, df):
        """
        This is complementary to _get_common_symbols_xs().
        """
        anyzero = df[["bb_count", "call_count"]].groupby(["file", "symbol"]).min().fillna(0) == 0
        unique_syms = anyzero.any(axis=1)
        unique_df = subset_xs(df, unique_syms)
        # Filter remaining symbols for validity
        valid_syms = unique_df.index.get_level_values("file") != "[unknown]"
        valid_call = (unique_df["bb_count"] != 0) & (unique_df["call_count"] != 0)
        valid_df = unique_df[valid_syms & valid_call]
        return valid_df

    def _get_missing_symbols_xs(self, df):
        """
        Return a cross-section of the (joined) qemu stats dataframe containing the symbols
        that could not be resolved for some of the datasets.
        """
        invalid_syms = df.index.get_level_values("file") == "[unknown]"
        valid_call = (df["bb_count"] != 0) & (df["call_count"] != 0)
        return df[invalid_syms & valid_call]

    def _get_inconsistent_symbols_xs(self, df):
        """
        Return a cross-section of the (joined) qemu stats dataframe containing inconsistent
        records, for which we have BB hits but no calls or calls without any BB hit.
        """
        invalid_call = (df["bb_count"] == 0) | (df["call_count"] == 0)
        return df[invalid_call]

    def prepare(self):
        """
        For each dataset (including the baseline) we show the dataframes as tables.
        Combine the qemu stats datasets into a single table for ease of inspection
        """
        bb_df = self.qemu_bb_dset.agg_df
        call_df = self.qemu_call_dset.agg_df
        # Note: the df here is implicitly a copy and following operations will not modify it
        df = bb_df.join(call_df, how="outer", rsuffix="_call")
        # pmc = self.pmc_dset.agg_df

        # Make the ratios a percentage
        df["norm_delta_bb_count"] = df["norm_delta_bb_count"] * 100
        df["norm_delta_call_count"] = df["norm_delta_call_count"] * 100
        # Now plot each table
        if not check_multi_index_aligned(df, "__dataset_id"):
            self.logger.error("Unaligned index, skipping plot")
            return
        self.surface.set_layout(1, 1, expand=True, how="row")
        common_df = self._get_common_symbols_xs(df)
        self._prepare_xs(common_df, "QEMU stats for common functions")
        unique_df = self._get_unique_symbols_xs(df)
        self._prepare_xs(unique_df, "QEMU stats for unique functions")
        missing_syms_df = self._get_missing_symbols_xs(df)
        self._prepare_xs(missing_syms_df, "QEMU stats for unresolved functions")
        inconsistent_df = self._get_inconsistent_symbols_xs(df)
        self._prepare_xs(inconsistent_df, "Inconsistent QEMU stats records")


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
    2. Relative delta between datasets and baseline
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
