import numpy as np
import pandas as pd

from ..core.dataset import DatasetName
from ..core.plot import (BarPlotDataView, BenchmarkPlot, BenchmarkSubPlot, LinePlotDataView, Mosaic, Scale,
                         get_col_or_idx)


class NetperfDataPlot(BenchmarkSubPlot):
    """
    Common netperf single-metric plot
    """
    def __init__(self, plot, metric, param=None, overhead=False):
        super().__init__(plot)
        self.ds = self.get_dataset(DatasetName.NETPERF_DATA)
        self.param = param
        self.metric = metric
        self.is_overhead = overhead
        if overhead:
            delta = "norm_delta_baseline"
            self.y_unit = "% "
        else:
            delta = "sample"
            self.y_unit = ""
        self.col = (metric, "median", delta)
        self.err_hi_col = (metric, "q75", delta)
        self.err_lo_col = (metric, "q25", delta)

        if metric == "Throughput":
            # Fetch the thoughput scale
            if self.ds.merged_df.groupby("Throughput Units").ngroups > 1:
                self.logger.warning("netperf throughput units differ: %s",
                                    self.ds.merged_df["Throughput Units"].unique())
            else:
                self.y_unit += self.ds.merged_df["Throughput Units"].unique()[0]

    def get_cell_title(self):
        if self.param:
            return f"Netperf {self.metric} w.r.t. {self.param}"
        else:
            return f"Netperf {self.metric}"

    def get_legend_info(self):
        base = self.build_legend_by_gid()
        return base.assign_colors_hsv("dataset_gid", h=(0.2, 1), s=(0.7, 0.7), v=(0.6, 0.9))

    def get_df(self):
        if self.param is None:
            return self.ds.agg_df.copy()
        else:
            return self.ds.cross_merged_df.copy()

    def _adjust_columns(self, df):
        """Adjust error columns to be relative to the median and make norm data a percentage"""
        # plots want relative errors
        df[self.err_hi_col] = df[self.err_hi_col] - df[self.col]
        df[self.err_lo_col] = df[self.col] - df[self.err_lo_col]
        if self.is_overhead:
            df.loc[:, self.col] *= 100
            df.loc[:, self.err_hi_col] *= 100
            df.loc[:, self.err_lo_col] *= 100
        return df


class NetperfBar(NetperfDataPlot):
    """
    Display a single netperf metric as a bar plot
    """
    def generate(self, fm, cell):
        self.logger.debug("extract Netperf metric %s (relative=%s)", self.metric, self.is_overhead)
        df = self.get_df()
        # We have a single group of bars for each plot, so just 1 seed coordinate
        df["x"] = 0
        df = self._adjust_columns(df)

        view = BarPlotDataView(df, x="x", yleft=self.col, err_hi=self.err_hi_col, err_lo=self.err_lo_col)
        view.bar_group = "dataset_gid"
        view.legend_info = self.get_legend_info()
        view.legend_level = ["dataset_gid"]
        cell.add_view(view)

        cell.x_config.ticks = [0]
        cell.x_config.tick_labels = [self.metric]
        cell.x_config.padding = 0.4
        cell.yleft_config.label = f"value {self.y_unit}"


class NetperfParamScalingPlot(NetperfDataPlot):
    """
    Generate line plot to show netperf stats scaling along a parameterisation axis.
    """
    def generate(self, fm, cell):
        """
        TODO the generation function is mostly the same everywhere for these plots,
        move to a shared base class?
        """
        df = self.get_df()
        self.logger.debug("Extract X %s along %s (relative=%s)", self.metric, self.param, self.is_overhead)
        df = self._adjust_columns(df)

        # Detect whether we should use a log scale for the X axis
        x_values = get_col_or_idx(df, self.param).unique()
        x_values = map(lambda x: int(np.log2(x)) == np.log2(x), x_values)
        if np.all(x_values):
            scale = Scale("log", base=2)
        else:
            scale = None

        view = LinePlotDataView(df, x=self.param, yleft=self.col, err_hi=self.err_hi_col, err_lo=self.err_lo_col)
        view.line_group = ["dataset_gid"]
        view.legend_info = self.build_legend_by_gid()
        view.legend_level = ["dataset_gid"]
        cell.add_view(view)

        cell.x_config.label = self.param
        cell.x_config.ticks = sorted(df.index.unique(self.param))
        cell.x_config.scale = scale
        cell.yleft_config.label = self.metric


class NetperfStats(BenchmarkPlot):
    """
    Interesting netperf reported statistics
    """
    require = {DatasetName.NETPERF_DATA}
    name = "netperf-stats"
    description = "Netperf stats"

    def _make_subplots_mosaic(self):
        netperf = self.get_dataset(DatasetName.NETPERF_DATA)
        m = self._make_mosaic_from_dataset_columns(NetperfBar, netperf, include_overhead=True)
        return m


class NetperfScalingStats(BenchmarkPlot):
    """
    Netperf throughput scaling across benchmark variants.
    """
    require = {DatasetName.NETPERF_DATA}
    name = "netperf-scaling-stats"
    description = "Netperf stats across benchmark variants"
    cross_analysis = True

    def _make_subplots_mosaic(self):
        netperf = self.get_dataset(DatasetName.NETPERF_DATA)
        m = self._make_mosaic_from_cross_merged_dataset(NetperfParamScalingPlot, netperf, include_overhead=True)
        return m
