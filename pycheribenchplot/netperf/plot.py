import numpy as np
import pandas as pd

from ..core.dataset import DatasetName
from ..core.plot import (BarPlotDataView, BenchmarkPlot, BenchmarkSubPlot, Mosaic)


class NetperfBar(BenchmarkSubPlot):
    """
    Display a single netperf metric as a bar plot
    """
    def __init__(self, plot, metric, overhead=False):
        super().__init__(plot)
        self.ds = self.get_dataset(DatasetName.NETPERF_DATA)
        self.metric = metric
        if overhead:
            delta = "norm_delta_baseline"
        else:
            delta = "sample"
        self.col = (metric, "median", delta)
        self.err_hi_col = (metric, "q75", delta)
        self.err_lo_col = (metric, "q25", delta)

        self.y_unit = ""
        if metric == "Throughput":
            # Fetch the thoughput scale
            if self.ds.merged_df.groupby("Throughput Units").ngroups > 1:
                self.logger.warning("netperf throughput units differ: %s",
                                    self.ds.merged_df["Throughput Units"].unique())
            else:
                self.y_unit = self.ds.merged_df["Throughput Units"].unique()[0]

    def get_cell_title(self):
        return f"Netperf {self.metric}"

    def get_legend_info(self):
        base = self.build_legend_by_dataset()
        return base.assign_colors_hsv("dataset_id", h=(0.2, 1), s=(0.7, 0.7), v=(0.6, 0.9))

    def generate(self, fm, cell):
        self.logger.debug("extract Netperf metric %s", self.metric)
        df = self.ds.agg_df.copy()
        # We have a single group of bars for each plot, so just 1 seed coordinate
        df["x"] = 0

        # bar plots want relative errors
        df[self.err_hi_col] = df[self.err_hi_col] - df[self.col]
        df[self.err_lo_col] = df[self.col] - df[self.err_lo_col]

        view = BarPlotDataView(df, x="x", yleft=self.col, err_hi=self.err_hi_col, err_lo=self.err_lo_col)
        view.bar_group = "dataset_id"
        view.legend_info = self.get_legend_info()
        view.legend_level = ["dataset_id"]
        cell.add_view(view)

        cell.x_config.ticks = [0]
        cell.x_config.tick_labels = [self.metric]
        cell.x_config.padding = 0.4
        cell.yleft_config.label = f"value {self.y_unit}"


class NetperfStats(BenchmarkPlot):
    """
    Interesting netperf reported statistics
    """
    require = {DatasetName.NETPERF_DATA}
    name = "netperf-stats"

    def _make_subplots_mosaic(self):
        subplots = {}
        layout = []
        netperf = self.get_dataset(DatasetName.NETPERF_DATA)
        for idx, metric in enumerate(netperf.data_columns()):
            name_abs = f"subplot-netperf-stats-{idx}-abs"
            name_delta = f"subplot-netperf-stats-{idx}-delta"
            if netperf.merged_df[metric].isna().all():
                continue
            subplots[name_abs] = NetperfBar(self, metric)
            subplots[name_delta] = NetperfBar(self, metric, True)
            layout.append([name_abs, name_delta])
        return Mosaic(layout, subplots)

    def get_plot_name(self):
        return "Netperf stats"
