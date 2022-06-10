import numpy as np
import pandas as pd

from ..core.dataset import (DatasetName, align_multi_index_levels, check_multi_index_aligned, pivot_multi_index_level)
from ..core.plot import (BenchmarkPlot, BenchmarkSubPlot, BenchmarkTable, LegendInfo, Scale, ScatterPlotDataView, Style,
                         Symbols, TableDataView)


class QEMUBBHitTable(BenchmarkSubPlot):
    """Basic block hits per function"""
    description = "QEMU Basic-Block function hits"

    def _get_sort_metric(self, df):
        return df[("hit_count", "median", "sample")]
        # sort_col = ("call_count", "mean", "delta_baseline")
        # total_delta_calls = df[sort_col].abs().sum()
        # relevance = (df[sort_col] / total_delta_calls).abs()
        # return relevance

    def _get_show_columns(self, view_df, legend_info):
        # For each data column, we show all the aggregation metrics for the
        # "sample" delta level key.
        # The remaining delta level keys are shown only for non-baseline columns.
        baseline = legend_info.labels[self.benchmark.uuid][0]
        data = (view_df.columns.get_level_values("metric") != "sort_metric")
        sample = data & (view_df.columns.get_level_values("delta") == "sample")
        delta = data & (~sample) & (view_df.columns.get_level_values("dataset_id") != baseline)
        select_cols = view_df.columns[sample]
        select_cols = select_cols.append(view_df.columns[delta])
        sorted_cols, _ = select_cols.sortlevel()
        return sorted_cols

    def _get_sort_columns(self, view_df, sort_col):
        sel = (slice(None), ) * (view_df.columns.nlevels - 1)
        indexes = view_df.columns.get_locs((sort_col, ) + sel)
        return view_df.columns[indexes].to_list()

    def _get_pivot_legend_info(self, df, legend_info):
        """
        Generate a legend map for the pivoted view_df.
        This will map <column index tuple> => (<dataset_id label>, <color>)
        """
        col_df = df.columns.to_frame()
        by_label = legend_info.with_label_index()
        _, pivot_colors = col_df.align(by_label["colors"], axis=0, level="dataset_id")
        _, pivot_labels = col_df.align(by_label["labels"], axis=0, level="dataset_id")
        new_map = LegendInfo.from_index(df.columns, colors=pivot_colors, labels=pivot_labels)
        return new_map

    def get_legend_info(self):
        legend = self.build_legend_by_dataset()
        legend.remap_colors("Greys", color_range=(0, 0.5))
        return legend

    def get_df(self):
        ds = self.get_dataset(DatasetName.QEMU_STATS_BB_HIT)
        return ds.agg_df.copy()

    def generate(self, fm, cell):
        df = self.get_df()
        if not check_multi_index_aligned(df, ["dataset_id", "dataset_gid"]):
            self.logger.error("Unaligned index, skipping plot")
            return
        # Make normalized fields a percentage
        norm_col_idx = df.columns.get_level_values("delta").str.startswith("norm_")
        norm_cols = df.columns[norm_col_idx]
        df[norm_cols] = df[norm_cols] * 100
        df["sort_metric"] = self._get_sort_metric(df)

        # Remap the dataset_id according to the legend to make it
        # more meaningful for visualization
        legend_info = self.get_legend_info()
        view_df = legend_info.map_labels_to_level(df, "dataset_id", axis=0)
        # Pivot the dataset_id level into the columns
        view_df = pivot_multi_index_level(view_df, "dataset_id")

        show_cols = self._get_show_columns(view_df, legend_info)
        sort_cols = self._get_sort_columns(view_df, "sort_metric")
        view_df = view_df.sort_values(by=sort_cols, ascending=False)

        pivot_legend_info = self._get_pivot_legend_info(view_df, legend_info)
        view = TableDataView(view_df, columns=show_cols)
        view.legend_info = pivot_legend_info
        view.legend_axis = "column"
        cell.add_view(view)


class QEMUSortedSubPlotBase(BenchmarkSubPlot):
    """
    Base class for any plot that requires sorting
    """
    def get_df(self):
        raise NotImplementedError("Must override")

    def get_legend_info(self):
        return self.build_legend_by_gid()

    def get_show_columns(self, view_df, legend_info):
        # For each data column, we show all the aggregation metrics for the
        # "sample" delta level key.
        # The remaining delta level keys are shown only for non-baseline columns.
        baseline = legend_info.labels[self.benchmark.g_uuid][0]
        data = (view_df.columns.get_level_values("metric") != "sort_metric")
        sample = data & (view_df.columns.get_level_values("delta") == "sample")
        delta = (~sample) & (view_df.columns.get_level_values("dataset_gid") != baseline)
        select_cols = view_df.columns[sample]
        select_cols = select_cols.append(view_df.columns[delta])
        sorted_cols, _ = select_cols.sortlevel()
        return sorted_cols

    def compute_sort_metric(self, df, sort_col):
        """Generate the sort_metic column, represending the row sorting order"""
        # sort_col = ("instr_count", "median", "delta_baseline")
        total_delta = df[sort_col].abs().sum()
        df["sort_metric"] = (df[sort_col] / total_delta).abs()

    def get_sort_columns(self, df):
        """
        Get the final columns to sort by in the pivoted dataframe. This is required
        because we pivot an index level into the columns, so the sort_metric is also
        affected.
        Take all columns levels under "sort_metric"
        """
        sel = (slice(None), ) * (df.columns.nlevels - 1)
        indexes = df.columns.get_locs(("sort_metric", ) + sel)
        return df.columns[indexes].to_list()


class QEMUBBICountTableBase(QEMUSortedSubPlotBase):
    """
    Base class for ICount tables
    """
    def get_legend_info(self):
        legend = super().get_legend_info()
        legend.remap_colors("Greys", color_range=(0, 0.5))
        return legend

    def get_pivot_legend_info(self, df, legend_info):
        """
        Generate a legend map for the pivoted view_df.
        This will map <column index tuple> => (<dataset_gid label>, <color>)
        """
        col_df = df.columns.to_frame()
        by_label = legend_info.with_label_index()
        _, pivot_colors = col_df.align(by_label["colors"], axis=0, level="dataset_gid")
        _, pivot_labels = col_df.align(by_label["labels"], axis=0, level="dataset_gid")
        new_map = LegendInfo.from_index(df.columns, colors=pivot_colors, labels=pivot_labels)
        return new_map

    def generate(self, fm, cell):
        df = self.get_df()
        if not check_multi_index_aligned(df, ["dataset_gid"]):
            self.logger.error("Unaligned index, skipping plot")
            return
        sort_col = ("instr_count", "median", "delta_baseline")
        self.compute_sort_metric(df, sort_col)

        legend_info = self.get_legend_info()
        view_df = legend_info.map_labels_to_level(df, "dataset_gid", axis=0)
        view_df = pivot_multi_index_level(view_df, "dataset_gid")

        show_cols = self.get_show_columns(view_df, legend_info)
        sort_cols = self.get_sort_columns(view_df)
        view_df = view_df.sort_values(by=sort_cols, ascending=False)

        view = TableDataView(view_df, columns=show_cols)
        view.legend_info = self.get_pivot_legend_info(view_df, legend_info)
        view.legend_axis = "column"
        cell.add_view(view)


class QEMUBBFnTable(QEMUBBICountTableBase):
    description = "QEMU icount per function"

    def get_df(self):
        ds = self.get_dataset(DatasetName.QEMU_STATS_BB_ICOUNT)
        df = ds.get_icount_per_function()
        return df.droplevel("dataset_id")


class QEMUBBCtxFnTable(QEMUBBICountTableBase):
    description = "QEMU icount per function by context"

    def get_df(self):
        ds = self.get_dataset(DatasetName.QEMU_STATS_BB_ICOUNT)
        return ds.agg_df.droplevel("dataset_id")


class QEMUBBTable(BenchmarkTable):
    require = {DatasetName.QEMU_STATS_BB_HIT}
    name = "qemu-bb-hit-tables"
    description = "QEMU Basic-Block hit tables"
    subplots = [QEMUBBHitTable]


class QEMUFnIcountTable(BenchmarkTable):
    require = {DatasetName.QEMU_STATS_BB_ICOUNT}
    name = "qemu-bb-icount-tables"
    description = "QEMU per-function icount tables"
    subplots = [QEMUBBFnTable, QEMUBBCtxFnTable]


class QEMUIcountFnPlot(QEMUSortedSubPlotBase):
    """
    Generate a scatter plot of the delta in per-function instruciton count
    w.r.t. the baseline forthe top 20 functions.
    """
    description = "QEMU icount per function"

    def get_df(self):
        ds = self.get_dataset(DatasetName.QEMU_STATS_BB_ICOUNT)
        df = ds.get_icount_per_function()
        # Remove baseline data as we will only show deltas.
        baseline = (df.index.get_level_values("dataset_gid") == self.benchmark.g_uuid)
        df = df[~baseline]
        return df.droplevel("dataset_id")

    def get_legend_info(self):
        legend = super().get_legend_info()
        return legend.assign_colors_hsv("dataset_gid", h=(0.2, 1), s=(0.7, 0.7), v=(0.6, 0.9))

    def generate(self, fm, cell):
        df = self.get_df()
        if not check_multi_index_aligned(df, ["dataset_gid"]):
            self.logger.error("Unaligned index, skipping plot")
            return
        show_col = ("instr_count", "median", "delta_baseline")
        sort_col = ("instr_count", "median", "delta_baseline")
        self.compute_sort_metric(df, sort_col)

        legend_info = self.get_legend_info()
        sort_cols = self.get_sort_columns(df)
        view_df = df.sort_values(by=sort_cols, ascending=False)
        view_df = view_df.iloc[:40]

        view = ScatterPlotDataView(view_df, x="symbol", yleft=show_col)
        view.style = Style(marker_width=10)
        view.legend_info = legend_info
        view.legend_level = ["dataset_gid"]
        view.group_by = ["dataset_gid"]
        cell.add_view(view)
        cell.x_config.label = "Function"
        cell.x_config.tick_rotation = 90
        cell.x_config.tick_labels = view_df.index.get_level_values("symbol")
        cell.yleft_config.label = f"{Symbols.DELTA} Instructions"
        cell.yleft_config.scale = Scale("symlog", base=10)


class QEMUICountPlot(BenchmarkPlot):
    require = {DatasetName.QEMU_STATS_BB_ICOUNT}
    name = "qemu-bb-icount-plot"
    description = "QEMU per-function icount plots"
    subplots = [QEMUIcountFnPlot]
