import numpy as np
import pandas as pd

from ..core.dataset import (DatasetName, check_multi_index_aligned, pivot_multi_index_level)
from ..core.plot import (AALineDataView, BarPlotDataView, BenchmarkPlot, BenchmarkSubPlot, BenchmarkTable, CellData,
                         LegendInfo, Symbols, TableDataView)


class VMStatTable(BenchmarkSubPlot):
    """
    Base class for vmstat tables
    """
    def get_legend_info(self):
        legend = self.build_legend_by_dataset()
        legend.remap_colors("Greys", color_range=(0, 0.5))
        return legend

    def _get_show_columns(self, view_df, legend_info):
        """
        For each data column, we show all the aggregation metrics for the
        "sample" delta level key.
        The remaining delta level keys are shown only for non-baseline columns.
        XXX this can be generalized in a common subplot type for tables.
        """
        baseline = legend_info.label(self.benchmark.uuid)
        sample = (view_df.columns.get_level_values("delta") == "sample")
        delta = (~sample) & (view_df.columns.get_level_values("dataset_id") != baseline)
        select_cols = view_df.columns[sample]
        select_cols = select_cols.append(view_df.columns[delta])
        sorted_cols, _ = select_cols.sortlevel()
        return sorted_cols

    def _get_pivot_legend_info(self, df, legend_info):
        """
        Generate the legend map for the pivoted view_df and map the
        dataset column index level to the legend labels.
        XXX this can be generalized in a common subplot type for tables.
        Assume that the dataset_id level has been already mapped to label values
        """
        col_df = df.columns.to_frame()
        by_label = legend_info.with_label_index()
        _, pivot_colors = col_df.align(by_label["colors"], axis=0, level="dataset_id")
        _, pivot_labels = col_df.align(by_label["labels"], axis=0, level="dataset_id")
        new_map = LegendInfo(df.columns, colors=pivot_colors, labels=pivot_labels)
        return new_map

    def generate(self, surface: "Surface", cell: CellData):
        df = self._get_vmstat_dataset()
        if not check_multi_index_aligned(df, "dataset_id"):
            self.logger.error("Unaligned index, skipping plot")
            return
        # Make normalized fields a percentage
        norm_col_idx = df.columns.get_level_values("delta").str.startswith("norm_")
        norm_cols = df.columns[norm_col_idx]
        df[norm_cols] = df[norm_cols] * 100

        legend_info = self.get_legend_info()
        view_df = legend_info.map_labels_to_level(df, "dataset_id", axis=0)
        view_df = pivot_multi_index_level(view_df, "dataset_id")

        show_cols = self._get_show_columns(view_df, legend_info)
        pivot_legend_info = self._get_pivot_legend_info(view_df, legend_info)
        view = TableDataView(view_df, columns=show_cols)
        view.legend_info = pivot_legend_info
        cell.add_view(view)


class VMStatMallocTable(VMStatTable):
    """
    Export a table with the vmstat malloc data for each kernel malloc zone.
    """
    @classmethod
    def get_required_datasets(cls):
        dsets = super().get_required_datasets()
        dsets += [DatasetName.VMSTAT_MALLOC]
        return dsets

    def get_cell_title(self):
        return "Kernel malloc stats"

    def _get_common_display_columns(self):
        """Columns to display for all benchmark runs"""
        return ["requests_mean", "large-malloc_mean", "requests_std", "large-malloc_std"]

    def _get_non_baseline_display_columns(self):
        """Columns to display for benchmarks that are not baseline (because they are meaningless)"""
        return [
            "delta_requests_mean", "norm_delta_requests_mean", "delta_large-malloc_mean", "norm_delta_large-malloc_mean"
        ]

    def _get_vmstat_dataset(self):
        return self.get_dataset(DatasetName.VMSTAT_MALLOC).agg_df.copy()


class VMStatUMATable(VMStatTable):
    """
    Export a table with the vmstat UMA data.
    """
    @classmethod
    def get_required_datasets(cls):
        dsets = super().get_required_datasets()
        dsets += [DatasetName.VMSTAT_UMA]
        return dsets

    def __init__(self, plot):
        super().__init__(plot)
        # Optional zone info dataset
        self.uma_stats = self.get_dataset(DatasetName.VMSTAT_UMA)
        self.uma_zone_info = self.get_dataset(DatasetName.VMSTAT_UMA_INFO)

    def get_cell_title(self):
        return "Kernel UMA stats"

    def _get_common_display_columns(self):
        """Columns to display for all benchmark runs"""
        stats_cols = []
        for c in self.uma_stats.data_columns():
            stats_cols.append(f"{c}_mean")
            stats_cols.append(f"{c}_std")
        if self.uma_zone_info:
            stats_cols.extend(self.uma_zone_info.data_columns())
        return stats_cols

    def _get_non_baseline_display_columns(self):
        """Columns to display for benchmarks that are not baseline (because they are meaningless)"""
        cols = []
        for c in self._get_common_display_columns():
            cols.append(f"delta_{c}")
            cols.append(f"norm_delta_{c}")
        return cols

    def _get_vmstat_dataset(self):
        if self.uma_zone_info:
            return self.uma_stats.agg_df.join(self.uma_zone_info.agg_df, how="left")
        else:
            return self.uma_stats.agg_df.copy()


class VMStatUMABucketAffinity(BenchmarkSubPlot):
    def get_cell_title(self):
        return "UMA Zone bucket affinity groups"

    def generate(self, surface, cell):
        pass


class VMStatTables(BenchmarkTable):
    """
    Show vmstat datasets as tabular output for inspection.
    """
    subplots = [
        VMStatMallocTable,
        VMStatUMATable,
        VMStatUMABucketAffinity,
    ]

    def get_plot_name(self):
        return "VMStat Tables"

    def get_plot_file(self):
        return self.benchmark.manager.plot_output_path / "vmstat_tables"


class VMStatUMAMetricHist(BenchmarkSubPlot):
    """
    Histogram of UMA metrics across datasets
    """
    @classmethod
    def get_required_datasets(cls):
        dsets = super().get_required_datasets()
        dsets += [DatasetName.VMSTAT_UMA]
        return dsets

    def __init__(self, plot, dataset, metric: str):
        super().__init__(plot)
        # Optional zone info dataset
        self.ds = dataset
        self.metric = metric

    def get_legend_info(self):
        # Use base legend to separate by axis
        base = self.build_legend_by_dataset()
        legend_left = base.map_label(lambda l: f"{Symbols.DELTA}{self.metric} " + l)
        legend_right = base.map_label(lambda l: f"% {Symbols.DELTA}{self.metric}" + l)
        legend = LegendInfo.multi_axis(left=legend_left, right=legend_right)
        legend.remap_colors("Paired")
        return legend

    def get_bar_limit(self):
        """
        Get maximum number of bars to show
        """
        return 20

    def get_cell_title(self):
        return f"UMA {self.metric} variation w.r.t. baseline"

    def generate(self, surface, cell):
        """
        We filter metric to show only the values for the top 90th percentile
        of the delta, this avoid cluttering the plots with meaningless data.
        """
        df = self.ds.agg_df[self.ds.agg_df.index.get_level_values("dataset_id") != self.benchmark.uuid].copy()
        delta_col = (self.metric, "median", "delta_baseline")
        rel_col = (self.metric, "median", "norm_delta_baseline")
        if delta_col not in df.columns:
            delta_col = (self.metric, "-", "delta_baseline")
            rel_col = (self.metric, "-", "norm_delta_baseline")
        self.logger.debug("extract plot metric %s", self.metric)

        df["abs_delta"] = df[delta_col].abs()
        abstop = df["abs_delta"] >= df["abs_delta"].quantile(0.90)
        df_sel = df[abstop].sort_values("abs_delta", ascending=False)
        maxbar = self.get_bar_limit()
        if len(df_sel) > maxbar:
            # Just take the largest N values
            df_sel = df_sel.iloc[:maxbar + 1]
        # Avoid assigning to a slice
        df_sel = df_sel.copy()
        df_sel[rel_col] *= 100
        df_sel["x"] = range(len(df_sel))

        view = BarPlotDataView(df_sel, x="x", yleft=delta_col, yright=rel_col)
        view.legend_info = self.get_legend_info()
        view.legend_level = ["dataset_id"]
        cell.add_view(view)

        # Add a line for the y
        df_line = df[["abs_delta"]].abs().median()
        view = AALineDataView(df_line, horizontal=["abs_delta"])
        view.style.line_style = "dashed"
        view.style.line_width = 0.5
        view.legend_info = LegendInfo(["abs_delta"],
                                      labels=[f"median {Symbols.DELTA} abs({self.metric})"],
                                      cmap_name="Greys",
                                      color_range=(0.5, 1))
        view.legend_level = ["metric"]
        cell.add_view(view)

        cell.x_config.label = "UMA Zone"
        cell.x_config.ticks = df_sel["x"]
        cell.x_config.tick_labels = df_sel.index.get_level_values("name")
        cell.x_config.tick_rotation = 90
        cell.yleft_config.label = f"{Symbols.DELTA}{self.metric}"
        cell.yright_config.label = f"% {Symbols.DELTA}{self.metric}"


class VMStatDistribution(BenchmarkPlot):
    """
    Show vstat datasets distribution of interesting metrics
    """
    @classmethod
    def check_required_datasets(cls, dsets: list[DatasetName]):
        """
        Check dataset list against qemu stats dataset names
        """
        required = set([DatasetName.VMSTAT_UMA, DatasetName.VMSTAT_UMA_INFO])
        return required.issubset(set(dsets))

    def _make_subplots(self):
        subplots = []
        uma_stats = self.get_dataset(DatasetName.VMSTAT_UMA)
        for metric in uma_stats.data_columns():
            subplots.append(VMStatUMAMetricHist(self, uma_stats, metric))
        uma_info = self.get_dataset(DatasetName.VMSTAT_UMA_INFO)
        for metric in uma_info.data_columns():
            subplots.append(VMStatUMAMetricHist(self, uma_info, metric))
        return subplots

    def get_plot_name(self):
        return "VMStat metrics distribution"

    def get_plot_file(self):
        return self.benchmark.manager.plot_output_path / "vmstat-histograms"
