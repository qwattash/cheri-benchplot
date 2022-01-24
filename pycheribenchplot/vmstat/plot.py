import numpy as np
import pandas as pd

from ..core.dataset import (DatasetName, check_multi_index_aligned, pivot_multi_index_level, subset_xs)
from ..core.excel import SpreadsheetSurface
from ..core.html import HTMLSurface
from ..core.plot import (BenchmarkPlot, BenchmarkSubPlot, CellData, ColorMap, Surface, TableDataView)


class VMStatTable(BenchmarkSubPlot):
    """
    Base class for vmstat tables
    """
    def get_legend_map(self):
        legend = {uuid: str(bench.instance_config.name) for uuid, bench in self.benchmark.merged_benchmarks.items()}
        legend[self.benchmark.uuid] = f"{self.benchmark.instance_config.name}(*)"
        legend_map = ColorMap.from_keys(legend.keys(), mapname="Greys", labels=legend.values(), color_range=(0, 0.5))
        return legend_map

    def _get_show_columns(self, view_df, legend_map):
        """
        For each data column, we show all the aggregation metrics for the
        "sample" delta level key.
        The remaining delta level keys are shown only for non-baseline columns.
        XXX this can be generalized in a common subplot type for tables.
        """
        baseline = legend_map.get_label(self.benchmark.uuid)
        sample = (view_df.columns.get_level_values("delta") == "sample")
        delta = (~sample) & (view_df.columns.get_level_values("__dataset_id") != baseline)
        select_cols = view_df.columns[sample]
        select_cols = select_cols.append(view_df.columns[delta])
        sorted_cols, _ = select_cols.sortlevel()
        return sorted_cols

    def _get_pivot_legend_map(self, df, legend_map):
        """
        Generate the legend map for the pivoted view_df and map the
        dataset column index level to the legend labels.
        XXX this can be generalized in a common subplot type for tables.
        Assume that the __dataset_id level has been already mapped to label values
        """
        col_df = df.columns.to_frame()
        by_label = legend_map.with_label_index()
        _, pivot_colors = col_df.align(by_label["colors"], axis=0, level="__dataset_id")
        _, pivot_labels = col_df.align(by_label["labels"], axis=0, level="__dataset_id")
        new_map = ColorMap(df.columns, pivot_colors, pivot_labels)
        return new_map

    def generate(self, surface: Surface, cell: CellData):
        df = self._get_vmstat_dataset()
        if not check_multi_index_aligned(df, "__dataset_id"):
            self.logger.error("Unaligned index, skipping plot")
            return
        # Make normalized fields a percentage
        norm_col_idx = df.columns.get_level_values("delta").str.startswith("norm_")
        norm_cols = df.columns[norm_col_idx]
        df[norm_cols] = df[norm_cols] * 100

        legend_map = self.get_legend_map()
        view_df = legend_map.map_labels_to_level(df, "__dataset_id", axis=0)
        view_df = pivot_multi_index_level(view_df, "__dataset_id")

        show_cols = self._get_show_columns(view_df, legend_map)
        pivot_legend_map = self._get_pivot_legend_map(view_df, legend_map)
        assert cell.legend_map is None
        cell.legend_map = pivot_legend_map
        view = TableDataView("table", view_df, columns=show_cols)
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


class VMStatTables(BenchmarkPlot):
    """
    Show QEMU datasets as tabular output for inspection.
    """
    subplots = [
        VMStatMallocTable,
        VMStatUMATable,
    ]

    def __init__(self, benchmark):
        super().__init__(benchmark, [HTMLSurface(), SpreadsheetSurface()])

    def get_plot_name(self):
        return "VMStat Tables"

    def get_plot_file(self):
        return self.benchmark.manager.session_output_path / "vmstat_tables"
