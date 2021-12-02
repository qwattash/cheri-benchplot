import numpy as np
import pandas as pd

from ..core.dataset import (DatasetName, check_multi_index_aligned,
                            rotate_multi_index_level, subset_xs)
from ..core.excel import SpreadsheetSurface
from ..core.html import HTMLSurface
from ..core.plot import (BenchmarkPlot, BenchmarkSubPlot, CellData, DataView,
                         Surface)


class VMStatTable(BenchmarkSubPlot):
    """
    Base class for vmstat tables
    """
    def get_legend_map(self):
        legend = {uuid: str(bench.instance_config.name) for uuid, bench in self.benchmark.merged_benchmarks.items()}
        legend[self.benchmark.uuid] = f"{self.benchmark.instance_config.name}(baseline)"
        return legend

    def _remap_display_columns(self, colmap: pd.DataFrame):
        """
        Remap original column names to the data view frame that has rotated the __dataset_id level.
        `colmap` is a dataframe in the format of the column mapping from `rotate_multi_index_level()`
        """
        common_cols = self._get_common_display_columns()
        rel_cols = self._get_non_baseline_display_columns()
        baseline = self.benchmark.uuid
        show_cols = np.append(colmap.loc[:, common_cols].values.T.ravel(), colmap.loc[colmap.index != baseline,
                                                                                      rel_cols].values.T.ravel())
        return show_cols

    def generate(self, surface: Surface, cell: CellData):
        df = self._get_vmstat_dataset()
        if not check_multi_index_aligned(df, "__dataset_id"):
            self.logger.error("Unaligned index, skipping plot")
            return
        # Make normalized fields a percentage
        norm_cols = [col for col in df.columns if col.startswith("norm_")]
        df[norm_cols] = df[norm_cols] * 100
        legend_map = self.get_legend_map()
        view_df, colmap = rotate_multi_index_level(df, "__dataset_id", legend_map)
        show_cols = self._remap_display_columns(colmap)
        view = surface.make_view("table", df=view_df, yleft=show_cols)
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
