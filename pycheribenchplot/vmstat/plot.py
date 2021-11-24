import numpy as np
import pandas as pd

from ..core.dataset import (DatasetID, subset_xs, check_multi_index_aligned, rotate_multi_index_level)
from ..core.plot import (CellData, DataView, BenchmarkPlot, BenchmarkSubPlot, Surface)
from ..core.html import HTMLSurface
from ..core.excel import SpreadsheetSurface


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
        dsets += [DatasetID.VMSTAT_MALLOC]
        return dsets

    def get_cell_title(self):
        return "Kernel malloc stats"

    def _get_common_display_columns(self):
        """Columns to display for all benchmark runs"""
        return ["requests", "large-malloc"]

    def _get_non_baseline_display_columns(self):
        """Columns to display for benchmarks that are not baseline (because they are meaningless)"""
        return ["delta_requests", "norm_delta_requests", "delta_large-malloc", "norm_delta_large-malloc"]

    def _get_vmstat_dataset(self):
        return self.get_dataset(DatasetID.VMSTAT_MALLOC).agg_df.copy()


class VMStatUMATable(VMStatTable):
    """
    Export a table with the vmstat UMA data.
    """
    @classmethod
    def get_required_datasets(cls):
        dsets = super().get_required_datasets()
        dsets += [DatasetID.VMSTAT_UMA]
        return dsets

    def __init__(self, plot):
        super().__init__(plot)
        # Optional zone info dataset
        self.uma_stats = self.get_dataset(DatasetID.VMSTAT_UMA)
        self.uma_zone_info = self.get_dataset(DatasetID.VMSTAT_UMA_INFO)

    def get_cell_title(self):
        return "Kernel UMA stats"

    def _get_common_display_columns(self):
        """Columns to display for all benchmark runs"""
        stats_cols = self.uma_stats.data_columns()
        if self.uma_zone_info:
            stats_cols += self.uma_zone_info.data_columns()
        return stats_cols

    def _get_non_baseline_display_columns(self):
        """Columns to display for benchmarks that are not baseline (because they are meaningless)"""
        delta_cols = [f"delta_{c}" for c in self.uma_stats.data_columns()]
        if self.uma_zone_info:
            delta_cols += [f"delta_{c}" for c in self.uma_zone_info.data_columns()]
        norm_cols = [f"norm_{c}" for c in delta_cols]
        return delta_cols + norm_cols

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
