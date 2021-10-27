import numpy as np
import pandas as pd

from ..core.dataset import (DatasetID, subset_xs, check_multi_index_aligned, rotate_multi_index_level)
from ..core.plot import (CellData, DataView, BenchmarkPlot, BenchmarkSubPlot, Surface)
from ..core.html import HTMLSurface
from ..core.matplotlib import MatplotlibSurface


class QEMUHistSubPlot(BenchmarkSubPlot):
    """
    Helpers for plots using the QEMU stat histogram dataset.
    The mixin attaches to BenchmarkSubplots.
    """
    def get_legend_map(self):
        legend = {
            uuid: str(bench.instance_config.kernelabi)
            for uuid, bench in self.benchmark.merged_benchmarks.items()
        }
        legend[self.benchmark.uuid] = f"{self.benchmark.instance_config.kernelabi}(baseline)"
        return legend

    def get_all_stats_df(self) -> pd.DataFrame:
        """
        Join the stats datasets on the index
        """
        bb_df = self.get_dataset(DatasetID.QEMU_STATS_BB_HIST).agg_df
        call_df = self.get_dataset(DatasetID.QEMU_STATS_CALL_HIST).agg_df
        return bb_df.join(call_df, how="inner", lsuffix="", rsuffix="_call")

    def _get_group_levels(self, indf: pd.DataFrame):
        assert "__dataset_id" in indf.index.names, "Missing __dataset_id index level"
        assert "process" in indf.index.names, "Missing process index level"
        assert "EL" in indf.index.names, "Missing EL index level"
        assert "file" in indf.index.names, "Missing file index level"
        assert "symbol" in indf.index.names, "Missing symbol index level"
        return ["process", "EL", "file", "symbol"]

    def filter_by_common_symbols(self, indf):
        """
        Return a dataframe containing the cross section of the input dataframe
        containing the symbols that are common to all runs (i.e. across all __dataset_id values).
        We consider valid common symbols those for which we were able to resolve the (file, sym_name)
        pair and have sensible BB count and call_count values.
        Care must be taken to keep the multi-index levels aligned.
        """
        group_levels = self._get_group_levels(indf)
        # Isolate the file:symbol pairs for each symbol marked valid in all datasets.
        # Since the filter is the same for all datasets, the cross-section will stay aligned.
        valid = (indf["valid_symbol"] == "ok") & (indf["bb_count"] != 0)
        valid_syms = valid.groupby(group_levels).all()
        return subset_xs(indf, valid_syms)

        assert "file" in indf.index.names, "Missing file index level"
        assert "symbol" in indf.index.names, "Missing symbol index level"
        # Isolate the file:symbol pairs for each symbol marked valid in all datasets.
        # Since the filter is the same for all datasets, the cross-section will stay aligned.
        valid = (indf["valid_symbol"] == "ok") & (indf["bb_count"] != 0)
        valid_syms = valid.groupby(["file", "symbol"]).all()
        return subset_xs(indf, valid_syms)

    def filter_by_noncommon_symbols(self, indf):
        """
        This is complementary to filter_by_common_symbols().
        """
        group_levels = self._get_group_levels(indf)
        # Isolate the file:symbol pairs for each symbol marked valid in at least one dataset,
        # but not all datasets.
        valid = indf["valid_symbol"] == "ok"
        # bb_count is valid if:
        # symbol is valid and bb_count != 0
        # symbol is invalid and bb_count == 0
        bb_count_ok = (indf["bb_count"] == 0) ^ valid
        # Here we only select symbols that have no issues in the bb_count column
        all_bb_count_ok = bb_count_ok.groupby(group_levels).all()
        all_valid = valid.groupby(group_levels).all()
        some_valid = valid.groupby(group_levels).any()
        unique_syms = all_bb_count_ok & some_valid & ~all_valid
        return subset_xs(indf, unique_syms)

    def filter_by_inconsistent_symbols(self, indf):
        """
        Return a cross-section of the input dataframe containing inconsistent
        records, for which we have BB hits but are marked invalid
        """
        group_levels = self._get_group_levels(indf)
        invalid = (indf["valid_symbol"] != "ok") & (indf["bb_count"] != 0)
        invalid_syms = invalid.groupby(group_levels).all()
        return subset_xs(indf, invalid_syms)


class QEMUHistTable(QEMUHistSubPlot):
    """
    Common base class for stat histogram filtered plots
    XXX Make the QEMUHistSubPlotMixin the base class directly...
    """
    @classmethod
    def get_required_datasets(cls):
        dsets = super().get_required_datasets()
        dsets += [DatasetID.QEMU_STATS_BB_HIST, DatasetID.QEMU_STATS_CALL_HIST]
        return dsets

    def _get_filtered_df(self):
        return self.get_all_stats_df()

    def _get_common_display_columns(self):
        """Columns to display for all benchmark runs"""
        return ["bb_count", "call_count", "start", "start_call", "valid_symbol"]

    def _get_non_baseline_display_columns(self):
        """Columns to display for benchmarks that are not baseline (because they are meaningless)"""
        return ["delta_bb_count", "norm_delta_bb_count", "delta_call_count", "norm_delta_call_count"]

    def _remap_display_columns(self, colmap: pd.DataFrame):
        """
        Remap original column names to the data view frame that has rotated the __dataset_id level.
        `colmap` is a dataframe in the format of the column mapping from `rotate_multi_index_level()`
        """
        common_cols = self._get_common_display_columns()
        rel_cols = self._get_non_baseline_display_columns()
        baseline = self.benchmark.uuid
        show_cols = np.append(colmap.loc[:, common_cols].values.T.ravel(),
                              colmap.loc[colmap.index != baseline, rel_cols].values.T.ravel())
        return show_cols

    def _get_sort_metric(self, df):
        # We should always compute a sort metric index in the dataset instead (post_agg)
        relevance = df["delta_call_count"]
        return relevance

    def generate(self, surface: Surface, cell: CellData):
        df = self._get_filtered_df()
        if not check_multi_index_aligned(df, "__dataset_id"):
            self.logger.error("Unaligned index, skipping plot!")
            return
        # Make normalized fields a percentage
        norm_cols = [col for col in df.columns if col.startswith("norm_")]
        df[norm_cols] = df[norm_cols] * 100
        df["sort_metric"] = self._get_sort_metric(df)

        legend_map = self.get_legend_map()
        # Remap columns such that for each index entry in the __dataset_id level,
        # we create a new set of columns with the name remapped as defined by legend_map
        # so instead of having a linear table we can compare benchmark values side-by-side
        # Note: It is essential that the dataframe is aligned on the index level for this.
        view_df, colmap = rotate_multi_index_level(df, "__dataset_id", legend_map)
        show_cols = self._remap_display_columns(colmap)
        # sort_cols = colmap.loc[:, "sort_metric"]
        # view_df = view_df.sort_values(sort_cols)
        # Proper hex formatting
        col_formatter = {
            col: lambda v: f"0x{int(v):x}" if not np.isnan(v) else "?"
            for col in colmap[["start", "start_call"]].values.ravel()
        }
        view = surface.make_view("table", df=view_df, yleft=show_cols, fmt=col_formatter)
        cell.add_view(view)


class QEMUAllSymbolHistTable(QEMUHistTable):
    """
    Plot a table of QEMU histogram statistics showing only the values corresponding to symbols
    being hit in all benchmark samples.
    """
    def get_cell_title(self):
        return "QEMU Raw Stats Table"


class QEMUCommonSymbolHistTable(QEMUHistTable):
    """
    Plot a table with QEMU histogram statistics restricted to symbols hit in all
    benchmark runs.
    """
    def get_cell_title(self):
        return "QEMU Common Symbol Stats Table"

    def _get_filtered_df(self):
        df = super()._get_filtered_df()
        return self.filter_by_common_symbols(df)


class QEMUExtraSymbolHistTable(QEMUHistTable):
    """
    Plot a table with QEMU histogram statistics restricted to symbols hit in all
    benchmark runs.
    """
    def get_cell_title(self):
        return "QEMU Extra Symbol Stats Table"

    def _get_filtered_df(self):
        df = super()._get_filtered_df()
        return self.filter_by_noncommon_symbols(df)


class QEMUStrangeSymbolHistTable(QEMUHistTable):
    """
    Plot a table with QEMU histogram statistics restricted to inconsistent
    records, mainly for debugging purposes.
    """
    def get_cell_title(self):
        return "QEMU Inconsistent Symbol Stats Table"

    def _get_filtered_df(self):
        df = super()._get_filtered_df()
        return self.filter_by_inconsistent_symbols(df)


class QEMUTables(BenchmarkPlot):
    """
    Show QEMU datasets as tabular output for inspection.
    """
    subplots = [
        QEMUAllSymbolHistTable,
        QEMUCommonSymbolHistTable,
        QEMUExtraSymbolHistTable,
        QEMUStrangeSymbolHistTable,
    ]

    def __init__(self, benchmark):
        super().__init__(benchmark, [HTMLSurface()])  #TODO add ExcelSurface()

    def get_plot_name(self):
        return "QEMU Stats Tables"

    def get_plot_file(self):
        return self.benchmark.manager.session_output_path / "qemu_stats_tables"
