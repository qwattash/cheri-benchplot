import numpy as np
import pandas as pd

from ..core.dataset import (DatasetName, check_multi_index_aligned, pivot_multi_index_level, subset_xs)
from ..core.excel import SpreadsheetSurface
from ..core.html import HTMLSurface
from ..core.plot import (BenchmarkPlot, BenchmarkSubPlot, CellData, ColorMap, DataView, Surface)


class QEMUHistSubPlot(BenchmarkSubPlot):
    """
    Helpers for plots using the QEMU stat histogram dataset.
    The mixin attaches to BenchmarkSubplots.
    """
    def get_legend_map(self) -> ColorMap:
        legend = {uuid: str(bench.instance_config.name) for uuid, bench in self.benchmark.merged_benchmarks.items()}
        legend[self.benchmark.uuid] = f"{self.benchmark.instance_config.name}(*)"
        lmap = ColorMap.from_keys(legend.keys(), mapname="Greys", labels=legend.values(), color_range=(0, 0.5))
        return lmap

    def get_all_stats_df(self) -> pd.DataFrame:
        """
        Join the stats datasets on the index
        """
        bb_df = self.get_dataset(DatasetName.QEMU_STATS_BB_HIST).agg_df
        call_df = self.get_dataset(DatasetName.QEMU_STATS_CALL_HIST).agg_df
        return bb_df.join(call_df, how="outer", lsuffix="", rsuffix="_call")

    def _get_group_levels(self, indf: pd.DataFrame):
        # This should be in sync with the dataset aggregation index
        assert "__dataset_id" in indf.index.names, "Missing __dataset_id index level"
        assert "process" in indf.index.names, "Missing process index level"
        assert "thread" in indf.index.names, "Missing thread index level"
        assert "EL" in indf.index.names, "Missing EL index level"
        assert "file" in indf.index.names, "Missing file index level"
        assert "symbol" in indf.index.names, "Missing symbol index level"
        return ["process", "thread", "EL", "file", "symbol"]


class QEMUHistTable(QEMUHistSubPlot):
    """Common base class for stat histogram filtered plots"""
    @classmethod
    def get_required_datasets(cls):
        dsets = super().get_required_datasets()
        dsets += [DatasetName.QEMU_STATS_BB_HIST, DatasetName.QEMU_STATS_CALL_HIST]
        return dsets

    def _get_filtered_df(self):
        return self.get_all_stats_df()

    def _get_sort_metric(self, df):
        sort_col = ("call_count", "mean", "delta_baseline")
        total_delta_calls = df[sort_col].abs().sum()
        relevance = (df[sort_col] / total_delta_calls).abs()
        return relevance

    def _get_show_columns(self, view_df, legend_map):
        # For each data column, we show all the aggregation metrics for the
        # "sample" delta level key.
        # The remaining delta level keys are shown only for non-baseline columns.
        baseline = legend_map.get_label(self.benchmark.uuid)
        data = (view_df.columns.get_level_values("metric") != "sort_metric")
        sample = data & (view_df.columns.get_level_values("delta") == "sample")
        delta = data & (~sample) & (view_df.columns.get_level_values("__dataset_id") != baseline)
        select_cols = view_df.columns[sample]
        select_cols = select_cols.append(view_df.columns[delta])
        sorted_cols, _ = select_cols.sortlevel()
        return sorted_cols

    def _get_sort_columns(self, view_df, sort_col):
        sel = (slice(None), ) * (view_df.columns.nlevels - 1)
        indexes = view_df.columns.get_locs((sort_col, ) + sel)
        return view_df.columns[indexes].to_list()

    def _get_pivot_legend_map(self, df, legend_map):
        """
        Generate a legend map for the pivoted view_df.
        This will map <column index tuple> => (<__dataset_id label>, <color>)
        """
        col_df = df.columns.to_frame()
        by_label = legend_map.with_label_index()
        _, pivot_colors = col_df.align(by_label["colors"], axis=0, level="__dataset_id")
        _, pivot_labels = col_df.align(by_label["labels"], axis=0, level="__dataset_id")
        new_map = ColorMap(df.columns, pivot_colors, pivot_labels)
        return new_map

    def generate(self, surface: Surface, cell: CellData):
        df = self._get_filtered_df()
        if not check_multi_index_aligned(df, "__dataset_id"):
            self.logger.error("Unaligned index, skipping plot")
            return
        # Make normalized fields a percentage
        norm_col_idx = df.columns.get_level_values("delta").str.startswith("norm_")
        norm_cols = df.columns[norm_col_idx]
        df[norm_cols] = df[norm_cols] * 100
        df["sort_metric"] = self._get_sort_metric(df)

        # Remap the __dataset_id according to the legend to make it
        # more meaningful for visualization
        legend_map = self.get_legend_map()
        view_df = legend_map.map_labels_to_level(df, "__dataset_id", axis=0)
        # Pivot the __dataset_id level into the columns
        view_df = pivot_multi_index_level(view_df, "__dataset_id")

        show_cols = self._get_show_columns(view_df, legend_map)
        sort_cols = self._get_sort_columns(view_df, "sort_metric")
        view_df = view_df.sort_values(by=sort_cols, ascending=False)

        pivot_legend_map = self._get_pivot_legend_map(view_df, legend_map)
        assert cell.legend_map is None
        cell.legend_map = pivot_legend_map
        view = surface.make_view("table", df=view_df, yleft=show_cols)
        cell.add_view(view)


class QEMUContextHistTable(QEMUHistTable):
    def __init__(self, plot: BenchmarkPlot, context_procname: str):
        super().__init__(plot)
        self.process = context_procname

    def get_cell_title(self):
        proc = self.process
        return f"QEMU PC stats for {proc}"

    def _get_filtered_df(self):
        df = self.get_all_stats_df()
        df = df.xs(self.process, level="process")
        return df


class QEMUTables(BenchmarkPlot):
    """
    Show QEMU datasets as tabular output for inspection.
    """
    subplots = [
        QEMUContextHistTable,
    ]

    @classmethod
    def check_required_datasets(cls, dsets: list[DatasetName]):
        """
        Check dataset list against qemu stats dataset names
        """
        required = set([DatasetName.QEMU_STATS_BB_HIST, DatasetName.QEMU_STATS_CALL_HIST])
        return required.issubset(set(dsets))

    def __init__(self, benchmark):
        super().__init__(benchmark, [HTMLSurface(), SpreadsheetSurface()])

    def _make_subplots(self):
        """
        Dynamically create subplots for each context ID
        """
        subplots = []
        contexts = set()
        bb_df = self.benchmark.get_dataset(DatasetName.QEMU_STATS_BB_HIST).agg_df
        contexts.update(bb_df.index.get_level_values("process").unique())
        call_df = self.benchmark.get_dataset(DatasetName.QEMU_STATS_CALL_HIST).agg_df
        contexts.update(call_df.index.get_level_values("process").unique())
        for ctx in contexts:
            subplots.append(QEMUContextHistTable(self, ctx))
        return subplots

    def get_plot_name(self):
        return "QEMU Stats Tables"

    def get_plot_file(self):
        return self.benchmark.manager.session_output_path / "qemu_stats_tables"
