import numpy as np
import pandas as pd

from ..core.dataset import (DatasetName, check_multi_index_aligned, pivot_multi_index_level)
from ..core.plot import (BenchmarkSubPlot, BenchmarkTable, CellData, LegendInfo, Mosaic, TableDataView)


class QEMUHistSubPlot(BenchmarkSubPlot):
    """
    Helpers for plots using the QEMU stat histogram dataset.
    The mixin attaches to BenchmarkSubplots.
    """
    def get_legend_info(self) -> LegendInfo:
        legend = self.build_legend_by_dataset()
        legend.remap_colors("Greys", color_range=(0, 0.5))
        return legend

    def get_all_stats_df(self) -> pd.DataFrame:
        """
        Join the stats datasets on the index
        """
        bb_df = self.get_dataset(DatasetName.QEMU_STATS_BB_HIT).agg_df
        call_df = self.get_dataset(DatasetName.QEMU_STATS_CALL_HIT).agg_df
        return bb_df.join(call_df, how="outer", lsuffix="", rsuffix="_call")

    def _get_group_levels(self, indf: pd.DataFrame):
        # This should be in sync with the dataset aggregation index
        assert "dataset_id" in indf.index.names, "Missing dataset_id index level"
        assert "process" in indf.index.names, "Missing process index level"
        assert "thread" in indf.index.names, "Missing thread index level"
        assert "EL" in indf.index.names, "Missing EL index level"
        assert "file" in indf.index.names, "Missing file index level"
        assert "symbol" in indf.index.names, "Missing symbol index level"
        return ["process", "thread", "EL", "file", "symbol"]


class QEMUHistTable(QEMUHistSubPlot):
    """Common base class for stat histogram filtered plots"""
    def _get_filtered_df(self):
        return self.get_all_stats_df()

    def _get_sort_metric(self, df):
        sort_col = ("call_count", "mean", "delta_baseline")
        total_delta_calls = df[sort_col].abs().sum()
        relevance = (df[sort_col] / total_delta_calls).abs()
        return relevance

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
        new_map = LegendInfo(df.columns, colors=pivot_colors, labels=pivot_labels)
        return new_map

    def generate(self, surface, cell):
        df = self._get_filtered_df()
        if not check_multi_index_aligned(df, "dataset_id"):
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
        cell.add_view(view)


class QEMUContextHistTable(QEMUHistTable):
    def __init__(self, plot, context_procname: str):
        super().__init__(plot)
        self.process = context_procname

    def get_cell_title(self):
        proc = self.process
        return f"QEMU PC stats for {proc}"

    def _get_filtered_df(self):
        df = self.get_all_stats_df()
        df = df.xs(self.process, level="process")
        return df


class QEMUTables(BenchmarkTable):
    """
    Show QEMU datasets as tabular output for inspection.
    """
    require = {DatasetName.QEMU_STATS_BB_HIT, DatasetName.QEMU_STATS_CALL_HIT}
    name = "qemu-stats-tables"
    subplots = [
        QEMUContextHistTable,
    ]

    def _make_subplots_mosaic(self):
        """
        Dynamically create subplots for each context ID in a mosaic column
        """
        subplots = {}
        layout = []
        contexts = set()
        bb_df = self.benchmark.get_dataset(DatasetName.QEMU_STATS_BB_HIT).agg_df
        contexts.update(bb_df.index.get_level_values("process").unique())
        call_df = self.benchmark.get_dataset(DatasetName.QEMU_STATS_CALL_HIT).agg_df
        contexts.update(call_df.index.get_level_values("process").unique())
        for idx, ctx in enumerate(contexts):
            name = f"subplot-{idx}"
            subplots[name] = QEMUContextHistTable(self, ctx)
            layout.append([name])
        return Mosaic(layout, subplots)

    def get_plot_name(self):
        return "QEMU Stats Tables"


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


class QEMUBBFnTable(BenchmarkSubPlot):
    description = "QEMU icount per function"

    def get_df(self):
        ds = self.get_dataset(DatasetName.QEMU_STATS_BB_ICOUNT)
        return ds.agg_df.droplevel("dataset_id")

    def get_legend_info(self):
        legend = self.build_legend_by_gid()
        legend.remap_colors("Greys", color_range=(0, 0.5))
        return legend

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

    def compute_sort_metric(self, df):
        """Generate the sort_metic column, represending the row sorting order"""
        sort_col = ("instr_count", "median", "delta_baseline")
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

    def generate(self, fm, cell):
        df = self.get_df()
        if not check_multi_index_aligned(df, ["dataset_gid"]):
            self.logger.error("Unaligned index, skipping plot")
            return
        self.compute_sort_metric(df)

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


class QEMUBBTable(BenchmarkTable):
    require = {DatasetName.QEMU_STATS_BB_HIT}
    name = "qemu-bb-hit-tables"
    description = "QEMU Basic-Block hit tables"
    subplots = [QEMUBBHitTable]


class QEMUFnIcountTable(BenchmarkTable):
    require = {DatasetName.QEMU_STATS_BB_ICOUNT}
    name = "qemu-bb-icount-tables"
    description = "QEMU icount tables"
    subplots = [QEMUBBFnTable]
