import numpy as np
import pandas as pd

from ..core.dataset import (DatasetName, check_multi_index_aligned, rotate_multi_index_level, subset_xs)
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
        legend[self.benchmark.uuid] = f"{self.benchmark.instance_config.name}(baseline)"
        lmap = ColorMap.from_keys(legend.keys(), mapname="Greys", labels=legend.values(), color_range=(0, 0.6))
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
    """
    Common base class for stat histogram filtered plots
    XXX Make the QEMUHistSubPlotMixin the base class directly...
    """
    @classmethod
    def get_required_datasets(cls):
        dsets = super().get_required_datasets()
        dsets += [DatasetName.QEMU_STATS_BB_HIST, DatasetName.QEMU_STATS_CALL_HIST]
        return dsets

    def _get_filtered_df(self):
        return self.get_all_stats_df()

    def _get_common_display_columns(self):
        """Columns to display for all benchmark runs"""
        cols = ["icount_mean", "icount_std", "call_count_mean", "call_count_std"]
        return cols

    def _get_non_baseline_display_columns(self):
        """Columns to display for benchmarks that are not baseline (because they are meaningless)"""
        cols = ["delta_icount_mean", "norm_delta_icount_mean", "delta_call_count_mean", "norm_delta_call_count_mean"]
        return cols

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

    def _remap_legend_to_pivot_columns(self, legend_map: ColorMap, column_map: pd.DataFrame):
        """
        Remap the original dataset-id colormap to use pivoted columns as keys
        """
        remap = {}
        for level_value in column_map.index:
            level_columns = column_map.loc[level_value]
            for col in level_columns:
                remap[col] = legend_map.get_color(level_value)
        return ColorMap(remap.keys(), remap.values(), remap.keys())

    def _get_sort_metric(self, df):
        total_delta_calls = df["delta_call_count_mean"].abs().sum()
        relevance = (df["delta_call_count_mean"] / total_delta_calls).abs()
        return relevance

    def _remap_sort_columns(self, metric_col: str, colmap: pd.DataFrame):
        baseline = self.benchmark.uuid
        sortby = colmap.loc[colmap.index != baseline, metric_col].values.T.ravel()
        return list(sortby)

    def generate(self, surface: Surface, cell: CellData):
        df = self._get_filtered_df()
        if not check_multi_index_aligned(df, "__dataset_id"):
            self.logger.error("Unaligned index, skipping plot")
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
        view_df, colmap = rotate_multi_index_level(df, "__dataset_id", legend_map.label_items())
        pivot_legend = self._remap_legend_to_pivot_columns(legend_map, colmap)
        show_cols = self._remap_display_columns(colmap)
        sort_cols = self._remap_sort_columns("sort_metric", colmap)
        view_df = view_df.sort_values(by=sort_cols, ascending=False)
        view = surface.make_view("table", df=view_df, yleft=show_cols)
        cell.add_view(view)

        assert cell.legend_map is None
        cell.legend_map = pivot_legend
        cell.legend_level = None


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
