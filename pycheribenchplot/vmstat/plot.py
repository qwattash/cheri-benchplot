import numpy as np
import pandas as pd

from ..core.dataset import (DatasetName, assign_sorted_coord, check_multi_index_aligned, pivot_multi_index_level,
                            quantile_slice)
from ..core.plot import (AALineDataView, BarPlotDataView, BenchmarkPlot, BenchmarkSubPlot, BenchmarkTable, CellData,
                         LegendInfo, Mosaic, Symbols, TableDataView)


class VMStatTable(BenchmarkSubPlot):
    """
    Base class for vmstat tables
    """
    def get_legend_info(self):
        legend = self.build_legend_by_dataset()
        return legend.remap_colors("Greys", color_range=(0, 0.5))

    def _get_show_columns(self, view_df, legend_info):
        """
        For each data column, we show all the aggregation metrics for the
        "sample" delta level key.
        The remaining delta level keys are shown only for non-baseline columns.
        XXX this can be generalized in a common subplot type for tables.
        """
        baseline = legend_info.labels[self.benchmark.uuid]
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
        index = pd.Index(df.columns, name="column")
        return LegendInfo.from_index(index, colors=pivot_colors, labels=pivot_labels)

    def generate(self, surface, cell):
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
        view.legend_level = []  # No row coloring, only columns
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
    """
    A table showing changes in zone bucket size, grouped by bucket size.
    """
    @classmethod
    def get_required_datasets(cls):
        dsets = super().get_required_datasets()
        dsets += [DatasetName.VMSTAT_UMA_INFO]
        return dsets

    def get_cell_title(self):
        return "UMA Zone bucket affinity groups"

    def get_legend_info(self):
        legend = self.build_legend_by_dataset()
        return legend.remap_colors("Greys", color_range=(0, 0.5))

    def _get_show_columns(self, df, legend_info):
        sample = (df.columns.get_level_values("delta") == "sample")
        metric = (df.columns.get_level_values("metric") == "bucket_size")
        select_cols = df.columns[(sample & metric)]
        baseline = (df.columns.get_level_values("dataset_id") == legend_info.labels[self.benchmark.uuid])
        sort_cols = df.columns[(sample & metric & baseline)].to_list()
        return sort_cols, select_cols

    def _get_table_legend_info(self, df, legend_info):
        col_df = df.columns.to_frame()
        by_label = legend_info.with_label_index()
        _, pivot_colors = col_df.align(by_label["colors"], axis=0, level="dataset_id")
        _, pivot_labels = col_df.align(by_label["labels"], axis=0, level="dataset_id")
        index = pd.Index(df.columns, name="column")
        return LegendInfo.from_index(index, colors=pivot_colors, labels=pivot_labels)

    def generate(self, surface, cell):
        df = self.get_dataset(DatasetName.VMSTAT_UMA_INFO).agg_df
        if not check_multi_index_aligned(df, "dataset_id"):
            self.logger.error("Unaligned index, skipping plot")
            return
        # Group by bucket_size in the baseline
        legend_info = self.get_legend_info()
        view_df = legend_info.map_labels_to_level(df, "dataset_id", axis=0)
        view_df = pivot_multi_index_level(view_df, "dataset_id")
        sort_cols, show_cols = self._get_show_columns(view_df, legend_info)
        view_df = view_df.sort_values(sort_cols)
        view = TableDataView(view_df, columns=show_cols)
        view.legend_info = self._get_table_legend_info(view_df, legend_info)
        view.legend_level = []
        cell.add_view(view)


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
        return self.benchmark.get_plot_path() / "vmstat_tables"


class VMStatUMAMetricHist(BenchmarkSubPlot):
    """
    Histogram of UMA metrics across datasets
    """
    def __init__(self, plot, dataset, metric: str, normalized: bool = False):
        """
        plot: the parent plot
        metric: the base metric to display
        normalized: whether to plot the delta or the norm_delta column
        """
        super().__init__(plot)
        # Optional zone info dataset
        self.ds = dataset
        self.metric = metric
        self.normalized = normalized
        # Compute column names to use
        delta_col = (self.metric, "median", "delta_baseline")
        rel_col = (self.metric, "median", "norm_delta_baseline")
        if delta_col not in self.ds.agg_df.columns:
            delta_col = (self.metric, "-", "delta_baseline")
            rel_col = (self.metric, "-", "norm_delta_baseline")
        self.delta_col = delta_col
        self.rel_col = rel_col

    def get_legend_info(self):
        # Use base legend to separate by axis
        base = self.build_legend_by_dataset()

        if self.normalized:
            legend = base.map_label(lambda l: f"% {Symbols.DELTA}{self.metric} " + l)
        else:
            legend = base.map_label(lambda l: f"{Symbols.DELTA}{self.metric} " + l)
        return legend.assign_colors_hsv("dataset_id", h=(0.2, 1), s=(0.7, 0.7), v=(0.6, 0.9))

    def get_line_legend_info(self):
        base = self.build_legend_by_dataset()

        if self.normalized:
            legend = base.map_label(lambda l: f"median % {Symbols.DELTA}|{self.metric}|")
        else:
            legend = base.map_label(lambda l: f"median {Symbols.DELTA}|{self.metric}|")
        return legend.remap_colors("Greys", color_range=(0.5, 1))

    @property
    def bar_limit(self):
        """Get maximum number of bars to show, split among groups"""
        return 60

    def get_cell_title(self):
        pct = "% " if self.normalized else ""
        return f"UMA {self.metric} {pct}variation w.r.t. baseline"

    def get_df(self):
        """
        Generate base dataset with any auxiliary columns we need.
        Drop the baseline data as we are showing deltas
        """
        df = self.ds.agg_df[self.ds.agg_df.index.get_level_values("dataset_id") != self.benchmark.uuid].copy()
        df["abs_delta"] = df[self.delta_col].abs()
        df[self.rel_col] *= 100
        return df

    def get_filtered_df(self, df):
        """
        Only select the entries in the high 10% absolute delta
        """
        ngroups = len(df.index.get_level_values("dataset_id").unique())
        max_entries = int(self.bar_limit / ngroups)
        if max_entries <= 1:
            self.warning("Broken plot for %s metric: too many datasets cut bar entries to less than 1 per group",
                         self.metric)
        high_df = quantile_slice(df, ["abs_delta"], quantile=0.9, max_entries=max_entries, level=["dataset_id"])
        # Drop values where abs_delta is 0 everywhere
        index_complement = high_df.index.names.difference(["dataset_id"])
        zeros = high_df.groupby(index_complement)["abs_delta"].transform(lambda g: ((g == 0) | g.isna()).all())
        return high_df[~zeros]

    def generate(self, surface, cell):
        """
        We filter metric to show only the values for the top 90th percentile
        of the delta, this avoid cluttering the plots with meaningless data.
        """
        self.logger.debug("extract plot metric %s normalized=%s", self.metric, self.normalized)

        df = self.get_df()
        high_df = self.get_filtered_df(df)
        high_df["x"] = assign_sorted_coord(high_df, sort=["abs_delta"], group_by=["dataset_id"], ascending=False)

        if self.normalized:
            ycolumn = self.rel_col
            ylabel = f"% {Symbols.DELTA}{self.metric}"
        else:
            ycolumn = self.delta_col
            ylabel = f"{Symbols.DELTA}{self.metric}"

        # If we don't have data, skip
        if len(high_df):
            view = BarPlotDataView(high_df, x="x", yleft=ycolumn)
            view.bar_axes_ordering = "interleaved"
            view.bar_group = "dataset_id"
            view.legend_info = self.get_legend_info()
            view.legend_level = ["dataset_id"]
            cell.add_view(view)

            # Add a line for the y
            df_line = df.groupby("dataset_id").median()[["abs_delta"]]
            view = AALineDataView(df_line, horizontal=["abs_delta"])
            view.style.line_style = "dashed"
            view.style.line_width = 0.5
            view.legend_info = self.get_line_legend_info()
            view.legend_level = ["dataset_id"]
            cell.add_view(view)

        cell.x_config.label = "UMA Zone"
        cell.x_config.ticks = high_df["x"].unique()
        cell.x_config.tick_labels = high_df.index.get_level_values("name").unique()
        cell.x_config.tick_rotation = 90
        cell.yleft_config.label = ylabel


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

    def _make_subplots_mosaic(self):
        """
        Make subplots mosaic as a single column.
        """
        subplots = {}
        layout = []
        uma_stats = self.get_dataset(DatasetName.VMSTAT_UMA)
        for idx, metric in enumerate(uma_stats.data_columns()):
            name = f"subplot-uma-stats-{idx}"
            name_norm = name + "-N"
            subplots[name] = VMStatUMAMetricHist(self, uma_stats, metric)
            subplots[name_norm] = VMStatUMAMetricHist(self, uma_stats, metric, normalized=True)
            layout.append([name, name_norm])
        uma_info = self.get_dataset(DatasetName.VMSTAT_UMA_INFO)
        for idx, metric in enumerate(uma_info.data_columns()):
            name = f"subplot-uma-info-{idx}"
            name_norm = name + "-N"
            subplots[name] = VMStatUMAMetricHist(self, uma_info, metric)
            subplots[name_norm] = VMStatUMAMetricHist(self, uma_info, metric, normalized=True)
            layout.append([name, name_norm])
        return Mosaic(layout, subplots)

    def get_plot_name(self):
        return "VMStat metrics distribution"

    def get_plot_file(self):
        return self.benchmark.get_plot_path() / "vmstat-histograms"
