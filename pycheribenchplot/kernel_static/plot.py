import numpy as np
import pandas as pd

from ..core.dataset import (DatasetName, assign_sorted_coord, check_multi_index_aligned, filter_aggregate,
                            pivot_multi_index_level, quantile_slice, stacked_histogram, subset_xs)
from ..core.plot import (AALineDataView, BarPlotDataView, BenchmarkPlot, BenchmarkSubPlot, BenchmarkTable,
                         HistPlotDataView, LegendInfo, Scale, Symbols, TableDataView)


class SetBoundsDistribution(BenchmarkSubPlot):
    """
    Draw kernel (static) csetbounds distribution by size of enforced bounds.
    """
    @classmethod
    def get_required_datasets(cls):
        dsets = super().get_required_datasets()
        dsets += [DatasetName.KERNEL_CSETBOUNDS_STATS]
        return dsets

    def __init__(self, plot):
        super().__init__(plot)
        self.bounds_stats = self.get_dataset(DatasetName.KERNEL_CSETBOUNDS_STATS)
        assert self.bounds_stats is not None, "Can not find required dataset"

    def get_stats_df(self):
        raise NotImplementedError("Must override")

    def get_legend_info(self):
        df = self.get_stats_df()
        datasets = df.index.get_level_values("dataset_id").unique()
        legend_info = self.build_legend_by_dataset()
        # Only use the datasets in the dataframe
        legend_index = legend_info.info_df.index
        avail_labels = legend_index[legend_index.isin(datasets)]
        return LegendInfo(legend_info.info_df.reindex(avail_labels))


class SetBoundsSimpleDistribution(SetBoundsDistribution):
    def generate(self, fm, cell):
        """
        Generate interleaved histograms with one set of bars for each dataset, so
        that we have side-by-side comparison of the buckets.
        """
        df = self.get_stats_df()
        nviews = len(df.index.get_level_values("dataset_id").unique())
        # Determine buckets we are going to use
        min_size = max(df["size"].min(), 1)
        max_size = max(df["size"].max(), 1)
        log_buckets = range(int(np.log2(min_size)), int(np.log2(max_size)) + 1)
        buckets = [2**i for i in log_buckets]

        # Build histograms for each dataset
        view = HistPlotDataView(df, x="size", buckets=buckets, bucket_group="dataset_id")
        view.legend_info = self.get_legend_info()
        cell.x_ticks = buckets
        cell.x_config.label = "Bounds size (bytes)"
        cell.x_config.scale = Scale("log", base=2)
        cell.yleft_config.label = "#csetbounds"
        cell.add_view(view)


class KernelBoundsDistribution(SetBoundsSimpleDistribution):
    def get_cell_title(self):
        return "Kernel bounds histogram"

    def get_stats_df(self):
        df = self.bounds_stats.merged_df
        return df[df["src_module"] == "kernel"]


class ModulesBoundsDistribution(SetBoundsSimpleDistribution):
    def get_cell_title(self):
        return "Modules bounds histogram"

    def get_stats_df(self):
        df = self.bounds_stats.merged_df
        return df[df["src_module"] != "kernel"]


class KernelBoundsDistributionByKind(SetBoundsDistribution):
    def get_cell_title(self):
        return "Kernel bounds by kind"

    def get_stats_df(self):
        df = self.bounds_stats.merged_df
        return df[df["src_module"] == "kernel"]

    def get_kind_legend_info(self):
        df = self.get_stats_df()
        kinds = df["kind"].unique()
        index = pd.Index(kinds, name="kind")
        return LegendInfo(index, cmap_name="Paired", labels=[k.name for k in kinds])

    def generate(self, fm, cell):
        """
        Generate interleaved and stacked histograms.
        Horizontal bars are created for each datasets, stacked bars are generated
        for each setbounds kind.
        """
        df = self.get_stats_df()
        nviews = len(df.index.get_level_values("dataset_id").unique())
        # Determine buckets we are going to use
        min_size = max(df["size"].min(), 1)
        max_size = max(df["size"].max(), 1)
        log_buckets = range(int(np.log2(min_size)), int(np.log2(max_size)) + 1)
        buckets = [2**i for i in log_buckets]

        # Build the stacked histogram dataframe
        hist_df = stacked_histogram(df, group="dataset_id", stack="kind", data_col="size", bins=buckets)

        view = BarPlotDataView(hist_df, x="bin_start", yleft="count", bar_group="dataset_id", stack_group="kind")

        # Build histograms for each dataset
        view.legend_info = self.get_kind_legend_info()
        view.legend_level = ["kind"]
        cell.x_config.ticks = buckets
        cell.x_config.label = "Bounds size (bytes)"
        cell.x_config.scale = Scale("log", base=2)
        cell.yleft_config.label = "#csetbounds"
        cell.add_view(view)


class KernelStructStatsPlot(BenchmarkSubPlot):
    """
    Base class for kernel struct size plots
    """
    @classmethod
    def get_required_datasets(cls):
        dsets = super().get_required_datasets()
        dsets += [DatasetName.KERNEL_STRUCT_STATS]
        return dsets

    def __init__(self, plot):
        super().__init__(plot)
        self.struct_stat = self.get_dataset(DatasetName.KERNEL_STRUCT_STATS)
        assert self.struct_stat is not None, "Can not find required dataset"

    def get_legend_info(self):
        legend = self.build_legend_by_dataset()
        legend.remap_colors("Paired")
        return legend


class KernelStructSizeHist(KernelStructStatsPlot):
    def get_median_line_legend(self):
        legend = {
            uuid: "median " + str(bench.instance_config.name)
            for uuid, bench in self.benchmark.merged_benchmarks.items()
        }
        legend[self.benchmark.uuid] = f"median {self.benchmark.instance_config.name}(*)"
        index = pd.Index(legend.keys(), name="dataset_id")
        return LegendInfo.from_index(index, cmap_name="Greys", labels=legend.values(), color_range=(0.5, 1))

    def get_hist_column(self):
        raise NotImplementedError("Must override")

    def get_df(self):
        return self.struct_stat.merged_df

    def get_agg_df(self):
        return self.struct_stat.agg_df

    def get_df_selector(self):
        return None

    def build_buckets(self, df):
        # Determine bucket sizes
        hcol = self.get_hist_column()
        min_size = df[hcol].min()
        max_size = df[hcol].max()
        if np.sign(min_size) == np.sign(max_size) or min_size == 0 or max_size == 0:
            interval = sorted(np.abs((min_size, max_size)))
            log_min, log_max = np.ceil(np.log2(np.maximum(interval, 1))).astype(int)
            buckets = [2**i for i in range(log_min, log_max + 1)]
            if min_size < 0:
                buckets = list(-1 * np.array(buckets))
        else:
            neg_min = abs(min_size)
            neg_log = np.ceil(np.log2(neg_min)).astype(int)
            pos_log = np.ceil(np.log2(max_size)).astype(int)
            neg_buckets = [-2**i for i in range(0, neg_log + 1)]
            pos_buckets = [2**i for i in range(0, pos_log + 1)]
            buckets = neg_buckets + pos_buckets
        return sorted(buckets)

    def generate(self, fm, cell):
        df = self.get_df()
        agg_df = self.get_agg_df()
        hcol = self.get_hist_column()
        buckets = self.build_buckets(df)

        # Build histogram
        view = HistPlotDataView(df, x=hcol, buckets=buckets, bucket_group="dataset_id")
        view.legend_info = self.get_legend_info()
        view.legend_level = ["dataset_id"]
        cell.add_view(view)

        # Add help lines for the median struct size
        line_cols = [hcol + ("median", )]
        view = AALineDataView(agg_df, vertical=line_cols)
        view.style.line_style = "dashed"
        view.legend_info = self.get_median_line_legend()
        view.legend_level = ["dataset_id"]
        cell.add_view(view)

        cell.x_config.label = "Size (bytes)"
        cell.x_config.scale = Scale("symlog", base=2, lintresh=1, linscale=0.25)
        cell.x_config.ticks = buckets
        cell.x_config.limits = (min(buckets), max(buckets))
        cell.yleft_config.label = "# structs"


class KernelStructSizeDistribution(KernelStructSizeHist):
    """
    Draw kernel structure size shift in distribution
    """
    def get_cell_title(self):
        return "Kernel structure size distribution"

    def get_hist_column(self):
        return ("size", "sample")


class KernelStructSizeOverhead(KernelStructSizeHist):
    def get_cell_title(self):
        return "Kernel structure size overhead"

    def get_hist_column(self):
        return ("size", "delta_baseline")

    def get_df(self):
        df = super().get_df()
        # take the non-baseline delta values only
        baseline = df.index.get_level_values("dataset_id") == self.benchmark.uuid
        return df[~baseline]

    def get_agg_df(self):
        df = super().get_agg_df()
        # take the non-baseline delta values only
        baseline = df.index.get_level_values("dataset_id") == self.benchmark.uuid
        return df[~baseline]

    def generate(self, fm, cell):
        super().generate(fm, cell)
        cell.x_config.label = "size delta (bytes)"
        cell.yleft_config.scale = Scale("log", base=10)


class KernelStructSizeRelOverhead(KernelStructSizeHist):
    def get_cell_title(self):
        return "Kernel structure size overhead"

    def get_hist_column(self):
        return ("size", "norm_delta_baseline")

    def get_df(self):
        df = super().get_df()
        # take the non-baseline delta values only
        baseline = df.index.get_level_values("dataset_id") == self.benchmark.uuid
        df = df[~baseline].copy()
        df[self.get_hist_column()] *= 100
        return df

    def get_agg_df(self):
        df = super().get_agg_df()
        # take the non-baseline delta values only
        baseline = df.index.get_level_values("dataset_id") == self.benchmark.uuid
        df = df[~baseline].copy()
        df[self.get_hist_column() + ("median", )] *= 100
        return df

    def build_buckets(self, df):
        min_val = df[self.get_hist_column()].min()
        max_val = df[self.get_hist_column()].max()
        min_val = int(np.floor(min_val / 10) * 10)
        max_val = int(np.ceil(max_val / 10) * 10)
        buckets = range(min_val, max_val + 10, 10)
        return buckets

    def generate(self, fm, cell):
        super().generate(fm, cell)
        cell.x_config.label = "% size delta"
        cell.x_config.scale = Scale("linear")
        cell.yleft_config.scale = Scale("log", base=10)


class KernelStructPaddingHighOverhead(KernelStructStatsPlot):
    def get_df(self):
        # Omit baseline as we are looking at deltas
        sel = (self.struct_stat.merged_df.index.get_level_values("dataset_id") != self.benchmark.uuid)
        return self.struct_stat.merged_df[sel]

    def get_legend_info(self):
        legend_base = self.build_legend_by_dataset()
        if len(self.columns) > 1:
            legend_merge = {}
            for col, desc in zip(self.columns, self.columns_desc):
                legend_merge[col] = legend_base.map_label(lambda l: desc + " " + l)
            legend_info = LegendInfo.combine("column", legend_merge)
            legend_info.remap_colors("Paired", group_by="dataset_id")
        else:
            legend_info = legend_base.map_label(lambda l: self.columns_desc[0] + " " + l)
            legend_info.remap_colors("Paired")
        return legend_info

    def get_high_overhead_df(self, quantile, maxbar):
        df = self.get_df()
        ngroups = len(df.index.get_level_values("dataset_id").unique())
        max_entries = int(maxbar / ngroups)
        match = df[df[self.columns] >= df[self.columns].quantile(quantile)]
        if len(match) > maxbar:
            self.logger.warning("capping high delta entries to %d, %dth percentile contains %d", maxbar, quantile * 100,
                                len(match) / ngroups)
        # Actually compute the frame. Note that this may have slightly more entries than maxbar
        high_df = quantile_slice(df, self.columns, quantile, max_entries, ["dataset_id"])
        return high_df

    def generate(self, fm, cell):
        high_df = self.get_high_overhead_df(0.9, 50)
        high_df["x"] = assign_sorted_coord(high_df, sort=self.columns, group_by=["dataset_id"], ascending=False)

        view = BarPlotDataView(high_df, x="x", yleft=self.columns, bar_group="dataset_id")
        view.legend_info = self.get_legend_info()
        view.legend_level = ["dataset_id"]
        cell.add_view(view)

        cell.x_config.label = "struct name"
        cell.x_config.ticks = high_df["x"].unique()
        cell.x_config.tick_labels = high_df.index.get_level_values("name").unique()
        cell.x_config.tick_rotation = 90
        cell.yleft_config.label = Symbols.DELTA + "padding (bytes)"


class KernelStructPaddingOverhead(KernelStructPaddingHighOverhead):
    columns = [("total_pad", "delta_baseline")]
    columns_desc = [Symbols.DELTA + "padding"]

    def get_cell_title(self):
        return "Top kernel struct padding overhead"


class KernelStructNestedPaddingOverhead(KernelStructPaddingHighOverhead):
    columns = [("nested_total_pad", "delta_baseline")]
    columns_desc = [Symbols.DELTA + "nested padding"]

    def get_cell_title(self):
        return "Top kernel cumulative struct padding overhead"


class KernelStructPaddingOverheadTable(KernelStructPaddingHighOverhead):
    columns = [("total_pad", "delta_baseline"), ("nested_total_pad", "delta_baseline")]
    columns_desc = [Symbols.DELTA + "padding", Symbols.DELTA + "nested padding"]

    def get_legend_info(self):
        return None

    def get_cell_title(self):
        return "Top kernel struct padding overhead"

    def generate(self, fm, cell):
        high_df = self.get_high_overhead_df(0.9, np.inf)
        view = TableDataView(high_df, columns=self.columns)
        view.legend_info = self.get_legend_info()
        view.legend_level = high_df.index.names
        cell.add_view(view)


class KernelStructPointerDensity(KernelStructStatsPlot):
    def get_cell_title(self):
        return "Kernel struct pointer density"

    def generate(self, fm, cell):
        df = self.struct_stat.merged_df.copy()
        ptr_count_col = ("ptr_count", "sample")
        m_count_col = ("member_count", "sample")

        df["ptr_density"] = (df[ptr_count_col] / df[m_count_col]) * 100
        buckets = range(0, 101, 5)

        view = HistPlotDataView(df, x="ptr_density", buckets=buckets, bucket_group="dataset_id")
        view.legend_info = self.build_legend_by_dataset()
        view.legend_level = ["dataset_id"]
        cell.add_view(view)

        cell.x_config.label = "% pointer density"
        cell.x_config.ticks = buckets
        cell.yleft_config.label = "# structs"


class KernelStructPackedSizeDelta(KernelStructStatsPlot):
    """
    Measure the size difference in structures, considering only the member size, as if the structure
    was packed.
    Note that nested structure padding is unchanged, perhaps it should be removed as well.
    """
    def get_cell_title(self):
        return "Kernel struct packed size delta"

    def generate(self, fm, cell):
        df = self.struct_stat.merged_df.copy()


class KernelStaticInfoPlot(BenchmarkPlot):
    """
    Report subobject bounds and struct statistics for the kernel
    """

    subplots = [
        KernelBoundsDistribution,
        ModulesBoundsDistribution,
        KernelBoundsDistributionByKind,
        KernelStructSizeDistribution,
        KernelStructSizeOverhead,
        KernelStructSizeRelOverhead,
        KernelStructPaddingOverhead,
        KernelStructNestedPaddingOverhead,
        KernelStructPointerDensity,
    ]

    def get_plot_name(self):
        return "Kernel compile-time stats"

    def get_plot_file(self):
        return self.benchmark.manager.plot_output_path / "kernel-static-stats"


class KernelStructSizeLargeOverhead(KernelStructStatsPlot):
    """
    Show the structures responsible for the largest overhead
    """
    def get_col(self):
        raise NotImplementedError("Must override")

    def generate(self, fm, cell):
        df = self.struct_stat.merged_df
        col = self.get_col()
        # Get range high 10%
        high_thresh = df[col].quantile(0.9)
        cond = (df[col] >= high_thresh)
        high_df = df[cond]
        if "__iteration" in df.index.names:
            high_df = high_df.droplevel("__iteration")
        assert high_df.index.is_unique, "Non unique index?"

        legend_info = self.get_legend_info()
        # Currently only use the legend_info to map labels to dataset_id
        show_df = legend_info.map_labels_to_level(high_df, "dataset_id", axis=0)
        show_df = pivot_multi_index_level(show_df, "dataset_id")
        # sort by highest delta
        sort_cols_sel = None
        for i, level_value in enumerate(col):
            sel = (show_df.columns.get_level_values(i) == level_value)
            if sort_cols_sel is None:
                sort_cols_sel = sel
            else:
                sort_cols_sel = sort_cols_sel & sel
        sort_cols = show_df.columns[sort_cols_sel].to_list()
        show_df.sort_values(by=sort_cols, ascending=False, inplace=True)

        show_cols = show_df.columns.get_level_values("metric").isin(self.struct_stat.data_columns())
        col_idx = show_df.columns[show_cols]
        view = TableDataView(show_df, columns=col_idx)
        # force no legend coloring, have not set it up yet
        view.legend_info = None
        cell.add_view(view)


class KernelStructSizeLargeRelOverhead(KernelStructSizeLargeOverhead):
    def get_cell_title(self):
        return "Large % overhead (>90th percentile)"

    def get_col(self):
        return ("size", "norm_delta_baseline")


class KernelStructSizeLargeAbsOverhead(KernelStructSizeLargeOverhead):
    def get_cell_title(self):
        return "Large absolute overhead (>90th percentile)"

    def get_col(self):
        return ("size", "delta_baseline")


class PAHoleTable(BenchmarkSubPlot):
    """
    Produce tabular output to provide information similar to what pahole generates.
    Struct members are on the Y axis. Datasets are pivoted to the columns axis.
    """
    @classmethod
    def get_required_datasets(cls):
        dsets = super().get_required_datasets()
        dsets += [DatasetName.KERNEL_STRUCT_STATS, DatasetName.KERNEL_STRUCT_MEMBER_STATS]
        return dsets

    def __init__(self, plot):
        super().__init__(plot)
        self.struct_stats = self.get_dataset(DatasetName.KERNEL_STRUCT_STATS)
        self.member_stats = self.get_dataset(DatasetName.KERNEL_STRUCT_MEMBER_STATS)

    def get_cell_title(self):
        return "Kernel pahole"

    def build_table_legend(self, view_df) -> tuple[LegendInfo, pd.DataFrame]:
        """
        Generate the legend for the pivoted table.
        We color rows in alternate colors and highlight the padding members.
        """
        legend_levels = ["color_stripe", "is_synth_member"]
        ntiles = len(view_df) / 2
        stripe_index = np.tile(np.arange(0, 2), int(ntiles))
        if ntiles != int(ntiles):
            stripe_index = np.append(stripe_index, 0)
        synth_member = view_df.index.get_level_values("member_name").str.endswith(".pad")
        # Insert the legend keys into the view frame
        view_df["color_stripe"] = stripe_index
        view_df["is_synth_member"] = synth_member

        # Build legend info
        legend_index = pd.MultiIndex.from_product([[0, 1], [True, False]], names=legend_levels)
        legend = LegendInfo.from_index(legend_index, [""] * len(legend_index))
        legend.info_df["colors"] = LegendInfo.gen_colors(legend.info_df,
                                                         mapname="Greys",
                                                         groupby=["color_stripe"],
                                                         color_range=(0.7, 1))
        return legend, legend_levels

    def generate(self, fm, cell):
        struct_df = self.struct_stats.merged_df
        member_df = self.member_stats.merged_df
        if "__iteration" in struct_df.index.names:
            struct_df = struct_df.droplevel("__iteration")
        if "__iteration" in member_df.index.names:
            member_df = member_df.droplevel("__iteration")

        # First drop any struct that has no change in padding and size
        same_pad = struct_df[("total_pad", "delta_baseline")] == 0
        same_size = struct_df[("size", "delta_baseline")] == 0
        not_changed_df = filter_aggregate(struct_df, same_pad & same_size, ["dataset_id"], how="all")
        drop = pd.Series(True, index=not_changed_df.index)

        pahole_df = self.member_stats.gen_pahole_table()
        pahole_df = subset_xs(pahole_df, drop, complement=True)
        # Ensure things are still aligned
        assert check_multi_index_aligned(pahole_df, ["name", "src_file", "src_line", "member_name"])

        # Pivot the dataset_id level to the columns. We only care about the member_offset and member_size
        # columns at this point
        legend_info = self.build_legend_by_dataset()
        view_df = pahole_df[["member_offset", "member_size"]]
        view_df = legend_info.map_labels_to_level(view_df, "dataset_id", axis=0)
        view_df = pivot_multi_index_level(view_df, "dataset_id")
        # After pivoting, we need to order the struct members by member_offset
        # member reordering is tricky to handle so just ignore it and sort by the baseline member_offset
        member_offset_baseline = ("member_offset", legend_info.labels[self.benchmark.uuid])
        view_df = view_df.sort_values(view_df.index.names + [member_offset_baseline])

        # Generate the table legend and associated columns
        table_legend, legend_levels = self.build_table_legend(view_df)

        table_columns = ["member_size"]
        col_sel = view_df.columns.get_level_values("metric").isin(table_columns)
        view = TableDataView(view_df, columns=view_df.columns[col_sel])
        view.legend_info = table_legend
        view.legend_level = legend_levels
        cell.add_view(view)


class KernelStaticInfoTables(BenchmarkTable):
    """
    Report tables of struct statistics with large deltas for the kernel
    """

    subplots = [
        KernelStructSizeLargeRelOverhead,
        KernelStructSizeLargeAbsOverhead,
        PAHoleTable,
    ]

    def get_plot_name(self):
        return "Kernel compile-time detailed stats"

    def get_plot_file(self):
        return self.benchmark.manager.plot_output_path / "kernel-static-tables"
