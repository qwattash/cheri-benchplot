import itertools as it

import numpy as np
import pandas as pd

from ..core.dataset import (DatasetName, assign_sorted_coord, broadcast_xs, subset_xs)
from ..core.plot import (AALineDataView, ArrowPlotDataView, BarPlotDataView, BenchmarkPlot, BenchmarkSubPlot, CellData,
                         HistPlotDataView, LegendInfo, Mosaic, Scale)
from ..vmstat.plot import VMStatUMAMetricHist


class UMABucketAnomalyFail(VMStatUMAMetricHist):
    """
    Highlight the purecap kernel vm_pgcache UMA anomaly
    """
    def get_zone_names(self):
        """
        Zone names to include in the plot
        """
        all_zones = self.ds.agg_df.index.get_level_values("name").unique()
        zones = [z for z in all_zones if "vm pgcache" in z]
        zones += ["8 Bucket", "64 Bucket", "128 Bucket", "256 Bucket"]
        return zones

    def get_cell_title(self):
        return "UMA vm_pgcache zone anomaly"

    def get_filtered_df(self, df):
        """
        Only select the zones we care about
        """
        zones = self.get_zone_names()
        return df[df.index.get_level_values("name").isin(zones)].copy()


class UMABucketAllocAnomaly(BenchmarkPlot):
    """
    Collect data to explain the UMA bucket allocation anomaly triggered by
    vm_pgcache. This is currently enabled for all benchmarks as it seems to
    be a common issue.
    """
    @classmethod
    def check_required_datasets(cls, dsets: list):
        """Check dataset list against qemu stats dataset names"""
        required = set([DatasetName.VMSTAT_UMA, DatasetName.VMSTAT_UMA_INFO])
        return required.issubset(set(dsets))

    def _get_uma_stats(self):
        return [
            "rsize", "ipers", "bucket_size", "requests", "free", "fail", "fail_import", "bucket_alloc", "bucket_free",
            "pressure"
        ]

    def _make_subplots_mosaic(self):
        """
        Build subplots as a single mosaic column
        """
        subplots = {}
        layout = []
        want_stats = self._get_uma_stats()
        uma_stats = self.get_dataset(DatasetName.VMSTAT_UMA)
        uma_info = self.get_dataset(DatasetName.VMSTAT_UMA_INFO)
        for idx, metric in enumerate(uma_stats.data_columns()):
            if metric in want_stats:
                name = f"subplot-uma-stats-{idx}"
                subplots[name] = UMABucketAnomalyFail(self, uma_stats, metric)
                layout.append([name])
        for idx, metric in enumerate(uma_info.data_columns()):
            if metric in want_stats:
                name = f"subplot-uma-info-{idx}"
                subplots[name] = UMABucketAnomalyFail(self, uma_info, metric)
                layout.append([name])
        return Mosaic(layout, subplots)

    def get_plot_name(self):
        return "UMA Bucket vm_pgcache anomaly"

    def get_plot_file(self):
        return self.benchmark.manager.plot_output_path / "uma-bucket-vm-pgcache-anomaly"


class UMABucketAffinityBase(BenchmarkSubPlot):
    """
    Base class for collecting information about UMA bucket zones
    """
    @classmethod
    def get_required_datasets(cls):
        dsets = super().get_required_datasets()
        dsets += [DatasetName.VMSTAT_UMA_INFO, DatasetName.KERNEL_STRUCT_STATS]
        return dsets

    def _get_ptr_size(self, dsid):
        if dsid in self.benchmark.merged_benchmarks:
            return self.benchmark.merged_benchmarks[dsid].instance_config.kernel_pointer_size
        else:
            return self.benchmark.instance_config.kernel_pointer_size

    def get_uma_df(self):
        """
        Return the UMA info dataset with an additional column to map the bucket size to the
        corresponding bucket zone

        XXX should we do this in the dataset directly?
        """
        size_col = ("bucket_size", "-", "sample")
        df = self.get_dataset(DatasetName.VMSTAT_UMA_INFO).agg_df.copy()
        struct_stats_df = self.get_dataset(DatasetName.KERNEL_STRUCT_STATS).merged_df
        # Note that UMA bucket sizes larger than 16 use some of the space for the bucket header.
        # so we compute the actual size of the header and subtract it where needed.
        s_uma_bucket = struct_stats_df.xs("uma_bucket", level="name")
        datasets = df.index.get_level_values("dataset_id").unique().to_frame()["dataset_id"]
        if len(s_uma_bucket) != len(datasets):
            self.logger.error("kernel DWARF stats for struct uma_bucket do not agree with vmstat data. Skipping plot.")
            return
        ptr_size = pd.Series([self._get_ptr_size(dsid) for dsid in datasets], index=datasets)
        hdr_size = s_uma_bucket.groupby("dataset_id").first()[("size", "sample")]
        hdr_items = (hdr_size / ptr_size).apply(np.ceil)
        # Build a mapping df with the base bucket sizes for each dataset
        # we then subtract the header items and use this to fixup the bucket_zone column
        bucket_size_base = pd.Series([0] + [2**i for i in range(1, 9)], name="bucket_zone")
        zone_mapping = pd.merge(datasets, bucket_size_base, how="cross").set_index("dataset_id")
        size_no_hdr = zone_mapping["bucket_zone"] - hdr_items
        zone_mapping["bucket_zone_size"] = zone_mapping["bucket_zone"].mask(zone_mapping["bucket_zone"] > 16,
                                                                            size_no_hdr)
        tmp_df = df.copy()
        tmp_df.columns = df.columns.to_flat_index()
        tmp_df = tmp_df.join(zone_mapping, on="dataset_id")
        high_cond = tmp_df[size_col] <= tmp_df["bucket_zone_size"]
        low_cond = tmp_df[size_col] > tmp_df["bucket_zone_size"].shift(1, fill_value=0)
        tmp_df = tmp_df[high_cond & low_cond]
        # Note bucket_zone will be nan for zones that have buckets disabled
        df["bucket_zone"] = tmp_df["bucket_zone"]
        return df


class UMABucketAffinityHist(UMABucketAffinityBase):
    """
    Show histogram of number of zones using each bucket zone. This helps understanding
    the pressure a zone receives from the rest of the system.
    """
    def get_cell_title(self):
        return "Bucket affinity distribution"

    def generate(self, fm, cell):
        df = self.get_uma_df()
        # The number of buckets in each histogram is the same, but change the effective bucket limits
        # histogram uses half open intervals [a, b), we want the reverse here (a, b] where a, b are
        # UMA bucket sizes, so we need to add 1 to get the limits
        buckets = [2**i for i in range(1, 9)]
        hist_limits = [1] + [2**i + 1 for i in range(1, 9)]
        view_df = pd.DataFrame(index=pd.Index(buckets, name="bucket_size"))
        for dsid, chunk in df.groupby("dataset_id"):
            count, out_limits = np.histogram(chunk["bucket_zone"], bins=hist_limits)
            view_df[dsid] = count

        view_df = view_df.melt(var_name="dataset_id", value_name="count", ignore_index=False)
        view_df.set_index("dataset_id", append=True, inplace=True)
        view = BarPlotDataView(view_df, x="bucket_size", yleft=["count"], bar_group="dataset_id")
        view.legend_info = self.build_legend_by_dataset()
        view.legend_level = ["dataset_id"]
        cell.x_config.label = "Bucket size"
        cell.x_config.ticks = buckets
        cell.x_config.tick_labels = buckets
        cell.x_config.scale = Scale("log", base=2)
        cell.yleft_config.label = "# Zones using the bucket zone"
        cell.add_view(view)


class UMABucketAffinityDelta(UMABucketAffinityBase):
    """
    Show the change in bucket zone by displaying an arrow that shows the change.
    The X axis is the bucket size, the Y axis is the zone name.
    """
    def get_cell_title(self):
        return "Bucket affinity delta"

    def get_mosaic_extent(self):
        return (3, 1)

    def get_legend_info(self):
        legend = self.build_legend_by_dataset()
        legend = legend.assign_colors_hsv("dataset_id", h=(0.2, 1), s=(0.7, 0.7), v=(0.6, 0.9))
        return legend

    def generate(self, fm, cell):
        df = self.get_uma_df()
        # Drop all entries where the bucket_zone does not change or where there is no bucket zone
        index_complement = df.index.names.difference(["dataset_id"])
        changed = df.groupby(index_complement).nunique()["bucket_zone"] > 1
        view_df = subset_xs(df, changed).copy()
        assert not view_df["bucket_zone"].isna().any()

        base = view_df.xs(self.benchmark.uuid, level="dataset_id")
        sorted_base = base.sort_values("bucket_zone")
        sorted_base["y"] = range(len(sorted_base))
        view_df["y"] = broadcast_xs(view_df, sorted_base["y"])

        view = ArrowPlotDataView(view_df,
                                 x="bucket_zone",
                                 y="y",
                                 group_by=["dataset_id"],
                                 base_group=[self.benchmark.uuid])
        view.legend_info = self.get_legend_info()
        view.legend_level = ["dataset_id"]
        cell.add_view(view)

        x_ticks = view_df["bucket_zone"].unique()
        y_ticks = view_df["y"].unique()
        cell.x_config.label = "Bucket zone"
        cell.x_config.ticks = x_ticks
        cell.x_config.tick_labels = [f"{x:.0f}" for x in x_ticks]
        cell.x_config.scale = Scale("log", base=2)
        cell.yleft_config.padding = 0.02
        cell.yleft_config.limits = (y_ticks.min(), y_ticks.max())
        cell.yleft_config.origin_line = False
        cell.yleft_config.label = "Zone"
        cell.yleft_config.ticks = y_ticks
        cell.yleft_config.tick_labels = view_df.index.get_level_values("name").unique()


class UMABucketRefillEff(BenchmarkSubPlot):
    """
    Show the number of requests made to each bucket zone. This is used to report pressure on each bucket
    zone.
    """
    @classmethod
    def get_required_datasets(cls):
        dsets = super().get_required_datasets()
        dsets += [DatasetName.VMSTAT_UMA, DatasetName.VMSTAT_UMA_INFO]
        return dsets

    def get_cell_title(self):
        return "Bucket zone refill efficiency"

    def get_bucket_zones(self):
        return [f"{2**i} Bucket" for i in range(1, 9)]

    def generate(self, fm, cell):
        info_df = self.get_dataset(DatasetName.VMSTAT_UMA_INFO).agg_df
        cols = [("bucket_refill_efficiency", "-", "sample")]

        sel = info_df.index.get_level_values("name").isin(self.get_bucket_zones())
        view_df = info_df[sel].copy()
        view_df[cols] *= 100
        view_df["x"] = assign_sorted_coord(view_df, sort=[("rsize", "-", "sample")], group_by=["dataset_id"])
        view = BarPlotDataView(view_df, x="x", yleft=cols, bar_group="dataset_id")
        view.legend_info = self.build_legend_by_dataset()
        view.legend_level = ["dataset_id"]
        cell.x_config.label = "Bucket zone"
        cell.x_config.ticks = view_df["x"].unique()
        cell.x_config.tick_labels = view_df.index.get_level_values("name").unique()
        cell.yleft_config.label = "% Bucket items on refill"
        cell.add_view(view)


class UMABucketAnalysis(BenchmarkPlot):
    """
    Collect bucket-related data for the pgcache anomaly analysis
    """
    subplots = [UMABucketAffinityHist, UMABucketRefillEff, UMABucketAffinityDelta]

    def get_plot_name(self):
        return "UMA Bucket analysis"

    def get_plot_file(self):
        return self.benchmark.manager.plot_output_path / "uma-bucket-vm-pgcache-summary"
