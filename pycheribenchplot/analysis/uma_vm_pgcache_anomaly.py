import itertools as it

import numpy as np
import pandas as pd

from ..core.dataset import DatasetName, assign_sorted_coord
from ..core.plot import (AALineDataView, BarPlotDataView, BenchmarkPlot, BenchmarkSubPlot, CellData, HistPlotDataView,
                         LegendInfo, Mosaic, Scale)
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


class UMABucketAffinityHist(BenchmarkSubPlot):
    """
    Show histogram of number of zones using each bucket zone. This helps understanding
    the pressure a zone receives from the rest of the system.
    """
    @classmethod
    def get_required_datasets(cls):
        dsets = super().get_required_datasets()
        dsets += [DatasetName.VMSTAT_UMA_INFO, DatasetName.KERNEL_STRUCT_STATS]
        return dsets

    def get_cell_title(self):
        return "Bucket affinity distribution"

    def generate(self, fm, cell):
        size_col = ("bucket_size", "-", "sample")
        df = self.get_dataset(DatasetName.VMSTAT_UMA_INFO).agg_df
        struct_stats_df = self.get_dataset(DatasetName.KERNEL_STRUCT_STATS).merged_df
        # Note that UMA bucket sizes larger than 16 use some of the space for the bucket header.
        # This must be accounted for when binning and it depends on the pointer size.
        s_uma_bucket = struct_stats_df.xs("uma_bucket", level="name")
        datasets = df.index.get_level_values("dataset_id").unique()
        if len(s_uma_bucket) != len(datasets):
            self.logger.error("kernel DWARF stats for struct uma_bucket do not agree with vmstat data. Skipping plot.")
            return
        # The number of buckets in each histogram is the same, but change the effective bucket limits
        buckets = [2**i for i in range(1, 9)]
        view_df = pd.DataFrame(index=pd.Index(buckets, name="bucket_size"))
        for dsid, chunk in df.groupby("dataset_id"):
            hdr_size = s_uma_bucket.xs(dsid, level="dataset_id")[("size", "sample")]
            assert len(hdr_size) == 1, "More than 1 uma_struct size per dataset_id?"
            hdr_size = hdr_size.iloc[0]
            if dsid in self.benchmark.merged_benchmarks:
                ptr_size = self.benchmark.merged_benchmarks[dsid].instance_config.kernel_pointer_size
            else:
                ptr_size = self.benchmark.instance_config.kernel_pointer_size
            # histogram uses half open intervals [a, b), we want the reverse here (a, b] where a, b are
            # UMA bucket sizes.
            eff_limits = [1, 3, 5, 9, 17]
            eff_limits += [2**i + 1 - np.ceil(hdr_size / ptr_size) for i in range(5, 9)]
            count, out_bins = np.histogram(chunk[size_col], bins=eff_limits)
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
    subplots = [UMABucketAffinityHist, UMABucketRefillEff]

    def get_plot_name(self):
        return "UMA Bucket analysis"

    def get_plot_file(self):
        return self.benchmark.manager.plot_output_path / "uma-bucket-vm-pgcache-summary"
