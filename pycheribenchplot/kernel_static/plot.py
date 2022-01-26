import numpy as np
import pandas as pd

from ..core.dataset import (DatasetName, check_multi_index_aligned, pivot_multi_index_level, subset_xs)
from ..core.matplotlib import MatplotlibSurface
from ..core.plot import (BenchmarkPlot, BenchmarkSubPlot, CellData, ColorMap, HistPlotDataView, Scale, Surface)


class KernelBoundsDistribution(BenchmarkSubPlot):
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

    def get_cell_title(self):
        return "Kernel bounds histogram"

    def generate(self, surface: Surface, cell: CellData):
        """
        Generate interleaved histograms with one set of bars for each dataset, so
        that we have side-by-side comparison of the buckets.
        """
        df = self.bounds_stats.agg_df
        nviews = len(df.index.get_level_values("__dataset_id").unique())
        # Determine buckets we are going to use
        min_size = max(df["size"].min(), 1)
        max_size = max(df["size"].max(), 1)
        log_buckets = range(int(np.log2(min_size)), int(np.log2(max_size)) + 1)
        buckets = [2**i for i in log_buckets]

        # Build histograms for each dataset
        view = HistPlotDataView("hist",
                                df,
                                x="size",
                                x_scale=Scale("log", base=2),
                                buckets=buckets,
                                bucket_group="__dataset_id")
        cell.add_view(view)


class KernelSubobjectPlot(BenchmarkPlot):
    """
    Report subobject bounds statistics for the kernel
    """

    subplots = [
        KernelBoundsDistribution,
    ]

    def __init__(self, benchmark):
        super().__init__(benchmark, [MatplotlibSurface()])

    def get_plot_name(self):
        return "Kernel subobject bounds stats"

    def get_plot_file(self):
        return self.benchmark.manager.session_output_path / "kernel-subobject-bounds"
