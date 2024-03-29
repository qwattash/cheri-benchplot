import numpy as np
import pandas as pd

from ..core.config import DatasetName
from ..core.dataset import check_multi_index_aligned, generalized_xs
from ..core.plot import (BenchmarkPlot, BenchmarkSubPlot, LegendInfo, LinePlotDataView, Mosaic)


class QEMUCounterTrack(BenchmarkSubPlot):
    """
    Base class for QEMU counter track line plots.
    """
    def __init__(self, plot, dataset, track, slot):
        super().__init__(plot)
        self.ds = dataset
        self.track_name = track
        self.track_slot = slot
        self.df = generalized_xs(dataset.merged_df, [track, slot], ["name", "slot"])
        if "counter_name" in self.df.index.names:
            counter_name = self.df.index.get_level_values("counter_name").unique()
            assert len(counter_name) == 1
            self.track_desc = counter_name[0]
        else:
            self.track_desc = track

    def generate(self, fm, cell):
        self.logger.debug("extract qemu counter track %s (%s:%d)", self.track_desc, self.track_name, self.track_slot)

        view = LinePlotDataView(self.df, x="ts", yleft="value", line_group=["dataset_id", "iteration"])
        view.legend_info = self.build_legend_by_dataset()
        view.legend_level = ["dataset_id"]
        cell.add_view(view)
        cell.x_config.label = "Time (ns)"
        cell.yleft_config.label = "Metric"


class UMAZoneCounter(QEMUCounterTrack):
    """
    Subplot to show a line plot of a single UMA per-zone counter.
    """
    def get_cell_title(self):
        return f"UMA zone counter {self.track_desc}"


class KernMemCounter(QEMUCounterTrack):
    """
    Subplot to show a line plot for a kernel mem track
    """
    def get_cell_title(self):
        return f"vm_kern counter {self.track_desc}"


class UMAQEMUCountersPlot(BenchmarkPlot):
    """
    Show vstat datasets distribution of interesting metrics
    """
    @classmethod
    def check_enabled(cls, datasets, config):
        required = {DatasetName.QEMU_UMA_COUNTERS, DatasetName.VMSTAT_UMA_INFO}
        return required.issubset(datasets)

    def _make_subplots_mosaic(self):
        subplots = {}
        layout = []
        counters = self.get_dataset(DatasetName.QEMU_UMA_COUNTERS)
        counter_tracks = counters.merged_df.index.get_level_values("name").unique()
        for idx, track in enumerate(counter_tracks):
            track_df = counters.merged_df.xs(track, level="name")
            for slot in track_df.index.get_level_values("slot").unique():
                name = f"subplot-qemu-uma-counter-{idx}-{slot}"
                subplots[name] = UMAZoneCounter(self, counters, track, slot)
                layout.append([name])
        return Mosaic(layout, subplots)

    def get_plot_name(self):
        return "QEMU UMA counters"

    def get_plot_file(self):
        return self.benchmark.get_plot_path() / "qemu-uma-counters"


class KernMemQEMUCountersPlot(BenchmarkPlot):
    """
    Show vstat datasets distribution of interesting metrics
    """
    @classmethod
    def check_enabled(cls, datasets, config):
        """
        Check dataset list against qemu stats dataset names
        """
        required = {
            DatasetName.QEMU_VM_KERN_COUNTERS,
        }
        return required.issubset(datasets)

    def _make_subplots_mosaic(self):
        subplots = {}
        layout = []
        counters = self.get_dataset(DatasetName.QEMU_VM_KERN_COUNTERS)
        counter_tracks = counters.merged_df.index.get_level_values("name").unique()
        for idx, track in enumerate(counter_tracks):
            track_df = counters.merged_df.xs(track, level="name")
            for slot in track_df.index.get_level_values("slot").unique():
                name = f"subplot-qemu-uma-counter-{idx}-{slot}"
                subplots[name] = UMAZoneCounter(self, counters, track, slot)
                layout.append([name])
        return Mosaic(layout, subplots)

    def get_plot_name(self):
        return "QEMU kmem counters"

    def get_plot_file(self):
        return self.benchmark.get_plot_path() / "qemu-vmkern-counters"
