import numpy as np
import pandas as pd

from ..core.dataset import (DatasetName, check_multi_index_aligned, generalized_xs)
from ..core.plot import (BenchmarkPlot, BenchmarkSubPlot, LegendInfo, LinePlotDataView, Mosaic)


class UMAZoneCounter(BenchmarkSubPlot):
    """
    Subplot to show a line plot of a single UMA per-zone counter.
    """
    def __init__(self, plot, dataset, track, slot):
        super().__init__(plot)
        self.ds = dataset
        self.track_name = track
        self.track_slot = slot
        self.df = generalized_xs(dataset.merged_df, [track, slot], ["name", "slot"])
        counter_name = self.df.index.get_level_values("counter_name").unique()
        assert len(counter_name) == 1
        self.track_desc = counter_name[0]

    def get_cell_title(self):
        return f"UMA Zone counter {self.track_desc}"

    def generate(self, fm, cell):
        self.logger.debug("extract qemu counter track %s (%s:%d)", self.track_desc, self.track_name, self.track_slot)

        view = LinePlotDataView(self.df, x="ts", yleft="value", line_group=["dataset_id", "__iteration"])
        view.legend_info = self.build_legend_by_dataset()
        view.legend_level = ["dataset_id"]
        cell.add_view(view)
        cell.x_config.label = "Time (ns)"
        cell.yleft_config.label = "Metric"


class UMAQEMUCountersPlot(BenchmarkPlot):
    """
    Show vstat datasets distribution of interesting metrics
    """
    @classmethod
    def check_required_datasets(cls, dsets: list[DatasetName]):
        """
        Check dataset list against qemu stats dataset names
        """
        required = set([DatasetName.QEMU_UMA_COUNTERS, DatasetName.VMSTAT_UMA_INFO])
        return required.issubset(set(dsets))

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
        return "QEMU UMA Counters"

    def get_plot_file(self):
        return self.benchmark.manager.plot_output_path / "qemu-uma-counters"
