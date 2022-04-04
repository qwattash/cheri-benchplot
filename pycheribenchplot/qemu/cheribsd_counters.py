from ..core.dataset import DatasetName, Field
from .dataset import QEMUGuestCountersDataset


class QEMUUMACountersDataset(QEMUGuestCountersDataset):
    """
    Dataset with UMA QEMU counters from cheribsd.
    Note that this requires the UMA info dataset to identify valid counter zone names.
    """
    dataset_config_name = DatasetName.QEMU_UMA_COUNTERS

    # Counter slots. Keep in sync with cheribsd/sys/vm/uma_core.c
    PCPU_CACHE_ITEMS = 0
    PCPU_CACHE_ALLOC = 50
    IMPORTED_ITEMS = 100
    BUCKETS = 101
    FALLBACK_ALLOC = 102
    FALLBACK_FREE = 103
    PRESSURE = 104
    NOVM_PRESSURE = 105
    KEG_PAGES = 106

    fields = [Field.index_field("counter_name", dtype=str, isderived=True)]

    counter_name_map = {
        PCPU_CACHE_ITEMS: "CPU {cpuidx} cached items",
        PCPU_CACHE_ALLOC: "CPU {cpuidx} active allocations",
        IMPORTED_ITEMS: "imported items",
        BUCKETS: "buckets",
        FALLBACK_ALLOC: "fallback 1-item allocs",
        FALLBACK_FREE: "fallback 1-item frees",
        PRESSURE: "useful bucket contention",
        NOVM_PRESSURE: "NOVM bucket contention",
        KEG_PAGES: "keg pages",
    }

    def _get_monotonic_slots(self):
        """
        Return a list of slot numbers that hold monotonic counters.
        These counters must be realigned after each iteration, so that they
        start from zero.
        """
        return [self.FALLBACK_ALLOC, self.FALLBACK_FREE, self.PRESSURE, self.NOVM_PRESSURE]

    def _get_counter_name(self, name, slot):
        slot_desc = self.counter_name_map[slot]
        cpuidx = None
        if slot < self.PCPU_CACHE_ALLOC:
            cpuidx = slot - self.PCPU_CACHE_ITEMS
        elif slot < self.IMPORTED_ITEMS:
            cpuidx = slot - self.PCPU_CACHE_ALLOC
        if cpuidx is not None:
            slot_desc = slot_desc.format(cpuidx=cpuidx)
        return name + " " + slot_desc

    def configure(self, opts):
        # Verify that the UMA info dataset is also enabled
        if self.benchmark.get_dataset(DatasetName.VMSTAT_UMA_INFO) is None:
            self.logger.error("%s requires the VMSTAT_UMA_INFO dataset, missing in config.", self.__class__.__name__)
            raise DatasetProcessingError("Failed configuration")
        return super().configure(opts)

    def pre_merge(self):
        super().pre_merge()
        # Drop the counters that do not have valid UMA zone names
        uma_info_df = self.benchmark.get_dataset(DatasetName.VMSTAT_UMA_INFO).df
        zone_names = uma_info_df.index.get_level_values("name").unique()
        valid_counters = self.df.index.isin(zone_names, level="name")
        new_df = self.df.loc[valid_counters].copy()
        # Generate counter name from name and slot
        slots = new_df.index.get_level_values("slot")
        pcpu_items = (slots >= self.PCPU_CACHE_ITEMS) & (slots < self.PCPU_CACHE_ALLOC)
        pcpu_alloc = (slots >= self.PCPU_CACHE_ALLOC) & (slots < self.IMPORTED_ITEMS)
        new_df["counter_name"] = new_df.index.to_frame().apply(lambda r: self._get_counter_name(r["name"], r["slot"]),
                                                               axis=1)
        new_df = new_df.set_index("counter_name", append=True)
        # Align monotonic counters after each iteration
        for slot in self._get_monotonic_slots():
            sel = (slots == slot)
            base_value = new_df.loc[sel].groupby(["dataset_id", "__iteration", "name"]).first()
            relative = new_df.loc[sel] - base_value + 1
            new_df.loc[sel] = relative.reorder_levels(new_df.index.names)
        assert not new_df["value"].isna().any()
        # Add synthetic slots for the sum of cache items and cache allocations
        # TODO
        self.df = new_df

    def aggregate(self):
        """
        Two-step aggregation:
        1. Aggregate across iterations
        2. Generate deltas across datasets
        """
        super().aggregate()
        # TODO


class QEMUKernMemCountersDataset(QEMUGuestCountersDataset):
    dataset_config_name = DatasetName.QEMU_VM_KERN_COUNTERS

    def _get_kmem_counter_names(self):
        names = ["kva", "kmem"]
        return names

    def pre_merge(self):
        super().pre_merge()
        # Only grab interesting counters
        valid_counters = self.df.index.isin(self._get_kmem_counter_names(), level="name")
        self.df = self.df.loc[valid_counters].copy()
