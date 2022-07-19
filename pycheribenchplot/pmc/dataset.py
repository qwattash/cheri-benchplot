import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.csv import CSVDataSetContainer
from ..core.dataset import (DatasetArtefact, DatasetName, Field, align_multi_index_levels, make_index_key)
from ..core.elf import SymInfo
from ..core.json import JSONDataSetContainer


class PMCStatData(CSVDataSetContainer):
    fields = [
        Field.index_field("progname", dtype=str),
        Field.index_field("archname", dtype=str),
    ]

    def __init__(self, benchmark, dset_key, config):
        self._index_transform = lambda df: []
        super().__init__(benchmark, dset_key, config)
        self.stats_df = None
        # self._gen_extra_index = lambda df: self.benchmark.pmc_map_index(df)
        # extra = self._gen_extra_index(pd.DataFrame(columns=["progname", "archname"]))
        # self._extra_index = list(extra.columns)

    def add_aggregate_col(self, new_df: pd.DataFrame, prefix: str):
        new_df = new_df.add_prefix(prefix + "_")
        if self.agg_df is None:
            self.agg_df = new_df
        else:
            self.agg_df = pd.concat([self.agg_df, new_df], axis=1)

    def _load_csv(self, path: Path, **kwargs):
        csv_df = super()._load_csv(path, **kwargs)

        # Generate extra index columns if any
        prog_and_arch = csv_df[["progname", "archname"]]
        idx_df = self._index_transform(prog_and_arch)
        if len(idx_df):
            csv_df = pd.concat((csv_df, idx_df), axis=1)
            # Properly set the dataset index
        return csv_df

    def load_iteration(self, iteration):
        path = self.iteration_output_file(iteration)
        csv_df = self._load_csv(path)
        csv_df["iteration"] = iteration
        self._append_df(csv_df)


class TooobaStatcountersData(PMCStatData):
    """
    Statcounters description and processing for CHERI Toooba (RISC-V)
    """
    dataset_config_name = DatasetName.PMC
    dataset_source_id = DatasetArtefact.PMC
    fields = [
        Field.data_field("cycles"),
        Field.data_field("instructions"),
        Field.data_field("time"),
        Field.data_field("redirect", "PC redirects"),  # 0x01
        Field.data_field("branch"),  # 0x03
        Field.data_field("jal"),  # 0x04
        Field.data_field("jalr"),  # 0x05
        Field.data_field("trap", "Stage 2 trap"),  # 0x02
        Field.data_field("auipc"),  # 0x06
        Field.data_field("load"),  # 0x07
        Field.data_field("store"),  # 0x08
        Field.data_field("LR"),  # 0x09
        Field.data_field("SC"),  # 0x0a
        Field.data_field("AMO"),  # 0x0b
        Field.data_field("serial_shift"),  # 0x0c
        Field.data_field("integer_muldiv", "Integer Mul/Div insn count"),  # 0x0d
        Field.data_field("FP", "Floating point insn count"),  # 0x0e
        Field.data_field("SC_success", "SC success"),  # 0x0f
        Field.data_field("load_wait", "Stage 2 cycles waiting on load"),  # 0x10
        Field.data_field("store_wait", "Stage 2 cycles waiting on store"),  # 0x11
        Field.data_field("fence", "Fence insn count"),  # 0x12
        Field.data_field("F_busy", "Cycles where stage F is busy"),  # 0x13
        Field.data_field("D_busy", "Cycles where stage D is busy"),  # 0x14
        Field.data_field("1_busy", "Cycles where stage 1 is busy"),  # 0x15
        Field.data_field("2_busy", "Cycles where stage 2 is busy"),  # 0x16
        Field.data_field("3_busy", "Cycles where stage 3 is busy"),  # 0x17
        Field.data_field("imprecise_setbounds", "Count setbounds NOT resulting in the exact bounds requested"),  # 0x18
        Field.data_field("cap_unrepresentable", "Count capability tag lost due to unrepresentable bounds"),  # 0x19
        Field.data_field("cap_load", "Stage 2 capability-wide load"),  # 0x1a
        Field.data_field("cap_store", "Stage 2 capability-wide store"),  # 0x1b
        Field.data_field("cap_load_tag_set", "Stage 2 loads tagged capability"),  # 0x1c
        Field.data_field("cap_store_tag_set", "Stage 2 stores tagged capability"),  # 0x1d
        Field.data_field("icache_load", "iCache load count"),  # 0x20
        Field.data_field("icache_load_miss", "iCache load missed"),  # 0x21
        Field.data_field("icache_load_miss_wait", "iCache load miss latency (cycles)"),  # 0x22
        Field.data_field("dcache_load", "dCache load count"),  # 0x30
        Field.data_field("dcache_load_miss", "dCache load missed"),  # 0x31
        Field.data_field("dcache_load_miss_wait", "dCache load miss latency (cycles)"),  # 0x32
        Field.data_field("icache_store", "iCache store count"),  # 0x23
        Field.data_field("icache_store_miss", "iCache store missed -- UNIMPL"),
        Field.data_field("icache_store_miss_wait", "iCache store miss latency (cycles) -- UNIMPL"),
        Field.data_field("dcache_store", "dCache store count"),  # 0x33
        Field.data_field("dcache_store_miss", "dCache store missed -- UNIMPL"),
        Field.data_field("dcache_store_miss_wait", "dCache store miss latency (cycles) -- UNIMPL"),
        Field.data_field("dcache_amo", "dCache atomic operation requested"),  # 0x36
        Field.data_field("dcache_amo_miss", "dCache atomic operation missed"),  # 0x37
        Field.data_field("dcache_amo_miss_wait", "dCache atomic operation miss latency (cycles)"),  # 0x38
        Field.data_field("itlb_access", "iTLB access count"),  # 0x29
        Field.data_field("itlb_miss", "iTLB miss count"),  # 0x2a
        Field.data_field("itlb_miss_wait", "iTLB miss latency (cycles)"),  # 0x2b
        Field.data_field("itlb_flush", "iTLB flush"),  # 0x2c
        Field.data_field("dtlb_access", "dTLB access count"),  # 0x39
        Field.data_field("dtlb_miss", "dTLB miss count"),  # 0x3a
        Field.data_field("dtlb_miss_wait", "dTLB miss latency (cycles)"),  # 0x3b
        Field.data_field("dtlb_flush", "iTLB flush"),  # 0x3c
        Field.data_field("icache_evict", "iCache eviction count"),  # 0x2d
        Field.data_field("dcache_evict", "dCache eviction count"),  # 0x3d
        Field.data_field("llcache_load_miss", "last-level cache load missed"),  # 0x61
        Field.data_field("llcache_load_miss_wait", "last-level cache load miss latency (cycles)"),  # 0x62
        Field.data_field("llcache_evict", "last-level cache eviction count"),  # 0x64
        Field.data_field("tagcache_store", "tag cache store count"),  # 0x40
        Field.data_field("tagcache_store_miss", "tag cache store missed"),  # 0x41
        Field.data_field("tagcache_load", "tag cache load count"),  # 0x42
        Field.data_field("tagcache_load_miss", "tag cache load missed"),  # 0x43
        Field.data_field("tagcache_evict", "tag cache eviction count"),  # 0x44
        Field.data_field("tagcache_set_store", "tag cache set tag write"),  # 0x45
        Field.data_field("tagcache_set_load", "tag cache set tag read"),  # 0x46
        # Derived metrics
        Field.data_field("ipc", "instruction per cycle", isderived=True),
        Field.data_field("icache_load_hit_rate", "iCache load hit rate", isderived=True),
        Field.data_field("dcache_load_hit_rate", "dCache load hit rate", isderived=True),
        Field.data_field("icache_store_hit_rate", "iCache store hit rate", isderived=True),
        Field.data_field("dcache_store_hit_rate", "dCache store hit rate", isderived=True),
        Field.data_field("itlb_hit_rate", "iTLB hit rate", isderived=True),
        Field.data_field("dtlb_hit_rate", "dTLB hit rate", isderived=True),
        Field.data_field("dcache_amo_hit_rate", "dCache atomic op hit rate", isderived=True),
        Field.data_field("llcache_load_hit_rate", "LL cache load hit rate", isderived=True),
        Field.data_field("tagcache_store_hit_rate", "tag cache store hit rate", isderived=True),
        Field.data_field("tagcache_load_hit_rate", "tag cache load hit rate", isderived=True),
    ]

    def valid_data_columns(self):
        data_cols = set(self.data_columns())
        avail_cols = set(self.df.columns).intersection(data_cols)
        return list(avail_cols)

    def valid_input_columns(self):
        data_cols = set(self.input_non_index_columns())
        avail_cols = set(self.df.columns).intersection(data_cols)
        return list(avail_cols)

    def _hit_rate(self, name):
        """
        Generate a derived cache hit-rate metric from the base PMC name.
        This uses the _miss and _access columns.
        """
        miss_col = f"{name}_miss"
        access_col = f"{name}_access"
        hit_col = f"{name}_hit_rate"
        if miss_col not in self.valid_input_columns():
            self.df[hit_col] = np.nan
            return

        if access_col in self.df.columns:
            total = self.df[access_col]
        elif name in self.df.columns:
            total = self.df[name]
        else:
            self.logger.warning("No total column for %s, skip computing derived hit_rate", name)
            total = np.nan
        self.df[hit_col] = ((total - self.df[miss_col]) / total) * 100
        if (self.df[hit_col] < 0).any():
            self.logger.error("Invalid negative hit rate for %s", name)

    def pre_merge(self):
        super().pre_merge()
        # Compute derived metrics
        self.df["ipc"] = self.df["instructions"] / self.df["cycles"]
        self._hit_rate("icache_load")
        self._hit_rate("dcache_load")
        self._hit_rate("icache_store")
        self._hit_rate("dcache_store")
        self._hit_rate("itlb")
        self._hit_rate("dtlb")
        self._hit_rate("dcache_amo")
        self._hit_rate("llcache_load")
        self._hit_rate("tagcache_load")
        self._hit_rate("tagcache_store")

    def _get_aggregation_strategy(self):
        agg_strat = super()._get_aggregation_strategy()
        # Remove the columns that are not currently present
        avail_cols = self.valid_data_columns()
        return {k: agg_list for k, agg_list in agg_strat.items() if k in avail_cols}

    def aggregate(self):
        # Median and quartiles
        group_index = self.dataset_id_columns() + ["progname", "archname"]
        grouped = self.merged_df[self.valid_data_columns()].groupby(group_index)
        self.agg_df = self._compute_aggregations(grouped)

    def post_aggregate(self):
        super().post_aggregate()
        # Compute relative metrics with respect to baseline
        tmp_df = self._add_delta_columns(self.agg_df)
        self.agg_df = self._compute_delta_by_dataset(tmp_df)

    def iteration_output_file(self, iteration):
        return super().iteration_output_file(iteration).with_suffix(".csv")


class StackTrace:
    """
    Helper class to contain stacktraces in the dataframe index.
    """
    def __init__(self, stacktrace: list[int], syminfo: list[SymInfo]):
        self.addr = tuple(stacktrace)
        self.syms = syminfo
        self.key = ";".join([si.name for si in syminfo])

    def __str__(self):
        return self.key

    def __repr__(self):
        return self.key

    def __eq__(self, other):
        return self.key == other.key

    def __lt__(self, other):
        return self.key < other.key

    def __hash__(self):
        return hash(self.key)


class ProfclockStackSamples(JSONDataSetContainer):
    """
    Handle libpmc/pmcstat output for profclock-based system sampling.

    When running, we expect the main benchmark to produce a file with pmcstat
    compatible contents. We invoke pmcstat to dump the data as json and later
    process it on the host machine.

    # XXX In order to make structured observations about callchains we build an interal representation
    # containing a tree for each iteration. We then combine the trees across iterations in the aggregation
    # phase and later to produce deltas.
    """
    dataset_config_name = DatasetName.PMC_PROFCLOCK_STACKSAMPLE
    dataset_source_id = DatasetArtefact.PMC_PROFCLOCK_STACKSAMPLE
    fields = [
        Field.index_field("seq", desc="Callchain sample sequence ID", dtype=int),
        Field.index_field("cpu", dtype=int),
        Field.index_field("pid", dtype=int),
        Field.index_field("mode", dtype=str),
        Field("stacktrace", dtype=object),
        Field.index_field("st", isderived=True, dtype=StackTrace),
        Field.derived_field("nsamples", dtype=int),
    ]

    def _build_stacktrace(self, idx: "namedtuple", row: pd.Series) -> StackTrace:
        stacktrace = map(lambda addr: int(addr, base=16), row["stacktrace"])
        if idx.mode == "s":
            resolver = self.benchmark.sym_resolver

            annotated = [resolver.lookup_fn(addr, as_key="kernel.full") or SymInfo.unknown(addr) for addr in stacktrace]
        else:
            annotated = [SymInfo.unknown(addr) for addr in stacktrace]
        return StackTrace(row["stacktrace"], annotated)

    def _resolve_pids(self):
        """
        Resolve PIDs to process names. In case of failure, use the pid as a string.
        """
        pidmap = self.benchmark.get_dataset_by_artefact(DatasetArtefact.PIDMAP)
        assert pidmap is not None, "The pidmap dataset is required for pmc sampling"
        join_df = self.df.merge(pidmap.df, how="left", left_on="pid", right_on="pid")
        self.df["process"] = join_df["command"]

    def _delta_align_levels(self):
        """
        Return the levels that need to be aligned across all datasets before computing deltas.
        """
        return ["pid", "mode", "st"]

    def iteration_output_guestfile(self, iteration):
        return super().iteration_output_file(iteration).with_suffix(".pmc")

    def iteration_output_file(self, iteration):
        return super().iteration_output_file(iteration).with_suffix(".json")

    def gen_pre_extract_results(self, script):
        super().gen_pre_extract_results(script)
        # Expect the benchmark to have generated the output guestfiles
        # for each iteration. We now dump them with pmcstat to the "real"
        # json output file
        for i in range(self.benchmark.config.iterations):
            pmc_output_path = self.iteration_output_guestfile(i)
            pmc_json_path = self.iteration_output_file(i)
            guest_pmc_output = script.local_to_remote_path(pmc_output_path)
            guest_json_output = script.local_to_remote_path(pmc_json_path)
            script.gen_cmd("pmcstat", ["--libxo", "json", "-R", guest_pmc_output, "-o", guest_json_output])

    async def after_extract_results(self, script, instance):
        await super().after_extract_results(script, instance)
        # Extract the profclock output files
        for i in range(self.benchmark.config.iterations):
            output_path = self.iteration_output_file(i)
            guest_json_path = script.local_to_remote_path(output_path)
            self.logger.debug("Extract %s -> %s", guest_json_path, output_path)
            await instance.extract_file(guest_json_path, output_path)

    def load_iteration(self, iteration):
        super().load_iteration(iteration)
        path = self.iteration_output_file(iteration)
        with open(path, "r") as fd:
            data = json.load(fd, strict=False)
        pmc_data = data["pmcstat"]["pmc-log-entry"]
        json_df = pd.DataFrame.from_records(pmc_data)
        json_df["iteration"] = iteration
        # Filter by type, we are only interested in callchain events
        # We may separately process the others to help with PID/process mapping
        callchain_df = json_df[json_df["type"] == "callchain"].reset_index()
        callchain_df["seq"] = callchain_df.index
        self._append_df(callchain_df)

    def pre_merge(self):
        super().pre_merge()
        IndexKey = make_index_key(self.df)
        # Resolve PID mappings
        self._resolve_pids()
        # Annotate callchains
        self.df["st"] = self.df.apply(lambda row: self._build_stacktrace(IndexKey(*row.name), row), axis="columns")
        self.df.set_index("st", append=True, inplace=True)
        # XXX could now drop the "stacktrace field if we care about its size"

        # Aggregate within each iteration to determine the number of samples for each callchain
        self.df = self.df.groupby(self.df.index.names.difference(["seq"])).count()
        self.df = self.df.rename(columns={"stacktrace": "nsamples"})

    def aggregate(self):
        super().aggregate()
        # Aggregate across iterations to sum common callchains
        grouped = self.merged_df.groupby(self.merged_df.index.names.difference(["iteration", "cpu"]))
        self.agg_df = self._compute_aggregations(grouped)

    def post_aggregate(self):
        super().post_aggregate()
        print(self.agg_df.index.names)
        align_levels = self._delta_align_levels()
        tmp_df = align_multi_index_levels(self.agg_df, align_levels, fill_value=0)
        tmp_df = self._add_delta_columns(tmp_df)
        self.agg_df = self._compute_delta_by_dataset(tmp_df)
