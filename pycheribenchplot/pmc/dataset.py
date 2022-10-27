import json
import typing
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from ..core.csv import CSVDataSetContainer
from ..core.dataset import (DatasetArtefact, DatasetName, Field, align_multi_index_levels, generalized_xs,
                            make_index_key)
from ..core.elf import SymInfo
from ..core.json import JSONDataSetContainer
from ..core.util import timing


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


class CallChainItem:
    def __init__(self, node_id: int, addr: int, sym: SymInfo, df: pd.DataFrame):
        self.node_id = node_id
        # Address
        self.addr = addr
        # Pandas DataFrame containing samples.
        # The shape depends on the aggregation steps performed on the tree,
        # initially a single value Series will be present
        self.df = df
        # Symbol information
        self.sym = sym

    def __hash__(self):
        return hash(self.node_id)

    def __str__(self):
        return f"[id:{self.node_id} {self.sym.name}@{self.addr:x}]"

    def __repr__(self):
        return str(self)


CallTreeType = typing.TypeVar("CallTree", bound="CallTree")


class CallTree:
    """
    Helper class to build and maintain the datastructure for stack samples
    """
    def __init__(self, df_factory: typing.Callable):
        """
        :param df_factory: Builder for the inner dataframe
        :param levels: Index levels associated to call chain items
        :param keys: Index keys for the current call tree
        """
        self.ct = nx.DiGraph()
        self.df_factory = df_factory
        self.root = self._make_node(-1, SymInfo.unknown(-1))
        self.ct.add_node(self.root)

    def _make_node(self, addr, sym):
        node = CallChainItem(len(self.ct), addr, sym, self.df_factory())
        return node

    def _combine_subtree(self, target: CallTreeType, to_merge: CallTreeType, node: CallChainItem,
                         node_to_merge: CallChainItem):
        """
        Combine step for two subtrees, this is designed to be a recursive combine operation.
        Tree depth should be in the order of 10s, as the callchain depth is no much larger.
        Very naive and slow implementation.
        """
        node.df.loc[node_to_merge.df.index, :] = node_to_merge.df
        for succ in to_merge.ct.successors(node_to_merge):
            for matching in target.ct.successors(node):
                if succ.sym.name == matching.sym.name:
                    break
            else:
                # The root will contain all the indexes that we will see
                # Every node will be readded when combining and go through this path
                matching = target._make_node(succ.addr, succ.sym)
                target.ct.add_node(matching)
                target.ct.add_edge(node, matching)
            matching.df.loc[succ.df.index, :] = succ.df
            self._combine_subtree(target, to_merge, matching, succ)

    def _map_subtree(self, target: CallTreeType | None, node: CallChainItem | None, local_node: CallChainItem,
                     fn: typing.Callable):
        """
        Apply function to the inner frame of the successors of local_node and store the result in
        the corresponding node successors. local_node is assumed to belong to the current tree.

        :param target: The target calltree where to generate nodes for the results, if None the results are replaced in place
        :param node: The current node in the target calltree we are inspecting, it is ignored if target is None
        :param local_node: The current node in the local calltree we are inspecting
        :param fn: The function to apply
        """
        if target is None:
            fn(local_node, local_node)
        else:
            fn(node, local_node)

        for succ in self.ct.successors(local_node):
            if target is None:
                mapped_node = succ
            else:
                mapped_node = target._make_node(succ.addr, succ.sym)
                target.ct.add_node(mapped_node)
                target.ct.add_edge(node, mapped_node)
            self._map_subtree(target, mapped_node, succ, fn)

    def _visit_subtree(self, node: CallChainItem, fn: typing.Callable, level: int = 0, parent: CallChainItem = None):
        fn(node, level, parent)
        for succ in self.ct.successors(node):
            self._visit_subtree(succ, fn, level + 1, node)

    def _insert_subtree(self, node: CallChainItem, cc: list[str | int], symbolizer_fn, insert_fn):
        """
        Recursive implementation for insert_callchain()
        """
        addr = cc.pop()
        if isinstance(addr, str):
            addr = int(addr, 16)
        # Resolve the symbol
        sym = symbolizer_fn(addr) or SymInfo.unknown(addr)
        # Find out whether we need to add a node
        for succ in self.ct.successors(node):
            if succ.sym.name == sym.name:
                # Continue down the path
                break
        else:
            succ = self._make_node(addr, sym)
            self.ct.add_node(succ)
            self.ct.add_edge(node, succ)
        assert succ is not None
        insert_fn(succ, isleaf=(len(cc) == 0))
        if cc:
            self._insert_subtree(succ, cc, symbolizer_fn, insert_fn)

    @property
    def index(self):
        """
        Return the current index for all inner dataframes.
        Note that the index is kept aligned, so the root node will contain the same index as the other nodes
        """
        return self.root.df.index

    @property
    def columns(self):
        """
        Return the current column index for all inner dataframes.
        Note that the index is kept aligned, so the root node will contain the same columns as the other nodes
        """
        return self.root.df.columns

    def insert_callchain(self, cc: list[str | int], symbolizer_fn: typing.Callable[[int], SymInfo],
                         insert_fn: typing.Callable[[pd.DataFrame], None]):
        """
        Add a callchain sample to the tree.

        For each node on the callchain, the insert_fn is called on the node dataframe.
        """
        if cc:
            insert_fn(self.root)
        self._insert_subtree(self.root, cc, symbolizer_fn, insert_fn)

    def combine(self, to_merge: list[CallTreeType]) -> CallTreeType:
        """
        Combine calltrees and store data needed to show median and quartiles
        of the samples for each node. Missing nodes will be assumed to have 0-filled frames.

        :param to_merge: Input calltree collection. Note that the index levels must
        be the same as the current calltree.
        """
        union_index = self.index
        for ct in to_merge:
            assert ct.index.names == union_index.names, f"Invalid index levels {ct.index.names} != {union_index.names}"
            union_index = union_index.union(ct.index)

        # First we create a new empty calltree for the union
        # Take care to retain the indexes correctly
        def df_factory():
            return pd.DataFrame(0, index=union_index, columns=self.columns)

        union_ct = CallTree(df_factory)

        # Now generate the tree union using the current tree and the others
        for ct in [self] + to_merge:
            self._combine_subtree(union_ct, ct, union_ct.root, ct.root)
        return union_ct

    def map(self, fn: typing.Callable, inplace=False) -> CallTreeType:
        """
        Execute an aggregation or transformation function on each internal frame.
        This cam be used to compute deltas and error bars.
        """
        if inplace:
            mapped_ct = None
            mapped_root = None
        else:
            mapped_ct = CallTree(self.df_factory)
            mapped_root = mapped_ct.root

        # Now populate the new tree
        def mapper(dst_node, src_node):
            dst_node.df = fn(src_node.df)

        self._map_subtree(mapped_ct, mapped_root, self.root, mapper)
        if inplace:
            return self
        else:
            return mapped_ct

    def visit(self, fn: typing.Callable[[CallChainItem, int, CallChainItem], None]):
        """
        Execute a function on each node, giving the current layer index and parent.

        :param fn: Callable invoked on each visit as fn(node, layer_index, parent)
        """
        self._visit_subtree(self.root, fn)


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
        Field.index_field("tid", dtype=float),  # Replace to int when we don't have NaN
        Field("mode", dtype=str),
        Field("stacktrace", dtype=object),
        Field.derived_field("nsamples", dtype=int),
    ]

    def __init__(self, benchmark, config):
        super().__init__(benchmark, config)
        # Maximum stack depth for this dataset
        self.max_stacktrace_depth = None

    def _resolve_pids(self, df):
        """
        Resolve PIDs to process names. In case of failure, use the pid as a string.
        """
        pidmap = self.benchmark.pidmap  #get_dataset_by_artefact(DatasetArtefact.PIDMAP)
        assert pidmap is not None, "The pidmap dataset is required for pmc sampling"

        cmd_notid = pidmap.df.groupby(pidmap.dataset_id_columns() + ["iteration", "pid"]).first()["command"]
        join_df = df.join(cmd_notid)
        assert len(join_df) == len(df)
        df["process"] = join_df["command"]
        df = join_df.rename(columns={"command": "process"})
        if df["process"].isna().any():
            missing = df["process"].isna()
            self.logger.warning("Missing pids %s", df.loc[df["process"].isna()].index.get_level_values("pid").unique())
            df["process"] = df.loc[:, "process"].mask(df["process"].isna(), None)
        return df

    def iteration_output_guestfile(self, iteration):
        return super().iteration_output_file(iteration).with_suffix(".pmc")

    def iteration_output_file(self, iteration):
        return super().iteration_output_file(iteration).with_suffix(".json")

    def stacktrace_columns(self):
        assert self.max_stacktrace_depth is not None
        return [f"stacktrace_{i}" for i in range(self.max_stacktrace_depth)]

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
        # Update max depth
        self.max_stacktrace_depth = self.df["stacktrace"].map(len).max()

    def pre_merge(self):
        super().pre_merge()
        sym_columns = self.stacktrace_columns()

        # Flatten stacks into the dataframe
        df_stacks = self.df.copy()

        def normalize_value(v):
            if type(v) == str:
                return np.uint64(int(v, 16))
            elif type(v) == int:
                return np.uint64(v)
            else:
                return 0

        for i in range(self.max_stacktrace_depth):
            df_stacks[f"stacktrace_{i}"] = self.df["stacktrace"].map(lambda st: normalize_value(st[-(i + 1)])
                                                                     if len(st) > i else 0)
        # Resolve PID mappings
        df_stacks = self._resolve_pids(df_stacks)
        # Translate addresses in-place, this could be done faster with a join()
        with timing("Symbolize stacks", logger=self.logger):
            grouped = df_stacks.groupby(["dataset_id", "process"])

            def resolve_syms(group):
                as_key = group.name[1]  # process
                # Would be better to use a proper NULL entry insted of unknown
                return group.applymap(lambda v: self.benchmark.sym_resolver.lookup_function(as_key, v)
                                      if v else SymInfo.unknown(-1))

            result = grouped[sym_columns].apply(resolve_syms)
            df_stacks[sym_columns] = result
        # XXX remove: support older traces that lack tid
        if df_stacks.index.unique("tid").isna().all():
            df_stacks = df_stacks.droplevel("tid")
        # This operation should not have resulted into a frame resize
        assert len(df_stacks) == len(self.df), "Call stacks dataframe size changed"
        # Count number of samples for each stack trace
        stacks_nsamples = df_stacks.groupby(df_stacks.index.names + sym_columns).size()
        stacks_nsamples.name = "nsamples"
        self.df = stacks_nsamples.reset_index(sym_columns)
        # Check frame structure before continuing
        assert "nsamples" in self.df.columns, "Missing nsamples column"

    def merge(self, other):
        super().merge(other)
        self.max_stacktrace_depth = max(self.max_stacktrace_depth, other.max_stacktrace_depth)

    def aggregate(self):
        super().aggregate()
        # Ensure that we don't have NaN in the stack columns
        clean_df = self.merged_df.fillna(SymInfo.unknown(-1))
        grouped = clean_df.groupby(self.dataset_id_columns() + self.stacktrace_columns())
        self.agg_df = self._compute_aggregations(grouped)

    def post_aggregate(self):
        super().post_aggregate()
        # align the aggregated stack counts
        aligned_df = align_multi_index_levels(self.agg_df, self.stacktrace_columns(), fill_value=0)
        delta_df = self._add_delta_columns(aligned_df)
        delta_df = self._compute_delta_by_dataset(delta_df)
        self.agg_df = delta_df
