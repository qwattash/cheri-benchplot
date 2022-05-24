import logging
import re
import typing
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from pypika import Order, Query, Table
from sortedcontainers import SortedList

from ..core.dataset import (DatasetArtefact, DatasetName, DatasetProcessingError, Field, align_multi_index_levels,
                            check_multi_index_aligned, rotate_multi_index_level, subset_xs)
from ..core.instance import PlatformOptions
from ..core.perfetto import PerfettoDataSetContainer


class QEMUTraceDataset(PerfettoDataSetContainer):
    """
    Base class for all datasets requiring qemu-perfetto trace output.
    This initializes the qemu instance configuration so that all the datasets will point to
    the same qemu output file. Subclasses should enable their own trace categories.
    """
    def output_file(self):
        return self.benchmark.get_output_path() / f"qemu-perfetto-{self.benchmark.uuid}.pb"

    def configure(self, opts: PlatformOptions) -> PlatformOptions:
        opts = super().configure(opts)
        opts.qemu_trace = True
        opts.qemu_trace_file = self.output_file()
        opts.qemu_trace_categories.add("ctrl")
        return opts


class ContextStatsHistogramBase(QEMUTraceDataset):
    """
    Base class that handles loading QEMU stats tracks from the perfetto backend, and attempts to resolve
    the context to which data records are associated.
    This delegates the actual data aggregation strategy to subclasses.
    """
    dataset_source_id = DatasetArtefact.QEMU_STATS
    fields = [
        Field.index_field("bucket", dtype=int),
        Field.index_field("file", dtype=str, isderived=True),
        Field.index_field("symbol", dtype=str, isderived=True),
        Field.index_field("process", dtype=str, isderived=True),
        Field.index_field("thread", dtype=str, isderived=True),
        Field.index_field("pid", dtype=int),
        Field.index_field("tid", dtype=int),
        Field.index_field("cid", dtype=int),
        Field.index_field("EL", dtype=int),
        # Field.index_field("AS", dtype=int),
        Field.derived_field("valid_symbol", dtype=object, isdata=False)
    ]

    def extract_context_tracks(self, tp: "TraceProcessor", query: Query = None):
        """
        Extract all CHERI context tracks from the trace into a temporary dataframe.
        """
        t_process = Table("process")
        t_thread = Table("thread")
        t_comp = Table("compartment")
        t_cheri_context_track = Table("cheri_context_track")
        if query is None:
            query = Query.from_(t_cheri_context_track).join(t_process).on_field("upid")
        query = query.join(self.t_thread).on_field("utid")
        query = query.join(self.t_compartment).on_field("ucid")
        query = query.select(self.t_process.pid, self.t_thread.tid, self.t_compartment.cid, self.t_compartment.el,
                             self.t_cheri_context_track.id)
        df = self._query_to_df(tp, query)
        return df

    def extract_counters(self, tp: "TraceProcessor", track: int, ts_start: float, ts_end: float):
        pass

    def extract_slices(self):
        pass

    def delta_columns(self):
        """Return columns for which to compute derived delta columns"""
        raise NotImplementedError("Must override")

    def _resolve_pid_tid(self):
        """
        Resolve process and thread names
        """
        pidmap = self.benchmark.get_dataset_by_artefact(DatasetArtefact.PIDMAP)
        assert pidmap is not None, "The pidmap dataset is required for qemu stats"
        # Find missing TIDs in pidmap and attempt to match them with the QEMU data
        detected = self.df.groupby(["pid", "tid"]).size().reset_index()[["pid", "tid"]]
        # Suffix is added to avoid clash during join
        pid_df = pidmap.fixup_missing_tid(detected)
        pid_df = pid_df.set_index(["pid", "tid"]).add_suffix("_pidmap")
        join_df = self.df.join(pid_df, how="left", on=["pid", "tid"])
        # There may be some NaN due to PID that were running during the benchmark but have since been terminated
        # We mark these as undetected
        na_cmd = join_df["command_pidmap"].isna()
        na_thr = join_df["thread_name_pidmap"].isna()
        join_df.loc[na_cmd, "command_pidmap"] = join_df.index.get_level_values("pid")[na_cmd].map(
            lambda pid: f"undetected:{pid}")
        join_df.loc[na_thr, "thread_name_pidmap"] = join_df.index.get_level_values("tid")[na_thr].map(
            lambda tid: f"unknown:{tid}")
        self.df["process"] = join_df["command_pidmap"]
        self.df["thread"] = join_df["thread_name_pidmap"]

    def _resolve_sym_column(self, col: str, addrspace_key: str) -> pd.DataFrame:
        """
        Resolve symbols/source-level information given a column containing addresses.
        The addrspace_key is the column providing the address-space name for the symbol resolver.
        This will not change the dataframe, instead returns a dataframe with the same index as
        the main dataframe, with columns "file", "symbol", "valid_symbol" containing the mapped file/symbol
        and the status result of the mapping process
        """
        resolver = self.benchmark.sym_resolver
        dwarf = self.benchmark.dwarf_helper
        # XXX-AM: the symbol size does not appear to be reliable?
        # otherwise we should check also the resolved syminfo size as in:
        # sym_end = resolved.map(lambda syminfo: syminfo.addr + syminfo.size if syminfo else np.nan)
        # size_mismatch = (~sym_end.isna()) & (self.df["start"] > sym_end)
        # self.df.loc[size_mismatch, "valid_symbol"] = "size-mismatch"
        resolved = self.df.apply(lambda row: resolver.lookup_fn(row[col], row[addrspace_key]), axis=1)
        resolved_df = pd.DataFrame(None, index=resolved.index)
        resolved_df["valid_symbol"] = resolved.mask(resolved.isna(), "no-match")
        resolved_df["valid_symbol"].where(resolved.isna(), "ok", inplace=True)

        resolved_df["symbol"] = resolved.map(lambda si: si.name, na_action="ignore")
        resolved_df["symbol"].mask(resolved.isna(), self.df[col].transform(lambda addr: f"0x{addr:x}"), inplace=True)
        # Note: For the file name, we omit the directory part as otherwise the same executable
        # in different directories will be picked up as a completely different file. This is
        # not useful when comparing different compilations that have different paths e.g. the kernel
        # TODO: We also have to handle rtld manually to map its name.
        resolved_df["file"] = resolved.map(lambda si: si.filepath.name, na_action="ignore")
        resolved_df["file"].mask(resolved.isna(), "unknown", inplace=True)
        return resolved_df

    def _get_context_agg_strategy(self):
        """Return mapping of column-name => aggregation function for the context data aggregation"""
        agg = {"valid_symbol": lambda vec: ",".join(vec.unique())}
        return agg

    def get_aggregate_index(self):
        """Index levels on which we aggregate"""
        return ["process", "thread", "EL", "file", "symbol"]

    def load(self):
        tp = self._get_trace_processor(self.output_file())
        iterations = self.extract_iteration_markers(tp)
        # Verify that the iteration markers agree with the configured number of iterations
        if len(iterations) != self.benchmark.config.iterations:
            self.logger.error("QEMU trace does not have the expected iteration markers: %d configured %d",
                              len(iterations), self.benchmark.config.iterations)
            raise DatasetProcessingError("QEMU trace has invalid iteration markers")
        for i, interval in enumerate(iterations):
            start, end = interval
            self._extract_iteration(tp, i, start, end)

    # def pre_merge(self):
    #     """
    #     Common pre-merge resolves the file/symbol and valid symbol column based on
    #     the column name given by _get_symbol_column()
    #     """
    #     super().pre_merge()
    #     self._resolve_pid_tid()
    #     self.df["process_name"] = self.df["process"].map(lambda p: Path(p).name)
    #     sym_col = self._get_symbol_column()
    #     resolved = self._resolve_sym_column(sym_col, "process_name")
    #     # Populate the file, symbol and valid_symbol columns
    #     self.df = pd.concat([self.df, resolved], axis=1)

    def aggregate(self):
        """
        Aggregation occurs in two steps:
        The first step aggregates per-context data for each iteration.
        The second step aggregates data across iterations.
        Customization should occur via get_aggregate_index(), _get_context_agg_strategy()
        and _get_aggregation_stragety().
        Note that we propagate some metadata fields to agg_df, we do not compute the
        aggregation metrics for these, instead we use the "meta" column name
        """
        super().aggregate()
        common_agg_index = self.dataset_id_columns() + self.get_aggregate_index()
        grouped = self.merged_df.groupby(common_agg_index + ["iteration"])
        # Cache the context grouping for later inspection if needed
        self.ctx_agg_df = grouped.aggregate(self._get_context_agg_strategy())
        # Now that we aggregated contexts within the same iteration, aggregate over iterations
        grouped = self.ctx_agg_df.groupby(common_agg_index)
        self.agg_df = self._compute_aggregations(grouped)

    def post_aggregate(self):
        super().post_aggregate()
        # Align dataframe on the (file, symbol) pairs where we want to get an union of
        # the symbols set for each file, repeated for each dataset_id.
        align_levels = self.get_aggregate_index()
        new_df = align_multi_index_levels(self.agg_df, align_levels, fill_value=0)
        agg_df = self._add_delta_columns(new_df)
        self.agg_df = self._compute_delta_by_dataset(agg_df)

    def configure(self, opts: PlatformOptions) -> PlatformOptions:
        opts = super().configure(opts)
        opts.qemu_trace_categories.add("stats")
        opts.qemu_trace_categories.add("marker")
        return opts


class QEMUStatsBBHistogramDataset(ContextStatsHistogramBase):
    """Basic-block hit count histogram"""
    dataset_config_name = DatasetName.QEMU_STATS_BB_HIST
    fields = [Field("start", dtype=np.uint), Field("end", dtype=np.uint), Field.data_field("hit_count", dtype=int)]

    def extract_intervals(self, tp: "TraceProcessor", ts_start: float, ts_end: float):
        """
        Extract intervals associated to a given track from the intervals table, within a
        timestamp interval.
        """
        t_interval = Table("interval")
        t_track = Table("cheri_context_interval_track")
        t_process = Table("process")
        t_thread = Table("thread")
        t_comp = Table("compartment")
        query = Query.from_(t_track).join(t_interval).on(t_interval.track_id == t_track.id)
        query = query.join(t_process).on_field("upid")
        query = query.join(t_thread).on_field("utid")
        query = query.join(t_comp).on_field("ucid")
        query = query.select(t_process.pid, t_thread.tid, t_comp.cid, t_comp.el, t_interval.ts, t_interval.start,
                             t_interval.end, t_interval.value)
        query = self._query_filter_ts(query, t_interval.ts, ts_start, ts_end)
        df = self._query_to_df(tp, query)
        # Merge overlapping intervals for each context
        grouped = df.groupby(["pid", "tid", "cid", "el"])
        df = grouped.apply(lambda g: self._merge_intervals(g))
        return df[df["value"] != 0].reset_index()

    def _merge_intervals(self, df):
        points = np.concatenate((df["start"], df["end"]))
        limits = sorted(np.unique(points))
        intervals = []
        for i in range(len(limits) - 1):
            start = limits[i]
            end = limits[i + 1]
            # select intervals that fully contain interval #i
            sel = (df["start"] <= start) & (df["end"] >= end)
            value = df[sel]["value"].sum()
            intervals.append((start, end, value))
        return pd.DataFrame.from_records(intervals, columns=["start", "end", "value"])

    def pre_merge(self):
        """
        Common pre-merge resolves the file/symbol/line and valid symbol column based on
        the PID and DWARF information
        """
        super().pre_merge()
        self._resolve_pid_tid()
        self.df["process_name"] = self.df["process"].map(lambda p: Path(p).name)
        resolved = self._resolve_sym_column("start", "process_name")
        # Populate the file, symbol and valid_symbol columns
        self.df = pd.concat([self.df, resolved], axis=1)

    def _get_context_agg_strategy(self):
        agg = super()._get_context_agg_strategy()
        agg.update({"start": "min", "end": "max", "hit_count": "sum"})
        return agg

    def _extract_iteration(self, tp, i, start, end):
        """
        Extract events from an iteration of the benchmark.
        """
        intervals_df = self.extract_intervals(tp, start, end)
        intervals_df.rename(columns={"value": "hit_count", "el": "EL"}, inplace=True)
        intervals_df["iteration"] = i
        # Append and add ID columns
        self._append_df(intervals_df)


class QEMUStatsBranchHistogramDataset(ContextStatsHistogramBase):
    """Branch instruction target hit count histogram"""
    dataset_config_name = DatasetName.QEMU_STATS_CALL_HIST
    fields = [
        Field.index_field("source_file", dtype=str, isderived=True),
        Field.index_field("source_symbol", dtype=str, isderived=True),
        Field("source", dtype=np.uint),
        Field("target", dtype=np.uint),
        Field("branch_count", dtype=int),
        Field.derived_field("call_count", dtype=int)
    ]

    def _get_slice_name(self):
        return "branch_hist"

    def _get_arg_key_map(self):
        mapping = {
            "qemu.histogram.branch_bucket.source": "source",
            "qemu.histogram.branch_bucket.target": "target",
            "qemu.histogram.branch_bucket.value": "branch_count"
        }
        return mapping

    def _get_symbol_column(self):
        return "target"

    def _get_context_agg_strategy(self):
        agg = super()._get_context_agg_strategy()
        agg.update({"source": "min", "target": "min", "call_count": "sum"})
        return agg

    def get_aggregate_index(self):
        cols = super().get_aggregate_index()
        return cols + ["source_file", "source_symbol"]

    def pre_merge(self):
        """
        Resolve symbols to mach each entry to the function containing it.
        We update the raw-data dataframe with a new column accordingly.
        """
        super().pre_merge()
        # Resolve the secondary file/symbol index for the source address
        resolved = self._resolve_sym_column("source", "process_name")
        resolved = resolved.add_prefix("source_")
        self.df = pd.concat((self.df, resolved), axis=1)
        # Generate now a new column only for entries that exactly match symbols, meaning that
        # these are function calls is the first basic-block of the function and is considered as an individual call
        # to that function
        resolver = self.benchmark.sym_resolver
        match = self.df.apply(lambda row: resolver.match_fn(row["target"], row["process_name"]), axis=1)
        is_call = ~match.isna()
        self.df["call_count"] = self.df["branch_count"].where(is_call, 0).astype(int)


class QEMUGuestCountersDataset(QEMUTraceDataset):
    """
    Dataset collecting global guest-driven qemu counters with the perfetto trace backend.
    This detects and separates counters samples in tracks for each dataset.
    """
    dataset_source_id = DatasetArtefact.QEMU_COUNTERS
    fields = [
        Field.index_field("ts", dtype=float),
        Field.index_field("name", dtype=str),
        Field.index_field("slot", dtype=int),
        Field.data_field("value", dtype=float),
    ]

    def _extract_counters(self, tp, iteration, ts_start, ts_end):
        """
        Extract counters during a single iteration
        """
        cnt_query = Query.from_(self.t_counter_track).join(
            self.t_counter).on(self.t_counter_track.id == self.t_counter.track_id)
        cnt_query = cnt_query.select(self.t_counter_track.name, self.t_counter.ts, self.t_counter.value)
        cnt_query = self._query_filter_ts(cnt_query, self.t_counter.ts, ts_start, ts_end)
        cnt_df = self._query_to_df(tp, cnt_query)

        name_split = cnt_df["name"].str.split(":")
        name = name_split.map(lambda v: v[0])
        slot = name_split.map(lambda v: v[1])
        cnt_df["name"] = name
        cnt_df["slot"] = slot
        cnt_df["dataset_id"] = self.benchmark.uuid
        cnt_df["iteration"] = iteration
        # Make timestamp relative to the beginning of the iteration
        cnt_df["ts"] = cnt_df["ts"] - ts_start
        self._append_df(cnt_df)

    def load(self):
        tp = self._get_trace_processor(self.output_file())
        iterations = self.extract_iteration_markers(tp)
        if len(iterations) != self.benchmark.config.iterations:
            self.logger.error("QEMU trace does not have the expected iteration markers: %d configured %d",
                              len(iterations), self.benchmark.config.iterations)
            raise DatasetProcessingError("QEMU trace has invalid iteration markers")
        for i, (start, end) in enumerate(iterations):
            self._extract_counters(tp, i, start, end)

    def configure(self, opts):
        opts = super().configure(opts)
        opts.qemu_trace_categories.add("counter")
        opts.qemu_trace_categories.add("marker")
        return opts
