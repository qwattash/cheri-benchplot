import logging
import re
import typing
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from pypika import Order, Query

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
        Field.index_field("histogram", dtype=int),
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

    def _get_slice_name(self) -> str:
        """Get the name of the slices containing histogram data for this dataset"""
        raise NotImplementedError("Must override")

    def _get_arg_key_map(self) -> typing.Dict[str, str]:
        """Get mapping of argument flat_key to destination column in the imported dataframe"""
        raise NotImplementedError("Must override")

    def _get_symbol_column(self):
        """Return the colum to use to populate the file/symbol indexes"""
        raise NotImplementedError("Must override")

    def delta_columns(self):
        """Return columns for which to compute derived delta columns"""
        raise NotImplementedError("Must override")

    def _decode_track_name(self, track_name: str):
        """
        XXX-AM: we should have a table for contexts as we have one for processes and threads.
        """
        track_info = {"pid": -1, "tid": -1, "cid": -1, "EL": -1}
        if track_name.startswith("CTX"):
            parts = track_name.split(":")
            assert len(parts) == 4, "Malformed context track name"
            pid = int(parts[0][len("CTX "):])
            track_info.update({"pid": pid, "tid": int(parts[1]), "cid": int(parts[2]), "EL": int(parts[3])})
        return track_info

    def _extract_track_stats(self, tp, track, ts_start, ts_end):
        """
        Build subsection of the dataframe pertaining to a given track.
        Note that we are expected to return a dataframe with a flat index as this happens before
        assigning the final index.
        """
        slice_name = self._get_slice_name()
        track_info = self._decode_track_name(track["name"])
        data_cols = {}
        for query_col, mapped_col in self._get_arg_key_map().items():
            query = self._query_slice_ts(ts_start, ts_end)
            query = query.select(self.t_slice.id, self.t_args.int_value)
            query = query.where((self.t_slice.category == "stats") & (self.t_slice.name == slice_name)
                                & (self.t_args.flat_key == query_col) & (self.t_slice.track_id == track.id)).orderby(
                                    self.t_args.id, order=Order.asc)
            mapped_col_data = self._query_to_df(tp, query.get_sql(quote_char=None))
            mapped_col_data.rename(columns={"id": "histogram"}, inplace=True)
            mapped_col_data.index.name = "bucket"  # Current index in just row count ordered by args.id
            mapped_col_data.set_index("histogram", append=True, inplace=True)
            # Note it is important that the data_cols are Series objects otherwise we will end up
            # with a multi-level column index in the concatenation result
            data_cols[mapped_col] = mapped_col_data["int_value"]
        df = pd.concat(data_cols, axis=1)
        for key, value in track_info.items():
            df[key] = value
        return df.reset_index(drop=False)

    def _extract_events(self, tp: "TraceProcessor", i: int, ts_start: int, ts_end: int):
        """
        Extract events from an iteration of the benchmark.
        i: iteration index
        ts_start: iteration start timestamp
        ts_end: iteration end timestamp
        """
        slice_name = self._get_slice_name()
        # Look up tracks in the datasets that have data slices we are interested in
        query = self._query_slice_ts(ts_start, ts_end).join(self.t_track).on(self.t_track.id == self.t_slice.track_id)
        query = query.where(
            self.t_args.flat_key.like("qemu%") & (self.t_slice.category == "stats")
            & (self.t_slice.name == slice_name)).select(self.t_track.id, self.t_track.name).distinct()
        tracks = self._query_to_df(tp, query.get_sql(quote_char=None))
        # Dataframe skeleton with all non-derived columns (no index)
        df = pd.DataFrame(columns=self.all_columns())
        for idx, track in tracks.iterrows():
            self.logger.debug("Detected track %s: extracting stats", track["name"])
            track_df = self._extract_track_stats(tp, track, ts_start, ts_end)
            track_df["dataset_id"] = self.benchmark.uuid
            track_df["iteration"] = i
            self._append_df(track_df)

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
        Resolve symbols given a column containing addresses.
        The addrspace_key is the column providing the address-space name for the symbol resolver.
        This will not change the dataframe, instead returns a dataframe with the same index as
        the main dataframe, with columns "file", "symbol", "valid_symbol" containing the mapped file/symbol
        and the status result of the mapping process
        """
        resolver = self.benchmark.sym_resolver
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
        iterations = self._extract_iteration_markers(tp)
        # Verify that the iteration markers agree with the configured number of iterations
        if len(iterations) != self.benchmark.config.iterations:
            self.logger.error("QEMU trace does not have the expected iteration markers: %d configured %d",
                              len(iterations), self.benchmark.config.iterations)
            raise DatasetProcessingError("QEMU trace has invalid iteration markers")
        for i, interval in enumerate(iterations):
            start, end = interval
            self._extract_events(tp, i, start, end)

    def pre_merge(self):
        """
        Common pre-merge resolves the file/symbol and valid symbol column based on
        the column name given by _get_symbol_column()
        """
        super().pre_merge()
        self._resolve_pid_tid()
        self.df["process_name"] = self.df["process"].map(lambda p: Path(p).name)
        sym_col = self._get_symbol_column()
        resolved = self._resolve_sym_column(sym_col, "process_name")
        # Populate the file, symbol and valid_symbol columns
        self.df = pd.concat([self.df, resolved], axis=1)

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
    fields = [Field("start", dtype=np.uint), Field("end", dtype=np.uint), Field.data_field("icount", dtype=int)]

    def _get_slice_name(self):
        return "bb_hist"

    def _get_arg_key_map(self):
        mapping = {
            "qemu.histogram.bb_bucket.start": "start",
            "qemu.histogram.bb_bucket.end": "end",
            "qemu.histogram.bb_bucket.value": "icount"
        }
        return mapping

    def _get_symbol_column(self):
        return "start"

    def _get_context_agg_strategy(self):
        agg = super()._get_context_agg_strategy()
        agg.update({"start": "min", "end": "max", "icount": "sum"})
        return agg


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
        cnt_df = self._query_to_df(tp, cnt_query.get_sql(quote_char=None))

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
        iterations = self._extract_iteration_markers(tp)
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
