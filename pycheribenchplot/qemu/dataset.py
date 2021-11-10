import logging
import re
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.dataset import (IndexField, DataField, DerivedField, Field, DatasetID, align_multi_index_levels,
                            rotate_multi_index_level, subset_xs, check_multi_index_aligned, DatasetProcessingException)
from ..core.instance import PlatformOptions
from ..core.perfetto import PerfettoDataSetContainer


class QEMUTracingCtxDataset(PerfettoDataSetContainer):
    """
    Helper dataset mostly useful for debugging purposes.
    This records tracing slice periods associated to the relative contexts.
    """
    dataset_id = "qemu-ctx-tracks"
    fields = []

    def raw_fields(self, include_derived=False):
        fields = super().raw_fields(include_derived)
        fields += [f for f in QEMUTracingCtxDataset.fields if include_derived or not f.isderived]
        return fields

    def _extract_events(self, tp: "TraceProcessor"):
        super()._extract_events(tp)
        tracks = self._query_to_df(tp, "SELECT * FROM track")
        tracing_ctrl = self._query_to_df(tp, "SELECT ts, dur FROM slice WHERE category = 'ctrl'")
        sched_ctrl = self._query_to_df(
            tp, "SELECT ts, flat_key, int_value FROM slice JOIN args ON " +
            "slice.arg_set_id = args.arg_set_id WHERE category = 'sched'")
        # ts|pid|tid|cid|evtname sorted by time


class ContextStatsHistogramBase(PerfettoDataSetContainer):
    """
    Base class that handles loading QEMU stats tracks from the perfetto backend, and attempts to resolve
    the context to which data records are associated.
    This delegates the actual data aggregation strategy to subclasses.
    """

    fields = [
        IndexField("histogram", dtype=int),
        IndexField("bucket", dtype=int),
        IndexField("file", dtype=str, isderived=True),
        IndexField("symbol", dtype=str, isderived=True),
        IndexField("process", dtype=str, isderived=True),
        IndexField("pid", dtype=int),
        IndexField("tid", dtype=int),
        IndexField("cid", dtype=int),
        IndexField("EL", dtype=int),
        # IndexField("AS", dtype=int),
        Field("start", dtype=np.uint),
        DerivedField("valid_symbol", dtype=object, isdata=False)
    ]

    def raw_fields(self, include_derived=False):
        fields = super().raw_fields(include_derived)
        fields += [f for f in ContextStatsHistogramBase.fields if include_derived or not f.isderived]
        return fields

    def _get_slice_name(self) -> str:
        raise NotImplementedError("Subclass must specialize")

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

    def _get_arg_key_map(self):
        """
        Get mapping of argument flat_key to destination column in the imported dataframe
        """
        return {"qemu.histogram.bucket.start": "start"}

    def _extract_track_stats(self, tp, track):
        """
        Build subsection of the dataframe pertaining to a given track.
        Note that we are expected to return a dataframe with a flat index as this happens before
        assigning the final index.
        """
        slice_name = self._get_slice_name()
        track_info = self._decode_track_name(track["name"])
        data_cols = {}
        for query_col, mapped_col in self._get_arg_key_map().items():
            query_str = (f"SELECT slice.id, int_value FROM slice JOIN args ON slice.arg_set_id = args.arg_set_id " +
                         f"WHERE slice.category = 'stats' AND slice.name = '{slice_name}' AND " +
                         f"args.flat_key = '{query_col}' AND slice.track_id = {track.id} ORDER BY args.id")
            mapped_col_data = self._query_to_df(tp, query_str)
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

    def _extract_events(self, tp: "TraceProcessor"):
        super()._extract_events(tp)
        slice_name = self._get_slice_name()
        # Look up tracks in the datasets that have data slices we are interested in
        tracks_query_str = (
            "SELECT DISTINCT(track.id), track.name FROM track JOIN slice ON track.id = slice.track_id " +
            "JOIN args ON slice.arg_set_id = args.arg_set_id WHERE args.flat_key LIKE 'qemu%' AND " +
            f"slice.category = 'stats' AND slice.name = '{slice_name}'")
        tracks = self._query_to_df(tp, tracks_query_str)
        # Dataframe skeleton with all non-derived columns (no index)
        df = pd.DataFrame(columns=self.all_columns())
        for idx, track in tracks.iterrows():
            self.logger.debug("Detected track %s: extracting stats", track["name"])
            track_df = self._extract_track_stats(tp, track)
            track_df["__dataset_id"] = self.benchmark.uuid
            track_df = track_df.astype(self._get_column_dtypes(include_converted=True))
            track_df.set_index(self.index_columns(), inplace=True)
            self.df = pd.concat([self.df, track_df])

    def pre_merge(self):
        """
        Resolve symbols to mach each entry to the function containing it.
        We update the raw-data dataframe with a new column accordingly.
        """
        super().pre_merge()
        resolver = self.benchmark.sym_resolver
        # Assign commands to PIDs
        pidmap = self.benchmark.get_dataset(DatasetID.PIDMAP)
        assert pidmap is not None
        pid_df = pidmap.df.set_index("pid").add_suffix("_pidmap")  # Avoid accidental column clash in join
        pids = self.df.join(pid_df, how="left", on="pid")["command_pidmap"]
        # There may be some NaN due to PID that were running during the benchmark but have since been terminated
        # We mark these as 'undetected'
        self.df["process"] = pids.fillna("undetected")

        # Resolve file:symbol for each address so that we can aggregate counts for each one of them
        resolved = self.df.apply(lambda row: resolver.lookup_fn(row["start"], Path(row["process"]).name), axis=1)
        self.df["valid_symbol"] = "ok"
        self.df.loc[resolved.isna(), "valid_symbol"] = "no-match"
        # XXX-AM: the symbol size does not appear to be reliable?
        # sym_end = resolved.map(lambda syminfo: syminfo.addr + syminfo.size if syminfo else np.nan)
        # size_mismatch = (~sym_end.isna()) & (self.df["start"] > sym_end)
        # self.df.loc[size_mismatch, "valid_symbol"] = "size-mismatch"

        invalid_syms = self.df["valid_symbol"] != "ok"
        self.df["symbol"] = resolved[~invalid_syms].map(lambda syminfo: syminfo.name)
        # self.df["symbol"] = resolved.map(lambda syminfo: syminfo.name if syminfo else None)
        self.df.loc[invalid_syms, "symbol"] = self.df.loc[invalid_syms, "start"].transform(lambda addr: f"0x{addr:x}")
        # Note: For the file name, we omit the directory part as otherwise the same executable
        # in different directories will be picked up as a completely different file. This is
        # not useful when comparing different compilations that have different paths e.g. the kernel
        # We also have to handle rtld manually to map its name.
        self.df["file"] = resolved[~invalid_syms].map(lambda syminfo: syminfo.filepath.name)
        self.df.loc[invalid_syms, "file"] = "unknown"

    def _get_agg_strategy(self):
        """Return mapping of column-name => aggregation function for the columns we need to aggregate"""
        agg = {"start": "min", "valid_symbol": lambda vec: ",".join(vec.unique())}
        agg.update({col: "sum" for col in self.delta_columns()})
        return agg

    def aggregate(self):
        super().aggregate()
        grouped = self.merged_df.groupby(["__dataset_id", "process", "EL", "file", "symbol"])
        self.agg_df = grouped.agg(self._get_agg_strategy())

    def delta_columns(self):
        return self.data_columns()

    def post_aggregate(self):
        super().post_aggregate()
        # Align dataframe on the (file, symbol) pairs where we want to get an union of
        # the symbols set for each file, repeated for each dataset_id.
        align_levels = ["process", "EL", "file", "symbol"]
        new_df = align_multi_index_levels(self.agg_df, align_levels, fill_value=0)
        # Backfill after alignment
        new_df.loc[new_df["valid_symbol"] == 0, "valid_symbol"] = "missing"
        # Now we can safely assign as there are no missing values.
        # 1) Compute delta for each metric for each function w.r.t. the baseline.
        # 2) Build the normalized delta for each metric, note that this will generate infinities where
        #   the baseline benchmark count is 0 (e.g. extra functions called only in other samples)
        baseline = new_df.xs(self.benchmark.uuid, level="__dataset_id")
        datasets = new_df.index.get_level_values("__dataset_id").unique()
        base_cols = self.delta_columns()
        delta_cols = [f"delta_{col}" for col in base_cols]
        norm_cols = [f"norm_{col}" for col in delta_cols]
        new_df[delta_cols] = 0
        new_df[norm_cols] = 0
        # XXX could avoid this loop by repeating N times the baseline values and vectorize the division?
        for ds_id in datasets:
            other = new_df.xs(ds_id, level="__dataset_id")
            delta = other[base_cols].subtract(baseline[base_cols])
            norm_delta = delta[base_cols].divide(baseline[base_cols])
            new_df.loc[ds_id, delta_cols] = delta[base_cols].values
            new_df.loc[ds_id, norm_cols] = norm_delta[base_cols].values
        assert not new_df["valid_symbol"].isna().any()
        self.agg_df = new_df

    def configure(self, opts: PlatformOptions) -> PlatformOptions:
        opts = super().configure(opts)
        opts.qemu_trace = True
        opts.qemu_trace_file = self.output_file()
        opts.qemu_trace_categories.add("ctrl")
        opts.qemu_trace_categories.add("stats")
        return opts

    def output_file(self):
        # This must be shared by all the histogram datasets
        return self.benchmark.result_path / f"qemu-stats-{self.benchmark.uuid}.pb"


class QEMUStatsBBHistogramDataset(ContextStatsHistogramBase):
    """Basic-block hit count histogram"""
    dataset_id = "qemu-stats-bb"
    fields = [
        Field("end", dtype=np.uint),
        DataField("bb_count", dtype=int),
        DerivedField("delta_bb_count", dtype=int),
        DerivedField("norm_delta_bb_count", dtype=float)
    ]

    def raw_fields(self, include_derived=False):
        fields = super().raw_fields(include_derived)
        fields += [f for f in QEMUStatsBBHistogramDataset.fields if include_derived or not f.isderived]
        return fields

    def _get_arg_key_map(self):
        mapping = super()._get_arg_key_map()
        mapping.update({"qemu.histogram.bucket.end": "end", "qemu.histogram.bucket.value": "bb_count"})
        return mapping

    def _get_slice_name(self):
        return "bb_hist"

    def _get_agg_strategy(self):
        agg = super()._get_agg_strategy()
        agg["end"] = "max"
        return agg


class QEMUStatsBranchHistogramDataset(ContextStatsHistogramBase):
    """Branch instruction target hit count histogram"""
    dataset_id = "qemu-stats-call"
    fields = [
        DataField("branch_count", dtype=int),
        DerivedField("call_count", dtype=int),
        DerivedField("delta_call_count", dtype=int),
        DerivedField("norm_delta_call_count", dtype=float)
    ]

    def _get_slice_name(self):
        return "branch_hist"

    def raw_fields(self, include_derived=False):
        fields = super().raw_fields(include_derived)
        fields += [f for f in QEMUStatsBranchHistogramDataset.fields if include_derived or not f.isderived]
        return fields

    def _get_arg_key_map(self):
        mapping = super()._get_arg_key_map()
        mapping.update({"qemu.histogram.bucket.value": "branch_count"})
        return mapping

    def pre_merge(self):
        """
        Resolve symbols to mach each entry to the function containing it.
        We update the raw-data dataframe with a new column accordingly.
        """
        super().pre_merge()
        resolver = self.benchmark.sym_resolver
        # Generate now a new column only for entries that exactly match symbols, meaning that
        # these are function calls is the first basic-block of the function and is considered as an individual call
        # to that function
        is_call = self.df.apply(lambda row: resolver.match_fn(row["start"],
                                                              Path(row["process"]).name) is not None,
                                axis=1)
        self.df["call_count"] = self.df["branch_count"].mask(~is_call, 0).astype(int)

    def delta_columns(self):
        cols = super().delta_columns()
        return cols + ["call_count"]
