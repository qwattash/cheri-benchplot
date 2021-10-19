import logging
import re
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.dataset import (IndexField, DataField, DerivedField, Field, align_multi_index_levels,
                            rotate_multi_index_level, subset_xs, check_multi_index_aligned, DatasetProcessingException)
from ..core.perfetto import PerfettoDataSetContainer
from ..core.util import timing


class QEMUTracingCtxDataset(PerfettoDataSetContainer):
    """
    Helper dataset mostly useful for debugging purposes.
    This records tracing slice periods associated to the relative contexts.
    """
    fields = []

    def raw_fields(self, include_derived=False):
        fields = super().raw_fields(include_derived)
        fields += [f for f in QEMUTracingCtxDataset.fields if include_derived or not f.isderived]
        return fields

    def _extract_events(self, tp: "TraceProcessor"):
        super()._extract_events()


class QEMUStatsHistogramDataset(PerfettoDataSetContainer):
    fields = [
        IndexField("bucket", dtype=int),
        IndexField("file", dtype=str, isderived=True),
        IndexField("symbol", dtype=str, isderived=True),
        IndexField("ctx_pid", dtype=int),
        IndexField("ctx_tid", dtype=int),
        IndexField("ctx_cid", dtype=int),
        IndexField("ctx_EL", dtype=int),
        IndexField("ctx_AS", dtype=int),
        Field("start", dtype=np.uint),
        DerivedField("valid_symbol", dtype=object, isdata=False)
    ]

    def raw_fields(self, include_derived=False):
        fields = super().raw_fields(include_derived)
        fields += [f for f in QEMUStatsHistogramDataset.fields if include_derived or not f.isderived]
        return fields

    def _get_slice_name(self) -> str:
        raise NotImplementedError("Subclass must specialize")

    def _extract_events(self, tp: "TraceProcessor"):
        super()._extract_events(tp)
        slice_name = self._get_slice_name()
        query_str = (f"SELECT * FROM slice INNER JOIN args ON " + "slice.arg_set_id = args.arg_set_id WHERE " +
                     f"slice.category = 'stats' AND slice.name = '{slice_name}' AND " + "args.flat_key LIKE 'qemu%'")
        # query_str = (f"SELECT * FROM track INNER JOIN slice ON track.id = slice.track_id " +
        #              f"INNER JOIN args ON slice.arg_set_id = args.arg_set_id " +
        #              f"WHERE slice.cat = 'stats' AND slice.name = '{slice_name}' AND " +
        #              f"args.flat_key LIKE 'qemu%'")
        with timing("Extract qemu stats events query", logging.DEBUG, self.logger):
            result = tp.query(query_str)
        # XXX-AM: This is unreasonably slow, build the dataframe manually for now
        # df = result.as_pandas_dataframe()
        query_df = pd.DataFrame.from_records(map(lambda row: row.__dict__, result))
        df = self._build_df(query_df)
        # Append dataframe to dataset
        self.df = pd.concat([self.df, df])

    def _build_df(self, input_df: pd.DataFrame):
        """
        Convert the input dataframe into the dataset dataframe
        This involves mappint the values of the key column into columns
        qemu.histogram.bucket[n].start -> start
        qemu.histogram.bucket[n].end -> end
        qemu.histogram.bucket[n].value -> count
        """
        # Check that the dataframe matches the expected format
        assert input_df["key"].apply(lambda k: k.startswith("qemu.histogram.bucket[")).all(
        ), "Malformed input perfetto dataframe: expected all argument keys as qemu.histogram.bucket[n].<field>"
        # First assign the bucket index as a column (a bit sad to use regex)
        extractor = re.compile("\[([0-9]+)\]")
        input_df["bucket"] = input_df["key"].apply(lambda k: int(extractor.search(k).group(1)))
        # Make sure the index is now aligned for all bucket field groups
        input_df.set_index(["arg_set_id", "bucket"], inplace=True)
        # Rotate the keys into columns using the bucket index as row index
        tmp_df = pd.DataFrame(columns=input_df["flat_key"].unique())
        for c in tmp_df.columns:
            # Assume all bucket fields are int_values (int or uint)
            tmp_df[c] = input_df[input_df["flat_key"] == c]["int_value"]
        # Fixup column names
        tmp_df = tmp_df.rename(columns=self.field_names_map())
        # Fill extra columns
        tmp_df["__dataset_id"] = self.benchmark.uuid
        # XXX-AM: For now initialize unimplemented index fields with empty stuff
        tmp_df["ctx_pid"] = 0
        tmp_df["ctx_tid"] = 0
        tmp_df["ctx_cid"] = 0
        tmp_df["ctx_EL"] = 0
        tmp_df["ctx_AS"] = 0
        # Normalize fields
        tmp_df = tmp_df.reset_index(drop=False).astype(self._get_column_dtypes(include_converted=True))
        tmp_df.set_index(self.index_columns(), inplace=True)
        return tmp_df

    def pre_merge(self):
        """
        Resolve symbols to mach each entry to the function containing it.
        We update the raw-data dataframe with a new column accordingly.
        """
        super().pre_merge()
        resolver = self.benchmark.sym_resolver
        self.df["valid_symbol"] = "ok"
        resolved = self.df["start"].map(lambda addr: resolver.lookup_fn(addr))
        self.df.loc[resolved.isna(), "valid_symbol"] = "no-match"
        # XXX-AM: the symbol size does not appear to be reliable?
        # sym_end = resolved.map(lambda syminfo: syminfo.addr + syminfo.size if syminfo else np.nan)
        # size_mismatch = (~sym_end.isna()) & (self.df["start"] > sym_end)
        # self.df.loc[size_mismatch, "valid_symbol"] = "size-mismatch"

        invalid_syms = self.df["valid_symbol"] != "ok"
        self.df["symbol"] = resolved.map(lambda syminfo: syminfo.name if syminfo else None)
        self.df.loc[invalid_syms, "symbol"] = self.df.loc[invalid_syms, "start"].transform(lambda addr: f"0x{addr:x}")
        # Note: For the file name, we omit the directory part as otherwise the same executable
        # in different directories will be picked up as a completely different file. This is
        # not useful when comparing different compilations that have different paths e.g. the kernel
        # We also have to handle rtld manually to map its name.
        self.df["file"] = resolved.map(lambda syminfo: syminfo.filepath.name if syminfo else None)
        self.df.loc[invalid_syms, "file"] = "unknown"
        self.df.set_index(["file", "symbol"], append=True)

    def _get_agg_strategy(self):
        """Return mapping of column-name => aggregation function for the columns we need to aggregate"""
        agg = {"start": "min", "valid_symbol": lambda vec: ",".join(vec.unique())}
        agg.update({col: "sum" for col in self.delta_columns()})
        return agg

    def aggregate(self):
        super().aggregate()
        tmp = self.merged_df.set_index(["file", "symbol"], append=True)
        grouped = tmp.groupby(["__dataset_id", "file", "symbol"])
        self.agg_df = grouped.agg(self._get_agg_strategy())

    def delta_columns(self):
        return self.data_columns()

    def post_aggregate(self):
        super().post_aggregate()
        # Align dataframe on the (file, symbol) pairs where we want to get an union of
        # the symbols set for each file, repeated for each dataset_id.
        new_df = align_multi_index_levels(self.agg_df, ["file", "symbol"], fill_value=0)
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


class QEMUStatsBBHistogramDataset(QEMUStatsHistogramDataset):
    """Basic-block hit count histogram"""
    dataset_id = "qemu-stats-bb"
    fields = [
        Field("end", dtype=np.uint),
        DataField("bb_count", dtype=int),
        DerivedField("bb_bytes", dtype=int),
        DerivedField("delta_bb_count", dtype=int),
        DerivedField("norm_delta_bb_count", dtype=float)
    ]

    def raw_fields(self, include_derived=False):
        fields = super().raw_fields(include_derived)
        fields += [f for f in QEMUStatsBBHistogramDataset.fields if include_derived or not f.isderived]
        return fields

    def field_names_map(self):
        return {
            "qemu.histogram.bucket.start": "start",
            "qemu.histogram.bucket.end": "end",
            "qemu.histogram.bucket.value": "bb_count"
        }

    def _get_slice_name(self):
        return "bb_hist"

    def pre_merge(self):
        super().pre_merge()
        # Generate number of bytes hit for each range of start/end addresses as a proxy for
        # real instruction count. The real number will be some fraction of this number, depending
        # on instruction size.
        self.df["bb_bytes"] = (self.df["end"] - self.df["start"]) * self.df["bb_count"]

    def _get_agg_strategy(self):
        agg = super()._get_agg_strategy()
        agg["end"] = "max"
        agg["bb_bytes"] = "sum"
        return agg


class QEMUStatsBranchHistogramDataset(QEMUStatsHistogramDataset):
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

    def field_names_map(self):
        return {"qemu.histogram.bucket.start": "start", "qemu.histogram.bucket.value": "branch_count"}

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
        is_call = self.df["start"].map(lambda addr: resolver.lookup_fn_exact(addr) is not None)
        self.df["call_count"] = self.df["branch_count"].mask(~is_call, 0).astype(int)

    def delta_columns(self):
        cols = super().delta_columns()
        return cols + ["call_count"]
