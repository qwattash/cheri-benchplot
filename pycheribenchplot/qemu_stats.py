import logging
import re
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from .core.dataset import (IndexField, DataField, DerivedField, Field, align_multi_index_levels,
                           rotate_multi_index_level, subset_xs, check_multi_index_aligned, DatasetProcessingException)
from .core.perfetto import PerfettoDataSetContainer
from .core.plot import Plot, MatplotlibSurface, CellData, DataView
from .core.html import HTMLSurface


class QEMUAddrRangeHistBars(Plot):
    def __init__(self, benchmark, dataset):
        super().__init__(benchmark, dataset, MatplotlibSurface())

    def _get_plot_title(self):
        return "QEMU PC hit count"

    def _get_plot_file(self):
        return self.benchmark.manager_config.output_path / "qemu-pc-hist-bars.pdf"

    def prepare(self):
        """
        For each non-baseline dataset_id in the aggregate frame, we extract the
        file/function pair on the X axis and the count on the y axis.
        We plot the absolute count (normalized) and the diff count on two separate cells
        of the same surface.
        """
        super().prepare()
        baseline = self.benchmark.uuid
        df = self.dataset.agg_df.sort_values(by="diff", key=abs, ascending=False)
        self.surface.set_layout(3, 1)
        not_baseline = df.index.get_level_values("__dataset_id") != baseline
        df = df[not_baseline]
        df = df.reset_index(["file", "symbol"])
        df["file_sym"] = df["file"] + ":" + df["symbol"]
        df = df.set_index("file_sym", append=True)
        # Put everything into dataframes formatted for Surface.add_view
        abs_diff_df = df.rename(columns={"diff": "y_left"}).rename(index={"file_sym": "x"})
        self.surface.add_view("bar-group", abs_diff_df)
        # view_df = df.rename(columns={"norm_diff": "y_left"}).rename(index={"file_sym": "x"})


class QEMUAddrRangeHistTable(Plot):
    """
    Note: this only supports the HTML surface
    """
    def __init__(self, benchmark, dataset):
        super().__init__(benchmark, HTMLSurface())
        self.dataset = dataset

    def _get_plot_title(self):
        return "QEMU Stats"

    def _get_plot_file(self):
        return self.benchmark.manager_config.output_path / "qemu-pc-hist-table.html"

    def _get_legend_map(self):
        legend = {
            uuid: str(bench.instance_config.kernelabi)
            for uuid, bench in self.benchmark.merged_benchmarks.items()
        }
        legend[self.benchmark.uuid] = f"{self.benchmark.instance_config.kernelabi}(baseline)"
        return legend

    def prepare(self):
        """
        For each dataset (including the baseline) we show the dataframes as tables in
        an HTML page.
        """
        legend_map = self._get_legend_map()
        baseline = self.benchmark.uuid
        # Stacked plots on a single column
        self.surface.set_layout(1, 1, expand=True, how="row")
        # Make sure we copy it and do not mess with the original
        df = self.dataset.agg_df.copy()
        if not check_multi_index_aligned(df, "__dataset_id"):
            self.logger.error("Unaligned index, skipping plot")
            return

        data_cols = self.dataset.data_columns(include_derived=True)
        # Make normalized fields a percentage
        norm_cols = [col for col in data_cols if col.startswith("norm_")]
        df[norm_cols] = df[norm_cols] * 100
        # base data columns, displayed for both baseline and measure runs
        common_cols = self.dataset.data_columns()
        # derived data columns, displayed only for measure runs as they are comparative results
        measure_cols = list(set(data_cols) - set(common_cols))

        # Table for common functions
        view_df, colmap = rotate_multi_index_level(df, "__dataset_id", legend_map)
        show_cols = np.append(colmap.loc[:, common_cols].to_numpy().transpose().ravel(),
                              colmap.loc[colmap.index != baseline, measure_cols].to_numpy().transpose().ravel())
        cell = self.surface.make_cell(title="QEMU stats for common functions")
        view = self.surface.make_view("table", df=view_df, yleft=show_cols)
        cell.add_view(view)
        self.surface.next_cell(cell)


class QEMUStatsHistogramDataset(PerfettoDataSetContainer):
    fields = [
        IndexField("arg_set_id", dtype=int),
        IndexField("bucket", dtype=int),
        Field("start", dtype=np.uint),
        DerivedField("valid_symbol", dtype=object, isdata=False)
    ]

    def raw_fields(self, include_derived=False):
        fields = super().raw_fields(include_derived)
        fields += [f for f in QEMUStatsHistogramDataset.fields if include_derived or not f.isderived]
        return fields

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
        # First assign the bucket index as a column
        extractor = re.compile("\[([0-9]+)\]")
        input_df["bucket"] = input_df["key"].apply(lambda k: int(extractor.search(k).group(1)))
        # Make sure the index is now aligned for all bucket field groups
        input_df.set_index(["arg_set_id", "bucket"], inplace=True)
        # Rotate the keys into columns using the bucket index as row index
        tmp_df = pd.DataFrame(columns=input_df["flat_key"].unique())
        for c in tmp_df.columns:
            # Assume all bucket fields are int_values (int or uint)
            tmp_df[c] = input_df[input_df["flat_key"] == c]["int_value"]

        # XXX-AM: this is somewhat common to all perfetto dataset processing
        # Fixup column names
        tmp_df = tmp_df.rename(columns=self.field_names_map())
        # Add dataset ID
        tmp_df["__dataset_id"] = self.benchmark.uuid
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
        # If there is no 'end' columns size will be a zero vector
        size = self.df.get("end", self.df["start"]) - self.df["start"]
        resolved = self.df["start"].map(lambda addr: resolver.lookup_bounded(addr))
        self.df.loc[resolved.isna(), "valid_symbol"] = "no-match"

        sym_size = resolved.map(lambda syminfo: syminfo.size if syminfo else np.nan)
        size_mismatch = (sym_size.isna()) | (sym_size < size)
        self.df.loc[size_mismatch, "valid_symbol"] = "size-mismatch"
        invalid_syms = self.df["valid_symbol"] != "ok"

        self.df["symbol"] = resolved.map(lambda syminfo: syminfo.name if syminfo else None)
        self.df.loc[invalid_syms, "symbol"] = self.df.loc[invalid_syms, "start"].transform(lambda addr: f"0x{addr:x}")
        # Note: For the file name, we omit the directory part as otherwise the same executable
        # in different directories will be picked up as a completely different file. This is
        # not useful when comparing different compilations that have different paths e.g. the kernel
        # We also have to handle rtld manually to map its name.
        self.df["file"] = resolved.map(lambda syminfo: syminfo.filepath.name if syminfo else None)
        self.df.loc[invalid_syms, "file"] = "unknown"

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

    def _get_sql_expr(self):
        query = ("SELECT * FROM slice INNER JOIN args ON " + "slice.arg_set_id = args.arg_set_id WHERE " +
                 "slice.category = 'stats' AND slice.name = 'bb_hist' AND " + "args.flat_key LIKE 'qemu%'")
        return query

    def _register_plots(self, benchmark):
        super()._register_plots(benchmark)
        benchmark.register_plot(QEMUAddrRangeHistTable(benchmark, self))

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

    fields = [
        DataField("branch_count", dtype=int),
        DerivedField("call_count", dtype=int),
        DerivedField("delta_call_count", dtype=int),
        DerivedField("norm_delta_call_count", dtype=float)
    ]

    def _get_sql_expr(self):
        query = ("SELECT * FROM slice INNER JOIN args ON " + "slice.arg_set_id = args.arg_set_id WHERE " +
                 "slice.category = 'stats' AND slice.name = 'branch_hist' AND " + "args.flat_key LIKE 'qemu%'")
        return query

    def raw_fields(self, include_derived=False):
        fields = super().raw_fields(include_derived)
        fields += [f for f in QEMUStatsBranchHistogramDataset.fields if include_derived or not f.isderived]
        return fields

    def field_names_map(self):
        return {"qemu.histogram.bucket.start": "start", "qemu.histogram.bucket.value": "branch_count"}

    def _register_plots(self, benchmark):
        super()._register_plots(benchmark)
        benchmark.register_plot(QEMUAddrRangeHistTable(benchmark, self))
        # benchmark.register_plot(QEMUAddrRangeHistBars(benchmark, self))

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
        is_call = self.df["start"].map(lambda addr: resolver.lookup_exact(addr) is not None)
        self.df["call_count"] = self.df["branch_count"].mask(~is_call, 0).astype(int)

    def delta_columns(self):
        cols = super().delta_columns()
        return cols + ["call_count"]
