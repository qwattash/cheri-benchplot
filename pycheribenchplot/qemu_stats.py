import logging
import re
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from .core.dataset import (IndexField, DataField, Field, align_multi_index_levels, rotate_multi_index_level, subset_xs,
                           check_multi_index_aligned, DatasetProcessingException)
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
        return "QEMU PC hit count"

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
        df = self.dataset.agg_df
        if not check_multi_index_aligned(df, "__dataset_id"):
            self.logger.error("Unaligned index, skipping plot")
            return

        df["norm_diff"] = df["norm_diff"] * 100  # make the ratio a percentage
        self.surface.set_layout(1, 1, expand=True, how="row")
        # Table for common functions
        nonzero = df["count"].groupby(["file", "symbol"]).min() != 0
        common_syms = nonzero & (nonzero != np.nan)
        common_df = subset_xs(df, common_syms)
        view_df, colmap = rotate_multi_index_level(common_df, "__dataset_id", legend_map)
        show_cols = np.append(
            colmap.loc[:, ["count", "call_count"]].to_numpy().transpose().ravel(),
            colmap.loc[colmap.index != baseline, ["diff", "norm_diff"]].to_numpy().transpose().ravel())
        sort_cols = colmap.loc[colmap.index != baseline, "norm_diff"].to_numpy().ravel()
        view_df2 = view_df[show_cols].sort_values(list(sort_cols), ascending=False, key=abs)
        cell = self.surface.make_cell(title="Common functions BB hit count")
        view = self.surface.make_view("table", df=view_df2)
        cell.add_view(view)
        self.surface.next_cell(cell)

        # Table for functions that are only in one of the runs
        extra_df = subset_xs(df, ~common_syms)
        view_df, colmap = rotate_multi_index_level(extra_df, "__dataset_id", legend_map)
        view_df = view_df[show_cols].sort_values(list(sort_cols), ascending=False, key=abs)
        cell = self.surface.make_cell(title="Extra functions")
        view = self.surface.make_view("table", df=view_df)
        cell.add_view(view)
        self.surface.next_cell(cell)


class QEMUStatsBBHistogramDataset(PerfettoDataSetContainer):
    fields = [
        IndexField("arg_set_id", dtype=int),
        IndexField("bucket", dtype=int),
        #Field("CPU", dtype=int),
        Field("start", dtype=np.uint),
        Field("end", dtype=np.uint),
        DataField("count", dtype=int)
    ]

    field_names_map = {
        "qemu.histogram.bucket.start": "start",
        "qemu.histogram.bucket.end": "end",
        "qemu.histogram.bucket.value": "count"
    }

    def raw_fields(self):
        return QEMUStatsBBHistogramDataset.fields

    def _get_sql_expr(self):
        query = ("SELECT * FROM slice INNER JOIN args ON " + "slice.arg_set_id = args.arg_set_id WHERE " +
                 "slice.category = 'stats' AND slice.name = 'bb_hist' AND " + "args.flat_key LIKE 'qemu%'")
        return query

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
            self.logger.debug(f"column {c} loop assign")
            # Assume all bucket fields are int_values (int or uint)
            tmp_df[c] = input_df[input_df["flat_key"] == c]["int_value"]

        # XXX-AM: this is somewhat common to all perfetto dataset processing
        # Fixup column names
        tmp_df = tmp_df.rename(columns=self.field_names_map)
        # Add dataset ID
        tmp_df["__dataset_id"] = self.benchmark.uuid
        # Normalize fields
        tmp_df = tmp_df.reset_index(drop=False).astype(self._get_column_dtypes(include_converted=True))
        tmp_df.set_index(self.index_columns(), inplace=True)
        return tmp_df

    # def _internalize_csv(self, csv_df: pd.DataFrame):
    #     super()._internalize_csv(csv_df)
    #     self._post_intern = pd.DataFrame(self.df)

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
        mapped = self.df["start"].map(lambda addr: resolver.lookup(addr))
        self.df["symbol"] = mapped.map(lambda syminfo: syminfo.name)
        # Note: For the file name, we omit the directory part as otherwise the same executable
        # in different directories will be picked up as a completely different file. This is
        # not useful when comparing different compilations that have different paths e.g. the kernel
        # We also have to handle rtld manually to map its name.
        self.df["file"] = mapped.map(lambda syminfo: syminfo.filepath.name)

        # Generate now a new column only for entries that EXACTLY match symbols, meaning that
        # this is the first basic-block of the function and is considered as an individual call
        # to that function
        is_call = self.df["start"].map(lambda addr: resolver.lookup_exact(addr) is not None)
        self.df["call_count"] = self.df["count"].mask(~is_call, 0).astype(np.uint)

        # Generate number of bytes hit for each range of start/end addresses as a proxy for
        # real instruction count. The real number will be some fraction of this number, depending
        # on instruction size.
        self.df["bcount"] = (self.df["end"] - self.df["start"]) * self.df["count"]

    def aggregate(self):
        super().aggregate()
        tmp = self.merged_df.set_index(["file", "symbol"], append=True)
        grouped = tmp.groupby(["__dataset_id", "file", "symbol"])
        self.agg_df = grouped.agg({"count": "sum", "call_count": "sum", "start": "min", "end": "max", "bcount": "sum"})
        # Check that the data is sensible
        not_sensible = self.agg_df["count"] < self.agg_df["call_count"].fillna(0)
        if not_sensible.any():
            self.logger.debug("Offending rows:\n%s", self.agg_df.loc[not_sensible])
            raise DatasetProcessingException("Call count must always be less than the total PC hit count")

    def post_aggregate(self):
        super().post_aggregate()
        # Align dataframe on the (file, symbol) pairs where we want to get an union of
        # the symbols set for each file, repeated for each dataset_id.
        new_df = align_multi_index_levels(self.agg_df, ["file", "symbol"], fill_value=0)
        # Now we can safely assign as there are no missing values.
        # 1) Compute difference in calls for each function w.r.t. the baseline.
        # 2) Now build the normalized absolute count, note that this will generate infinities where
        #   the baseline benchmark count is 0 (e.g. extra functions called only in other samples)
        baseline = new_df.xs(self.benchmark.uuid, level="__dataset_id")
        datasets = new_df.index.get_level_values("__dataset_id").unique()
        new_df[["diff", "diff_bcount", "norm_diff", "call_diff", "norm_call_diff"]] = 0
        new_df["norm_diff"] = 0
        # XXX could avoid this loop by repeating N times the baseline values and vectorize the division?
        for ds_id in datasets:
            other = new_df.xs(ds_id, level="__dataset_id")
            diff = other.subtract(baseline)
            norm_diff = diff["count"].divide(baseline["count"])
            norm_call_diff = diff["call_count"].subtract(baseline["call_count"])
            new_df.loc[ds_id, "diff"] = diff["count"].values
            new_df.loc[ds_id, "diff_bcount"] = diff["bcount"].values
            new_df.loc[ds_id, "call_diff"] = diff["call_count"].values
            new_df.loc[ds_id, "norm_diff"] = norm_diff.values
            new_df.loc[ds_id, "norm_call_diff"] = norm_call_diff.values
        self.agg_df = new_df
