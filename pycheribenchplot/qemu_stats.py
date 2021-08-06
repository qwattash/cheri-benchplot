import logging
import re
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from .core.dataset import (DataSetContainer, IndexField, DataField, Field, align_multi_index_levels)
from .core.plot import TablePlot, MatplotlibSurface, StackedPlot


class QEMUAddrRangeHistTable(StackedPlot):
    """
    Note: this only supports the HTML surface
    """
    def __init__(self, benchmark, dataset):
        super().__init__(benchmark, dataset, MatplotlibSurface())

    def _get_plot_title(self):
        return "QEMU PC hit count"

    def _get_plot_file(self):
        return self.benchmark.manager_config.output_path / "qemu-pc-hist-table.html"


# class QEMUAddrRangeHistPlot(StackedBarPlot):
#     pass


class QEMUAddressRangeHistogram(DataSetContainer):
    fields = [
        Field("CPU", dtype=int),
        Field("start", dtype=int, importfn=lambda x: int(x, 16)),
        Field("end", dtype=int, importfn=lambda x: int(x, 16)),
        DataField("count", dtype=int)
    ]

    def raw_fields(self):
        return QEMUAddressRangeHistogram.fields

    def _load_csv(self, path, **kwargs):
        kwargs["sep"] = ",\s*"
        kwargs["engine"] = "python"
        return super()._load_csv(path, **kwargs)

    def _register_plots(self, benchmark):
        super()._register_plots(benchmark)
        benchmark.register_plot(QEMUAddrRangeHistTable(benchmark, self))
        # benchmark.register_plot(QEMUAddrRangeHistPlot(benchmark, self))

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

    def aggregate(self):
        super().aggregate()
        tmp = self.merged_df.set_index(["file", "symbol"], append=True)
        grouped = tmp.groupby(["__dataset_id", "file", "symbol"])
        self.agg_df = grouped.agg({"count": "sum", "start": "min", "end": "max"})

    def post_aggregate(self):
        super().post_aggregate()
        # Align dataframe on the (file, symbol) pairs where we want to get an union of
        # the symbols set for each file, repeated for each dataset_id.
        new_df = align_multi_index_levels(self.agg_df, ["file", "symbol"], fill_value=0)
        # Now we can safely assign as there are no missing values.
        # Compute difference in calls for each function w.r.t. the baseline
        baseline = new_df.xs(self.benchmark.uuid, level="__dataset_id")
        datasets = new_df.index.get_level_values("__dataset_id").unique()
        new_df["diff"] = 0
        for ds_id in datasets:
            other = new_df.xs(ds_id, level="__dataset_id")
            diff = other.subtract(baseline)
            new_df.loc[ds_id, "diff"] = diff["count"].values
        self.agg_df = new_df
        # This is more for the plotting driver
        self.agg_df = new_df.sort_values(by="diff", key=abs, ascending=False)
        self.logger.info(self.agg_df.loc[[d for d in datasets if d != self.benchmark.uuid]])
