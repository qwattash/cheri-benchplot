import logging
import typing
from pathlib import Path

import numpy as np
import pandas as pd

from .csv import CSVDataSetContainer
from .dataset import DataField, DatasetArtefact, Field, StrField


class ProcstatDataset(CSVDataSetContainer):
    dataset_source_id = DatasetArtefact.PROCSTAT
    fields = [
        Field("PID", dtype=int),
        DataField("START", dtype=int, importfn=lambda x: int(x, 16)),
        DataField("END", dtype=int, importfn=lambda x: int(x, 16)),
        StrField("PRT"),
        Field("RES"),
        Field("PRES"),
        Field("REF"),
        Field("SHD"),
        StrField("FLAG"),
        StrField("TP"),
        StrField("PATH")
    ]

    def raw_fields(self, include_derived=False):
        return ProcstatDataset.fields

    def _load_csv(self, path: Path, **kwargs):
        kwargs["sep"] = "\s+"
        return super()._load_csv(path, **kwargs)

    def load(self):
        path = self.output_file()
        csv_df = self._load_csv(path)
        self._append_df(csv_df)
        # Register the mapped binaries to the benchmark symbolizer
        for base, guest_path in self.mapped_binaries(self.benchmark.uuid):
            local_path = self.benchmark.rootfs / guest_path.relative_to("/")
            self.benchmark.register_mapped_binary(base, local_path)

    def mapped_binaries(self, dataset_id) -> typing.Iterator[tuple[int, str]]:
        """
        Iterate over (base_addr, path) of all the binaries mapped for the
        given dataset id.
        """
        xsection = self.df.xs(dataset_id, level="__dataset_id")
        binaries = xsection["PATH"][xsection["PATH"] != ""].unique()
        for name in binaries:
            addr = xsection.loc[xsection["PATH"] == name]["START"].min()
            yield (addr, Path(name))

    def output_file(self):
        return super().output_file().with_suffix(".csv")

    def _gen_run_procstat(self, proc_handle: "VariableRef"):
        """
        This should be used in subclasses to implement gen_pre_benchmark().
        Running procstat requires knowledge of the way to stop the benchmark at the correct time,
        unless we can use a generic way to stop at main() or exit()
        """
        self._script.gen_cmd("procstat", ["-v", proc_handle], outfile=self.output_file())
        self.logger.debug("Collected procstat info")
