import logging
import typing
from pathlib import Path

import pandas as pd
import numpy as np

from .dataset import DatasetArtefact, DataField, StrField, Field
from .csv import CSVDataSetContainer


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

    def load(self, path):
        super().load(path)
        # Register the mapped binaries to the benchmark symbolizer
        for base, guest_path in self.mapped_binaries(self.benchmark.uuid):
            local_path = self.benchmark.rootfs / guest_path.relative_to("/")
            self.benchmark.register_mapped_binary(base, local_path)

    def mapped_binaries(self, dataset_id) -> typing.Iterator[tuple[int, str]]:
        """
        Iterate over (base_addr, path) of all the binaries mapped for the
        given dataset id.
        """
        xsection = self.df.xs(dataset_id)
        binaries = xsection["PATH"][xsection["PATH"] != ""].unique()
        for name in binaries:
            addr = xsection.loc[xsection["PATH"] == name]["START"].min()
            yield (addr, Path(name))

    def output_file(self):
        return super().output_file().with_suffix(".csv")

    async def _run_procstat(self, remote_pid: int):
        """
        This should be used in subclasses to implement run_pre_benchmark().
        Running procstat requires knowledge of the way to stop the benchmark at the correct time,
        unless we can use a generic way to stop at main() or exit()
        """
        with open(self.output_file(), "w+") as outfd:
            await self.benchmark.run_cmd("procstat", ["-v", str(remote_pid)], outfile=outfd)
        self.logger.debug("Collected procstat info")
