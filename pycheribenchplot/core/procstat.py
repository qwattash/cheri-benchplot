import logging
from pathlib import Path

import pandas as pd
import numpy as np

from .dataset import CSVDataSetContainer, DataField, StrField, Field
from .instanced import InstanceCheriBSD, InstancePlatform


class ProcstatDataset(CSVDataSetContainer):
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

    def raw_fields(self):
        return ProcstatDataset.fields

    def _load_csv(self, path: Path, **kwargs):
        kwargs["sep"] = "\s+"
        return super()._load_csv(path, **kwargs)

    def mapped_binaries(self, dataset_id) -> tuple[int, str]:
        """
        Iterate over (base_addr, path) of all the binaries mapped for the
        given dataset id.
        """
        xsection = self.df.xs(dataset_id)
        binaries = xsection["PATH"][xsection["PATH"] != ""].unique()
        for name in binaries:
            addr = xsection.loc[xsection["PATH"] == name]["START"].min()
            yield (addr, Path(name))
