
import logging
import re
from pathlib import Path

import pandas as pd

from .core.dataset import DataSetContainer, IndexField, DataField


class QEMUAddressRangeHistogram(DataSetContainer):
    fields = [
        DataField("start"),
        DataField("end"),
        DataField("count")
    ]

    def raw_fields(self):
        return QEMUAddressRangeHistogram.fields

    def __init__(self, options, sym_resolver, prefix=""):
        super().__init__(options)
        self.file_prefix = prefix
        self.file_matcher = "{}qemu-([a-zA-Z0-9-]+)\.csv".format(prefix)

    def load(self, path: Path):
        match = re.match(self.file_matcher, path.name)
        if not match:
            logging.warning("Malformed qemu address range histogram file name %s:" +
                            "expecting %sqemu-<UUID>.csv", path, self.file_prefix)
            return
        dataset_id = match.group(1)
        csv_df = self._load_csv(path)
        csv_df["__dataset_id"] = dataset_id
        self._internalize_csv(csv_df)

    def process(self):
        pass
