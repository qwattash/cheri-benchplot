import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.dataset import (DataField, DatasetArtefact, DatasetName,
                            DatasetRunOrder, Field, IndexField,
                            align_multi_index_levels)
from ..core.json import JSONDataSetContainer


class NetstatDataset(JSONDataSetContainer):
    dataset_config_name = DatasetName.NETSTAT
    dataset_source_id = DatasetArtefact.NETSTAT
    fields = [
        IndexField("foo")
    ]

    def raw_fields(self, include_derived=False):
        return NetstatDataset.fields

    def output_file(self):
        return super().output_file().with_suffix(".json")

    def load(self):
        path = self.output_file()
        with open(path, "r") as fd:
            json_data = json.load(fd)
        records = json_data["netisr"]["workstream"][0]["work"]
        df = pd.DataFrame.from_records(records)
        df["__dataset_id"] = self.benchmark.uuid
        self._append_df(df)

    async def run_pre_benchmark(self):
        with open(self.output_file(), "w+") as netstat_out:
            await self.benchmark.run_cmd("netstat", ["--libxo", "json", "-Q"], outfile=netstat_out)

    async def run_post_benchmark(self):
        with open(self.output_file(), "w+") as netstat_out:
            await self.benchmark.run_cmd("netstat", ["--libxo", "json", "-Q"], outfile=netstat_out)
