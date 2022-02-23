import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.dataset import (DatasetArtefact, DatasetName, DatasetRunOrder, Field, align_multi_index_levels)
from ..core.json import JSONDataSetContainer


class NetstatDataset(JSONDataSetContainer):
    dataset_config_name = DatasetName.NETSTAT
    dataset_source_id = DatasetArtefact.NETSTAT
    fields = [
        Field.index_field("workstream", dtype=int),
        Field.index_field("cpu", dtype=int),
        Field.index_field("name", dtype=str),
        Field("length", dtype=int),
        Field("watermark", dtype=int),
        Field.data_field("dispatched", dtype=int),
        Field("hybrid-dispatched", dtype=int),
        Field.data_field("queue-drops", dtype=int),
        Field.data_field("queued", dtype=int),
        Field.data_field("handled", dtype=int),
    ]

    def output_file(self):
        return super().output_file().with_suffix(".json")

    def load(self):
        path = self.output_file()
        pre = open(path.with_suffix(".pre"), "r")
        post = open(path.with_suffix(".post"), "r")
        try:
            index_cols = self.input_base_index_columns()
            pre_data = json.load(pre)
            post_data = json.load(post)
            pre_records = []
            post_records = []
            for item in pre_data["netisr"]["workstream"]:
                pre_records += item["work"]
            for item in post_data["netisr"]["workstream"]:
                post_records += item["work"]
            pre_df = pd.DataFrame.from_records(pre_records)
            pre_df.set_index(index_cols, inplace=True)
            post_df = pd.DataFrame.from_records(post_records)
            post_df.set_index(index_cols, inplace=True)
            df = post_df.subtract(pre_df).reset_index()
            df["dataset_id"] = self.benchmark.uuid
        finally:
            pre.close()
            post.close()
        self._append_df(df)

    def gen_pre_benchmark(self):
        netstat_out = self.output_file().with_suffix(".pre")
        self._script.gen_cmd("netstat", ["--libxo", "json", "-Q"], outfile=netstat_out)

    def gen_post_benchmark(self):
        netstat_out = self.output_file().with_suffix(".post")
        self._script.gen_cmd("netstat", ["--libxo", "json", "-Q"], outfile=netstat_out)
