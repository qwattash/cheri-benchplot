import logging
from abc import abstractmethod
from pathlib import Path

import pandas as pd

from .dataset import DataSetContainer


class JSONDataSetContainer(DataSetContainer):
    """
    Base class to hold collections of fields to be loaded from JSON files.
    """
    def _load_json(self, path: Path, **kwargs) -> pd.DataFrame:
        """
        Load a raw CSV file into a dataframe compatible with the columns given in all_columns.
        """
        kwargs.setdefault("dtype", self._get_input_columns_dtype())
        df = pd.io.json.read_json(path, **kwargs)
        df["dataset_id"] = self.benchmark.uuid
        return df

    def _append_df(self, json_df: pd.DataFrame):
        for col, importfn in self._get_input_columns_conv().items():
            json_df[col] = json_df[col].transform(importfn)
        super()._append_df(json_df)
