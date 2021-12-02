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
        kwargs.setdefault("dtype", self._get_column_dtypes(include_converted=False))
        df = pd.io.json.read_json(path, **kwargs)
        df["__dataset_id"] = self.benchmark.uuid
        return df

    def _append_df(self, json_df: pd.DataFrame):
        json_df = json_df.astype(self._get_column_dtypes(include_converted=False))
        for f in self.raw_fields():
            if f.importfn:
                json_df[f.name] = json_df[f.name].transform(f.importfn)
        super()._append_df(json_df)
