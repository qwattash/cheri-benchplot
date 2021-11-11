import logging
from abc import abstractmethod
from pathlib import Path

import pandas as pd

from .dataset import DataSetContainer, DatasetProcessingException


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

    def _internalize_json(self, json_df: pd.DataFrame):
        """
        Import the given dataframe into the main container dataframe.
        This means that:
        - The index columns must be in the given dataframe and must agree with the container dataframe.
        - The columns must be a subset of all_columns().
        """
        json_df = json_df.astype(self._get_column_dtypes(include_converted=False))
        for f in self.raw_fields():
            if f.importfn:
                json_df[f.name] = json_df[f.name].transform(f.importfn)
        json_df.set_index(self.index_columns(), inplace=True)
        column_subset = set(json_df.columns).intersection(set(self.all_columns_noindex()))
        self.df = pd.concat([self.df, json_df[column_subset]])
        # Check that we did not accidentally change dtype, this may cause weirdness due to conversions
        dtype_check = self.df.dtypes[column_subset] == json_df.dtypes[column_subset]
        if not dtype_check.all():
            changed = dtype_check.index[~dtype_check]
            for col in changed:
                self.logger.error("Unexpected dtype change in %s: %s -> %s", col, json_df.dtypes[col],
                                  self.df.dtypes[col])
            raise DatasetProcessingException("Unexpected dtype change")

    def load(self, path: Path):
        json_df = self._load_json(path)
        self._internalize_json(json_df)
