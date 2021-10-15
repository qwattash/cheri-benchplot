import logging
from abc import abstractmethod
from pathlib import Path

import pandas as pd

from .dataset import DataSetContainer, DatasetProcessingException


class CSVDataSetContainer(DataSetContainer):
    """
    Base class to hold collections of fields to be loaded from CSV files.
    """
    def _load_csv(self, path: Path, **kwargs) -> pd.DataFrame:
        """
        Load a raw CSV file into a dataframe compatible with the columns given in all_columns.
        """
        kwargs.setdefault("dtype", self._get_column_dtypes(include_converted=False))
        kwargs.setdefault("converters", self._get_column_conv())
        csv_df = pd.read_csv(path, **kwargs)
        csv_df["__dataset_id"] = self.benchmark.uuid
        return csv_df

    def _internalize_csv(self, csv_df: pd.DataFrame):
        """
        Import the given csv dataframe into the main container dataframe.
        This means that:
        - The index columns must be in the given dataframe and must agree with the container dataframe.
        - The columns must be a subset of all_columns().
        """
        csv_df.set_index(self.index_columns(), inplace=True)
        column_subset = set(csv_df.columns).intersection(set(self.all_columns_noindex()))
        self.df = pd.concat([self.df, csv_df[column_subset]])
        # Check that we did not accidentally change dtype, this may cause weirdness due to conversions
        dtype_check = self.df.dtypes[column_subset] == csv_df.dtypes[column_subset]
        if not dtype_check.all():
            changed = dtype_check.index[~dtype_check]
            for col in changed:
                self.logger.error("Unexpected dtype change in %s: %s -> %s", col, csv_df.dtypes[col],
                                  self.df.dtypes[col])
            raise DatasetProcessingException("Unexpected dtype change")

    def load(self, path: Path):
        csv_df = self._load_csv(path)
        self._internalize_csv(csv_df)
