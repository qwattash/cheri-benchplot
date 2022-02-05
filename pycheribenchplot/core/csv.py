from abc import abstractmethod
from pathlib import Path

import pandas as pd

from .dataset import DataSetContainer


class CSVDataSetContainer(DataSetContainer):
    """
    Base class to hold collections of fields to be loaded from CSV files.
    """
    def _load_csv(self, path: Path, **kwargs) -> pd.DataFrame:
        """
        Load a raw CSV file into a dataframe compatible with the columns given in all_columns.
        """
        kwargs.setdefault("dtype", self._get_input_columns_dtype())
        converters = self._get_input_columns_conv()
        if "converters" in kwargs:
            converters.update(kwargs["converters"])
        kwargs["converters"] = converters
        for key in kwargs["converters"].keys():
            kwargs["dtype"].pop(key, None)
        csv_df = pd.read_csv(path, **kwargs)
        csv_df["__dataset_id"] = self.benchmark.uuid
        return csv_df
