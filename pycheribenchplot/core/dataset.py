from enum import Enum
from pathlib import Path

import pandas as pd
import numpy as np


class DataSetParser(Enum):
    """
    Parser names to resolve a dataset configuration to the
    correct factory for the parser.
    """
    PMC = "pmc"
    NETPERF_DATA = "netperf-data"
    QEMU_STATS = "qemu-stats"

    def __str__(self):
        return self.value


class Field:
    """
    Helper class to describe a data column from a CSV or other file
    """
    def __init__(self, name, desc=None, dtype=float, isdata=False, isindex=False, importfn=None):
        self.name = name
        self.dtype = dtype
        self.desc = desc if desc is not None else name
        self.isdata = isdata
        self.isindex = isindex
        self.importfn = importfn


class StrField(Field):
    def __init__(self, name, desc=None, **kwargs):
        kwargs["dtype"] = str
        super().__init__(name, desc=None, **kwargs)


class DataField(Field):
    """
    A field representing benchmark measurement data instead of
    benchmark information.
    """
    def __init__(self, name, desc=None, **kwargs):
        kwargs["isdata"] = True
        super().__init__(name, desc=None, **kwargs)


class IndexField(Field):
    """
    A field representing benchmark setup index over which we can plot.
    """
    def __init__(self, name, desc=None, **kwargs):
        kwargs["isindex"] = True
        super().__init__(name, desc=None, **kwargs)


class DataSetContainer:
    """
    Base class to hold collections of fields to be loaded from CSV files.
    Each benchmark run is associated with an UUID which is used to cross-reference
    data from different files.
    """

    def __init__(self, options):
        self.options = options
        self.df = pd.DataFrame(columns=self.all_columns())
        self.df.set_index(self.index_columns(), inplace=True)

    def raw_fields(self) -> "typing.Sequence[Field]":
        """All fields that MAY be present in the input files"""
        return []

    def all_columns(self) -> "typing.Sequence[str]":
        """All column names in the container dataframe, including the index names"""
        return set(self.index_columns() + [f.name for f in self.raw_fields()])

    def index_columns(self):
        """All column names that are to be used as dataset index in the container dataframe"""
        return ["__dataset_id"] + [f.name for f in self.raw_fields() if f.isindex]

    def data_columns(self):
        """
        All data column names in the container dataframe.
        This MUST NOT include synthetic data columns that are generated after importing the dataframe.
        """
        return [f.name for f in self.raw_fields() if f.isdata]

    def _load_csv(self, path: Path, **kwargs) -> pd.DataFrame:
        """
        Load a raw CSV file into a dataframe compatible with the columns given in all_columns.
        """
        dtype_map = {}
        converter_map = {}
        csv_df = pd.read_csv(path, **kwargs)

    def _internalize_csv(self, csv_df: pd.DataFrame):
        """
        Import the given csv dataframe into the main container dataframe.
        This means that:
        - The index columns must be in the given dataframe and must agree with the container dataframe.
        - The columns must be a subset of all_columns().
        """
        csv_df.set_index(self.index_columns(), inplace=True)
        column_subset = set(csv_df.columns).intersection(set(self.data_columns()))
        self.df = pd.concat([self.df, csv_df[column_subset]])


def get_numeric_columns(self, df):
    columns = []
    for idx, col in enumerate(self.stats.columns):
        if np.issubdtype(self.stats.dtypes[i], np.number):
            columns.append(col)
    return columns

def col2stat(prefix, colnames):
    """
    Map base column namens to the respective statistic column with
    the given prefix
    """
    return list(map(lambda c: "{}_{}".format(prefix, c), colnames))
