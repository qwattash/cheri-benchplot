import logging
import typing
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from contextlib import contextmanager

import pandas as pd
import numpy as np


class DatasetProcessingException(Exception):
    pass


class DataSetParser(Enum):
    """
    Parser names to resolve a dataset configuration to the
    correct factory for the parser.
    """
    PMC = "pmc"
    NETPERF_DATA = "netperf-data"
    QEMU_STATS_BB_HIST = "qemu-stats-bb"
    QEMU_STATS_CALL_HIST = "qemu-stats-call"

    def __str__(self):
        return self.value


class Field:
    """
    Helper class to describe a data column from a CSV or other file
    """
    def __init__(self, name, desc=None, dtype=float, isdata=False, isindex=False, isderived=False, importfn=None):
        self.name = name
        self.dtype = dtype
        self.desc = desc if desc is not None else name
        self.isdata = isdata
        self.isindex = isindex
        self.isderived = isderived
        self.importfn = importfn


class StrField(Field):
    def __init__(self, name, desc=None, **kwargs):
        kwargs["dtype"] = str
        kwargs.setdefault("importfn", str)
        super().__init__(name, desc=None, **kwargs)


class DataField(Field):
    """
    A field representing benchmark measurement data instead of
    benchmark information.
    """
    def __init__(self, name, desc=None, dtype=float, **kwargs):
        kwargs["isdata"] = True
        super().__init__(name, desc=None, dtype=dtype, **kwargs)


class IndexField(Field):
    """
    A field representing benchmark setup index over which we can plot.
    """
    def __init__(self, name, desc=None, **kwargs):
        kwargs["isindex"] = True
        super().__init__(name, desc=None, **kwargs)


class DerivedField(Field):
    """
    Indicates a field that is generated during processing and will be guaranteed to
    appear only in the final aggregate dataframe for the dataset.
    """
    def __init__(self, name, desc=None, **kwargs):
        kwargs["isderived"] = True
        kwargs.setdefault("isdata", True)
        super().__init__(name, desc, **kwargs)


class DataSetContainer(ABC):
    """
    Base class to hold collections of fields containing benchmark data
    Each benchmark run is associated with an UUID which is used to cross-reference
    data from different files.
    """
    @classmethod
    def get_parser(cls, benchmark: "BenchmarkBase", dset_key: str):
        return cls(benchmark, dset_key)

    def __init__(self, benchmark: "Benchmarkbase", dset_key: str):
        """
        Arguments:
        benchmark: the benchmark instance this dataset belongs to
        dset_key: the key this dataset is associated to in the BenchmarkRunConfig
        """
        self.name = dset_key
        self.benchmark = benchmark
        self.config = benchmark.config
        self.logger = logging.getLogger(f"{self.config.name}:{dset_key}")
        self.df = pd.DataFrame(columns=self.all_columns())
        self.df = self.df.astype(self._get_column_dtypes(include_converted=True))
        self.df.set_index(self.index_columns(), inplace=True)
        self.merged_df = None
        self.agg_df = None
        if self.benchmark.instance_config.baseline:
            self._register_plots(benchmark)

    def raw_fields(self, include_derived=False) -> "typing.Sequence[Field]":
        """
        All fields that MAY be present in the input files.
        This is the set of fields that we expect to build the input dataframe (self.df).
        Other processing steps during pre-merge, and post-merge may add derived fields and
        index levels as necessary.
        """
        return []

    def all_columns(self, include_derived=False) -> "typing.Sequence[str]":
        """All column names in the container dataframe, including the index names"""
        return set(self.index_columns() + [f.name for f in self.raw_fields(include_derived)])

    def index_columns(self):
        """All column names that are to be used as dataset index in the container dataframe"""
        return ["__dataset_id"] + [f.name for f in self.raw_fields() if f.isindex]

    def all_columns_noindex(self) -> "typing.Sequence[str]":
        return set(self.all_columns()) - set(self.index_columns())

    def data_columns(self, include_derived=False):
        """
        All data column names in the container dataframe.
        This MUST NOT include synthetic data columns that are generated after importing the dataframe.
        """
        return [f.name for f in self.raw_fields(include_derived) if f.isdata]

    def _get_column_dtypes(self, include_converted=False) -> dict[str, type]:
        return {f.name: f.dtype for f in self.raw_fields() if include_converted or f.importfn is None}

    def _get_column_conv(self) -> dict:
        return {f.name: f.importfn for f in self.raw_fields() if f.importfn is not None}

    def _register_plots(self, benchmark: "BenchmarkBase"):
        pass

    @abstractmethod
    def load(self, path: Path):
        """Load the dataset from the given file"""
        ...

    def pre_merge(self):
        """
        Pre-process a dataset from a single benchmark run.
        This can be used as a hook to generate composite metrics before merging the datasets.
        """
        self.logger.debug("Pre-process %s", self.config.name)

    def init_merge(self):
        """
        Initialize merge state on the baseline instance we are merging into.
        """
        if self.merged_df is None:
            self.merged_df = self.df

    def merge(self, other: "DataSetContainer"):
        """
        Merge datasets from all the runs that we need to compare
        Note that the merged dataset will be associated with the baseline run, so the
        benchmark.uuid on the merge and post-merge operations will refer to the baseline implicitly.
        """
        self.logger.debug("Merge %s", self.config.name)
        if self.merged_df is None:
            src = self.df
        else:
            src = self.merged_df
        self.merged_df = pd.concat([src, other.df])

    def post_merge(self):
        """
        After merging, this is used to generate composite or relative metrics on the merged dataset.
        """
        self.logger.debug("Post-merge %s", self.config.name)

    def aggregate(self):
        """
        Aggregate the metrics in the merged runs.
        """
        self.logger.debug("Aggregate %s", self.config.name)

    def post_aggregate(self):
        """
        Generate composite metrics or relative metrics after aggregation.
        """
        self.logger.debug("Post-aggregate %s", self.config.name)


class CSVDataSetContainer(DataSetContainer):
    """
    Base class to hold collections of fields to be loaded from CSV files.
    """
    def _load_csv(self, path: Path, **kwargs) -> pd.DataFrame:
        """
        Load a raw CSV file into a dataframe compatible with the columns given in all_columns.
        """
        kwargs.setdefault("dtype", self._get_column_dtypes())
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


@contextmanager
def dataframe_debug():
    """Helper context manager to print whole dataframes"""
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        yield


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


def check_multi_index_aligned(df: pd.DataFrame, level: str):
    """
    Check that the given index level(s) are aligned.
    """
    group_size = df.groupby(level).count().iloc[:, 0]
    aligned = (group_size == group_size.iloc[0]).all()
    return aligned


def align_multi_index_levels(df: pd.DataFrame, align_levels: list[str], fill_value=None):
    """
    Align a subset of the levels of a multi-index.
    This will generate the union of the sets of values in the align_levels parameter.
    The union set is then repeated for each other dataframe index level, so that every
    combination of the other levels, have the same set of aligned level combinations.
    """
    # Get an union of the sets of levels to align as the index of the grouped dataframe
    align_sets = df.groupby(align_levels).count()
    # Values of the non-aggregated levels of the dataframe
    other_levels = [lvl for lvl in df.index.names if lvl not in align_levels]
    # Now get the unique combinations of other_levels
    other_sets = df.groupby(other_levels).count()
    # For each one of the other_sets levels, we need to repeat the aligned index union set, so we
    # create repetitions to make room for all the combinations
    align_cols = align_sets.index.to_frame().reset_index(drop=True)
    other_cols = other_sets.index.to_frame().reset_index(drop=True)
    align_cols_rep = align_cols.iloc[align_cols.index.repeat(len(other_sets))].reset_index(drop=True)
    other_cols_rep = other_cols.iloc[np.tile(other_cols.index, len(align_sets))].reset_index(drop=True)
    new_index = pd.concat([other_cols_rep, align_cols_rep], axis=1)
    new_index = pd.MultiIndex.from_frame(new_index)
    return df.reindex(new_index, fill_value=fill_value).sort_index()


def rotate_multi_index_level(df: pd.DataFrame,
                             level: str,
                             suffixes: dict[str, str] = None,
                             fill_value=None) -> tuple[pd.DataFrame]:
    """
    Given a dataframe with multiple datasets indexed by one level of the multi-index, rotate datasets into
    columns so that the index level is removed and the column values related to each dataset are concatenated
    and renamed with the given suffix map.
    We also emit a dataframe for the level/column mappings as follows.

    Example:
    ID  name  |  value
    A   foo   |    0
    A   bar   |    1
    B   foo   |    2
    B   bar   |    3

    is rotated into

    name  |  value_A value_B
    foo   |    0       2
    bar   |    1       3

    with the following column mapping
    ID  |  value
    A   |  value_A
    B   |  value_B
    """
    rotate_groups = df.groupby(level)
    if suffixes is None:
        suffixes = rotate_groups.groups.keys()
    colmap = pd.DataFrame(columns=df.columns, index=df.index.get_level_values(level).unique())
    groups = []
    for key, group in rotate_groups.groups.items():
        suffix = suffixes[key]
        colmap.loc[key, :] = colmap.columns.map(lambda c: f"{c}_{suffix}")
        rotated = df.loc[group].reset_index(level, drop=True).add_suffix(f"_{suffix}")
        groups.append(rotated)
    new_df = pd.concat(groups, axis=1)
    return new_df, colmap


def subset_xs(df: pd.DataFrame, selector: typing.Sequence[bool]):
    """
    Extract a cross section of the given levels of the dataframe, regarless of frame index ordering,
    where the values match the given set of values.
    """
    # XXX align the dataframe?
    levels = selector.index.names
    # First, make our levels the last ones
    swapped_levels = [n for n in df.index.names if n not in levels]
    swapped_levels.extend(levels)
    swapped = df.reorder_levels(swapped_levels)
    # Now we tile the selector
    ntiles = len(swapped) / len(selector)
    assert ntiles == int(ntiles)
    tiled = selector.loc[np.tile(selector.index, int(ntiles))]
    # Now we can cross-section with the boolean selection
    values = swapped.loc[tiled.values]
    return values.reorder_levels(df.index.names)
