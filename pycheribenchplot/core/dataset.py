import typing
from enum import Enum
from pathlib import Path
from contextlib import contextmanager

import pandas as pd
import numpy as np

from .util import new_logger


class DatasetProcessingException(Exception):
    pass


class DatasetID(Enum):
    """
    Internal enum to reference datasets
    """
    PMC = "pmc"
    NETPERF_DATA = "netperf-data"
    QEMU_STATS_BB_HIST = "qemu-stats-bb"
    QEMU_STATS_CALL_HIST = "qemu-stats-call"
    QEMU_CTX_CTRL = "qemu-ctx-tracks"
    PROCSTAT = "procstat"
    PROCSTAT_NETPERF = "procstat-netperf"
    PIDMAP = "pidmap"
    VMSTAT_MALLOC = "vmstat-malloc"
    VMSTAT_UMA = "vmstat-uma"

    def __str__(self):
        return self.value


class DatasetRegistry(type):
    dataset_types = {}

    def __init__(self, name, bases, kdict):
        super().__init__(name, bases, kdict)
        if self.dataset_id:
            did = DatasetID(self.dataset_id)
            DatasetRegistry.dataset_types[did] = self


class Field:
    """
    Helper class to describe column and associated metadata to aid processing
    XXX-AM: Consider adding some sort of tags to the fields so that we can avoid hardcoding the
    names for some processing steps (e.g. normalized fields that should be shown as percentage,
    or address fields for hex visualization).
    May also help to split derived fields by the stage in which they are created
    (e.g. pre-merge, post-merge, post-agg). This should move the burden of declaring which fields to process.
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


class FieldTracker:
    """
    Manage field name transformations so that we know which derived fields come
    from which top-level fields
    """
    pass


class DataSetContainer(metaclass=DatasetRegistry):
    """
    Base class to hold collections of fields containing benchmark data
    Each benchmark run is associated with an UUID which is used to cross-reference
    data from different files.
    """
    # Identifier of the dataset used for registration
    dataset_id = None
    # Filter to enable this dataset only when the given benchmark_dataset ID is active
    # If None, this dataset is the default one used if there are no other matching datasets
    # for the ID.
    benchmark_dataset_id = None
    # Data class for the dataset-specific run options in the configuration file
    run_options_class = None

    def __init__(self, benchmark: "Benchmarkbase", dset_key: str, config: "BenchmarkDataSetConfig"):
        """
        Arguments:
        benchmark: the benchmark instance this dataset belongs to
        dset_key: the key this dataset is associated to in the BenchmarkRunConfig
        """
        self.name = dset_key
        self.benchmark = benchmark
        self.config = config
        self.logger = new_logger(f"{benchmark.config.name}:{dset_key}")
        self.df = pd.DataFrame(columns=self.all_columns())
        self.df = self.df.astype(self._get_column_dtypes())
        self.df.set_index(self.index_columns(), inplace=True)
        self.merged_df = None
        self.agg_df = None

    @property
    def bench_config(self):
        # This needs to be dynamic to grab the up-to-date configuration of the benchmark
        return self.benchmark.config

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

    def index_columns(self, include_derived=False):
        """All column names that are to be used as dataset index in the container dataframe"""
        return ["__dataset_id"] + [f.name for f in self.raw_fields(include_derived) if f.isindex]

    def all_columns_noindex(self) -> "typing.Sequence[str]":
        return set(self.all_columns()) - set(self.index_columns())

    def data_columns(self, include_derived=False):
        """
        All data column names in the container dataframe.
        This, by default, does not include synthetic data columns that are generated after importing the dataframe.
        """
        return [f.name for f in self.raw_fields(include_derived) if f.isdata]

    def _get_column_dtypes(self, include_converted=True, include_index=True, include_derived=False) -> dict[str, type]:
        fields = self.raw_fields(include_derived)
        idx = self.index_columns()
        dtypes = {}
        for f in fields:
            if not include_index and f.name in idx:
                continue
            if include_converted or f.importfn is None:
                dtypes[f.name] = f.dtype
        return dtypes

    def _get_column_conv(self) -> dict:
        return {f.name: f.importfn for f in self.raw_fields() if f.importfn is not None}

    def output_file(self):
        """
        Generate the output file for this dataset.
        Any extension suffix should be added in subclasses.
        """
        return self.benchmark.result_path / f"{self.name}-{self.benchmark.uuid}"

    def get_addrspace_key(self):
        """
        Return the name of the address-space to use for the benchmark address space in the symbolizer.
        This is only relevant for datasets that are intended to be used as the main benchmark dataset.
        """
        raise NotImplementedError("The address-space key must be specified by subclasses")

    def configure(self, options: "PlatformOptions"):
        """
        Finalize the dataset run_options configuration and add any relevant platform options
        to generate the dataset.
        """
        self.logger.debug("Configure dataset")
        if self.run_options_class:
            self.config = self.run_options_class(**self.config.run_options).bind(self.benchmark)
        return options

    async def run_pre_benchmark(self):
        self.logger.debug("Pre-benchmark")

    async def run_benchmark(self):
        self.logger.debug("Benchmark")

    async def run_post_benchmark(self):
        self.logger.debug("Post-benchmark")

    def load(self, path: Path):
        """Load the dataset from the given file"""
        pass

    def pre_merge(self):
        """
        Pre-process a dataset from a single benchmark run.
        This can be used as a hook to generate composite metrics before merging the datasets.
        """
        self.logger.debug("Pre-process")

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
        self.logger.debug("Merge")
        if self.merged_df is None:
            src = self.df
        else:
            src = self.merged_df
        self.merged_df = pd.concat([src, other.df])

    def post_merge(self):
        """
        After merging, this is used to generate composite or relative metrics on the merged dataset.
        """
        self.logger.debug("Post-merge")

    def aggregate(self):
        """
        Aggregate the metrics in the merged runs.
        """
        self.logger.debug("Aggregate")
        # Do nothing by default
        self.agg_df = self.merged_df

    def post_aggregate(self):
        """
        Generate composite metrics or relative metrics after aggregation.
        """
        self.logger.debug("Post-aggregate")


@contextmanager
def dataframe_debug():
    """Helper context manager to print whole dataframes"""
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        yield


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
    if len(df) == 0:
        return True
    group_size = df.groupby(level).size()
    aligned = (group_size == group_size.iloc[0]).all()
    return aligned


def align_multi_index_levels(df: pd.DataFrame, align_levels: list[str], fill_value=np.nan):
    """
    Align a subset of the levels of a multi-index.
    This will generate the union of the sets of values in the align_levels parameter.
    The union set is then repeated for each other dataframe index level, so that every
    combination of the other levels, have the same set of aligned level combinations.
    """
    assert df.index.is_unique, "Need unique index"
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

    # XXX-AM: Check that the levels are aligned, otherwise we may get unexpected results due to NaN popping out

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
    if len(groups):
        new_df = pd.concat(groups, axis=1)
        return new_df, colmap
    else:
        # Return the input dataframe without the index level, but no extra columns as there are
        # no index values to rotate
        return df.reset_index(level=level, drop=True), colmap


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


def reorder_columns(df: pd.DataFrame, ordered_cols: typing.Sequence[str]):
    """
    Reorder columns as the given column name list. Any remaining column is
    appended at the end.
    """
    extra_cols = list(set(df.columns) - set(ordered_cols))
    result_df = df.reindex(columns=np.append(ordered_cols, extra_cols))
    return result_df
