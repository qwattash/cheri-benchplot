import typing
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import (is_integer_dtype, is_numeric_dtype, is_object_dtype)

# from .config import DatasetArtefact, DatasetConfig, DatasetName
from .util import new_logger


class DatasetProcessingError(Exception):
    pass


class DatasetRunOrder(IntEnum):
    """
    Run ordering for datasets to extract data.
    This allows to control which dataset should run closer to the benchmark to
    avoid probe effect from other operations.
    """
    FIRST = 0
    ANY = 1
    LAST = 10


@dataclass
class Field:
    """
    Helper class to describe column and associated metadata to aid processing
    """
    name: str
    desc: str = None
    dtype: typing.Type = float
    isdata: bool = False
    isindex: bool = False
    isderived: bool = False
    importfn: typing.Callable = None

    @classmethod
    def str_field(cls, *args, **kwargs):
        kwargs.setdefault("importfn", str)
        return cls(*args, dtype=str, **kwargs)

    @classmethod
    def data_field(cls, *args, **kwargs):
        """A field representing benchmark measurement data instead of benchmark information."""
        return cls(*args, isdata=True, **kwargs)

    @classmethod
    def index_field(cls, *args, **kwargs):
        """A field representing benchmark setup index over which we can plot."""
        return cls(*args, isindex=True, **kwargs)

    @classmethod
    def derived_field(cls, *args, **kwargs):
        """A field that is generated during processing."""
        kwargs.setdefault("isdata", True)
        return cls(*args, isderived=True, **kwargs)

    @property
    def default_value(self):
        if is_integer_dtype(self.dtype):
            return 0
        elif is_numeric_dtype(self.dtype):
            return np.nan
        elif is_object_dtype(self.dtype):
            return None
        else:
            return ""

    def __post_init__(self):
        if self.desc is None:
            self.desc = self.name


class DatasetRegistry(type):
    dataset_types = defaultdict(list)

    @classmethod
    def resolve_name(cls, ds_name: "DatasetName") -> typing.Type["DataSetContainer"]:
        """
        Find the dataset class with the given configuration name.
        It is an error if multiple matches are found.
        """
        resolved = []
        for dset_list in cls.dataset_types.values():
            for dset in dset_list:
                if dset.dataset_config_name == ds_name:
                    resolved.append(dset)
        if len(resolved) == 0:
            raise KeyError("No dataset registered with the name %s", ds_name)
        if len(resolved) > 1:
            raise ValueError("Multiple datasets match %s", ds_name)
        return resolved[0]

    def __init__(self, name, bases, kdict):
        super().__init__(name, bases, kdict)
        if self.dataset_config_name:
            # Only attempt to register datasets that can be named in the configuration file
            assert self.dataset_source_id, "Missing dataset_source_id"
            # did = DatasetArtefact(self.dataset_source_id)
            did = None
            duplicates = [
                dset for dset in DatasetRegistry.dataset_types[did]
                if dset.dataset_config_name == self.dataset_config_name
            ]
            assert len(duplicates) == 0
            DatasetRegistry.dataset_types[did].append(self)
        all_fields = []
        for base in bases:
            if hasattr(base, "fields"):
                all_fields += base.fields
        all_fields += kdict.get("fields", [])
        self._all_fields = all_fields


class DataSetContainer(metaclass=DatasetRegistry):
    """
    Base class to hold collections of fields containing benchmark data
    Each benchmark run is associated with an UUID which is used to cross-reference
    data from different files.

    Each dataset exposes 3 dataframes:
    - df: the input dataframe. There is one input dataframe for each instance of the dataset,
    belonging to each existing benchmark run. This is the dataframe on which we operate until
    we reach the *merge* step.
    There are two mandatory index levels: the dataset_id and iterations levels.
    The dataset_id is the UUID of the benchmark run for which the data was captured.
    The iteration index level contains the iteration number, if it is meaningless for the
    data source, then it is set to -1.
    - merged_df: the merged dataframe contains the concatenated data from all the benchmark
    runs of a given benchmark. This dataframe is built only once for each benchmark, in the
    dataset container belonging to the baseline benchmark run. This is by convention, as the
    baseline dataset is used to aggregate all the other runs of the benchmark.
    - agg_df: the aggregate dataset. The aggregation dataset is generated from the merged
    dataset by aggregating across iterations (to compute mean, median, etc...) and any other
    index field relevant to the data source. This is considered to be the final output of the
    dataset processing phase. In the post-aggregation phase, it is expected that the dataset
    will produce delta values between relevant runs of the benchmark.

    Dataframe indexing and fields:
    The index, data and metadata fields that we want to import from the raw dataset should be
    declared as class properties in the DataSetContainer. The registry metaclass will take care
    to collect all the Field properties and make them available via the get_fields()
    method.
    The resulting dataframes use multi-indexes on both rows and columns.
    The row multi-index levels are dataset-dependent and are declared as IndexField(), in addition
    to the implicit dataset_id and iteration index levels.
    (Note that the iteration index level should be absent in the agg_df as it would not make sense).
    The column index levels are the following (by convention):
    - The 1st column level contains the name of each non-index field declared as input
    (including derived fields from pre_merge()).
    - The next levels contain the name of aggregates or derived columns that are generated.
    """
    # Unique name of the dataset in the configuration files
    dataset_config_name: "DatasetName" = None
    # Internal identifier of the dataset, this can be reused if multiple containers use the
    # same source data to produce different datasets
    dataset_source_id: "DatasetArtefact" = None
    dataset_run_order = DatasetRunOrder.ANY
    # Data class for the dataset-specific run options in the configuration file
    run_options_class = None

    def __init__(self, benchmark: "Benchmark", config: "DatasetConfig"):
        """
        Arguments:
        benchmark: the benchmark instance this dataset belongs to
        dset_key: the key this dataset is associated to in the BenchmarkRunConfig
        """
        self.benchmark = benchmark
        self.config = config.run_options
        self.logger = new_logger(f"{self.dataset_config_name}", parent=self.benchmark.logger)
        self.df = pd.DataFrame()
        self.merged_df = None
        self.agg_df = None
        self.cross_merged_df = None

    def _get_input_columns_dtype(self) -> typing.Dict[str, type]:
        """
        Get a dictionary suitable for pandas DataFrame.astype() to normalize the data type
        of input fields.
        This will include both index and non-index fields
        """
        return {f.name: f.dtype for f in self.input_fields()}

    def _get_input_columns_conv(self) -> dict:
        """
        Get a dictionary mapping input columns to the column conversion function, if any
        """
        return {f.name: f.importfn for f in self.input_fields() if f.importfn is not None}

    def _get_all_columns_dtype(self) -> typing.Dict[str, type]:
        """
        Get a dictionary suitable for pandas DataFrame.astype() to normalize the data type
        of all dataframe fields.
        This will include both index and non-index fields
        """
        return {f.name: f.dtype for f in self.__class__._all_fields}

    def _append_df(self, df):
        """
        Import the given dataframe for one or more iterations into the main container dataframe.
        This means that:
        - The index columns must be in the given dataframe and must agree with the container dataframe.
        - The columns must be a subset of all_columns().
        - The missing columns that are part of input_all_columns() are added and filled with NaN or None.
        this will not include derived or implicit index columns.
        """
        if "dataset_id" not in df.columns:
            self.logger.debug("No dataset column, using default")
            df["dataset_id"] = self.benchmark.uuid
        if "iteration" not in df.columns:
            self.logger.debug("No iteration column, using default (-1)")
            df["iteration"] = -1
        if "dataset_gid" not in df.columns:
            self.logger.debug("No dataset group, using default")
            df["dataset_gid"] = self.benchmark.g_uuid
        for pcol in self.parameter_index_columns():
            if pcol not in df.columns:
                param = self.benchmark.config.parameters[pcol]
                self.logger.debug("No parameter %s column, generate from config %s", pcol, param)
                df[pcol] = param
        if len(df) == 0:
            self.logger.warning("Appending empty dataframe")
        # Normalize columns to always contain at least all input columns
        existing = df.columns.to_list() + list(df.index.names)
        default_columns = []
        for f in self.input_fields():
            if f.name not in existing:
                col = pd.Series(f.default_value, index=df.index, name=f.name)
                default_columns.append(col)
        if default_columns:
            self.logger.debug("Add defaults for fields not found in input dataset.")
            df = pd.concat([df] + default_columns, axis=1)
        # Normalize type for existing columns
        col_dtypes = self._get_input_columns_dtype()
        df = df.astype(col_dtypes)
        df.set_index(self.input_index_columns(), inplace=True)
        # Only select columns from the input that are registered as fields, the ones in the index are
        # already selected
        dataset_columns = set(self.input_non_index_columns())
        avail_columns = set(df.columns)
        column_subset = list(avail_columns.intersection(dataset_columns))
        if self.df is None:
            self.df = df[column_subset]
        else:
            new_df = pd.concat([self.df, df[column_subset]])
            if len(new_df) < len(self.df):
                self.logger.error("Dataframe shrunk after append?")
                raise ValueError("Invalid dataframe append")
            self.df = new_df
            # Check that we did not accidentally change dtype, this may cause weirdness due to conversions
            dtype_check = self.df.dtypes[column_subset] == df.dtypes[column_subset]
            if not dtype_check.all():
                changed = dtype_check.index[~dtype_check]
                for col in changed:
                    self.logger.error("Unexpected dtype change in %s: %s -> %s", col, df.dtypes[col],
                                      self.df.dtypes[col])
                raise DatasetProcessingError("Unexpected dtype change")

    def _add_delta_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Initialize and add the delta columns index level.
        This is generally used as the third index level to hold the delta for each aggregated
        column with respect to other benchmark runs.
        The original data columns are labeled "sample".
        """
        col_idx = df.columns.to_frame()
        col_idx["delta"] = "sample"
        df = df.copy()
        df.columns = pd.MultiIndex.from_frame(col_idx)
        return df

    def _add_aggregate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Initialize and add the aggregate columns index level.
        This is intended to be used by datasets that do not aggregate on iterations
        but still need to have an empty level for alignment purposes.
        """
        col_idx = df.columns.to_frame()
        col_idx["aggregate"] = "-"
        df = df.copy()
        df.columns = pd.MultiIndex.from_frame(col_idx)
        return df

    def _set_delta_columns_name(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper to rename a new set of columns along the delta column index level.
        Note that we rename every column in the level.
        """
        level_index = df.columns.names.index("delta")
        new_index = df.columns.map(lambda t: t[:level_index] + (name, ) + t[level_index + 1:])
        df.columns = new_index
        return df

    def _compute_data_delta(self, data: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
        """
        Compute a simple delta columnwise between the data and baseline dataframes.
        """
        baseline_xs = baseline.loc[:, data.columns]
        assert data.columns.equals(baseline_xs.columns)
        aligned_baseline = broadcast_xs(data, baseline_xs)
        delta = data.subtract(aligned_baseline)
        norm_delta = delta.divide(aligned_baseline)
        result = [
            self._set_delta_columns_name("delta_baseline", delta),
            self._set_delta_columns_name("norm_delta_baseline", norm_delta)
        ]
        return result

    def _compute_iqr_delta(self, data: pd.DataFrame, baseline: pd.DataFrame, median_df,
                           median_delta: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the error propagation for a delta between the data and baseline columns.
        We subtract directly the quartile values so that the resulting error becomes:
        new_err_hi = data_err_hi + bs_err_lo
        new_err_lo = data_err_lo + bs_err_hi
        """
        bs_q75_xs = baseline.xs(("q75", "sample"), level=["aggregate", "delta"], axis=1, drop_level=False)
        bs_q25_xs = baseline.xs(("q25", "sample"), level=["aggregate", "delta"], axis=1, drop_level=False)
        q75_xs = data.xs(("q75", "sample"), level=["aggregate", "delta"], axis=1, drop_level=False)
        q25_xs = data.xs(("q25", "sample"), level=["aggregate", "delta"], axis=1, drop_level=False)
        # broadcast baseline along data
        bs_q75_xs = broadcast_xs(q75_xs, bs_q75_xs)
        bs_q25_xs = broadcast_xs(q25_xs, bs_q25_xs)
        # Need to rename the columns we are subtracting for clean subtraction or just use the values,
        # the latter relies on column ordering but it should be stable here
        q75_delta = q75_xs.subtract(bs_q25_xs.values)
        q25_delta = q25_xs.subtract(bs_q75_xs.values)
        result = [
            self._set_delta_columns_name("delta_baseline", q75_delta),
            self._set_delta_columns_name("delta_baseline", q25_delta)
        ]
        # Now compute the normalized quantile values. These are simple normalization of the delta w.r.t the
        # baseline median.
        bs_median = baseline.xs(("median", "sample"), level=["aggregate", "delta"], axis=1)
        # align the indexes by broadcasting the median cross section with the frames we divide
        tmp = broadcast_xs(q75_delta, bs_median)
        q75_norm = q75_delta.divide(tmp)
        tmp = broadcast_xs(q25_delta, bs_median)
        q25_norm = q25_delta.divide(tmp)
        result += [
            self._set_delta_columns_name("norm_delta_baseline", q75_norm),
            self._set_delta_columns_name("norm_delta_baseline", q25_norm)
        ]
        return result

    def _compute_delta_by_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        General operation to compute the delta of aggregated data columns between
        benchmark runs (identified by dataset_id).
        This will add the delta columns in the delta index level.
        It is assumed that the current benchmark instance is the baseline instance,
        this is the case if called from post_aggregate().
        Note: this is intended to be called only on the baseline benchmark instances.
        """
        if self.benchmark.session.analysis_config.baseline_gid:
            assert self.benchmark.g_uuid == self.benchmark.session.analysis_config.baseline_gid,\
                "computing delta columns on non-baseline benchmark"
        else:
            assert self.benchmark.config.instance.baseline,\
                "computing delta columns on non-baseline benchmark"
        assert check_multi_index_aligned(df, self.dataset_id_columns())
        assert "metric" in df.columns.names, "Missing column metric level"
        assert "aggregate" in df.columns.names, "Missing column aggregate level"
        assert "delta" in df.columns.names, "Missing column delta level"
        assert df.index.is_unique, "Input dataframe index must be unique"
        assert df.columns.is_unique, "Input dataframe columns must be unique"

        result = []
        # Find valid data columns that we have in the dataframe
        metric_cols = df.columns.unique("metric")
        valid_metrics = set(self.data_columns()).intersection(metric_cols)
        df = df.loc[:, df.columns.get_level_values("metric").isin(valid_metrics)]

        dataset_id_levels = self.dataset_id_columns()
        dataset_id_baseline = self.dataset_id_values()
        baseline = generalized_xs(df, dataset_id_baseline, levels=dataset_id_levels, droplevels=True)

        # For each existing column (metric, <x>, sample) we decide the action to take
        # based on the value of x.
        # - The data columns (e.g. mean, median and value) are simply subtracted.
        # - The error columns are combined according to error propagation rules.
        try:
            sample_xs = df.xs(("-", "sample"), level=["aggregate", "delta"], axis=1, drop_level=False)
            sample_delta = self._compute_data_delta(sample_xs, baseline)
            result.extend(sample_delta)
        except KeyError:
            self.logger.debug("No 'sample' agg value, skip deltas")

        try:
            median_xs = df.xs(("median", "sample"), level=["aggregate", "delta"], axis=1, drop_level=False)
            median_delta = self._compute_data_delta(median_xs, baseline)
            result.extend(median_delta)
            # assume that IQR columns are also there
            err_iqr = self._compute_iqr_delta(df, baseline, median_xs, median_delta)
            result.extend(err_iqr)
        except KeyError:
            self.logger.debug("No 'median' agg value, skip deltas")

        try:
            # XXX TODO
            mean_xs = df.xs(("mean", "sample"), level=["aggregate", "delta"], axis=1, drop_level=False)
        except KeyError:
            self.logger.debug("No 'mean' agg value, skip deltas")

        delta_df = pd.concat([df] + result, axis=1)
        delta_df = delta_df.sort_index(axis=1)

        # Sanity check on the results
        assert delta_df.index.is_unique, "Index not unique"
        assert delta_df.columns.is_unique, "Columns not unique"
        return delta_df

    def _get_aggregation_strategy(self) -> dict:
        """
        Return the aggregation strategy to use for each data column of interest.
        The return dictionary is suitable to be used in pd.DataFrameGroupBy.aggregate()
        """
        def q25(v):
            return np.quantile(v, q=0.25)

        def q75(v):
            return np.quantile(v, q=0.75)

        def q90(v):
            return np.quantile(v, q=0.90)

        agg_list = ["mean", "median", "std", q25, q75, q90]
        return {c: agg_list for c in self.data_columns()}

    def _compute_aggregations(self, grouped: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
        """
        Helper to generate the aggregate dataframe and normalize the column index level names.
        """
        agg = grouped.aggregate(self._get_aggregation_strategy())
        assert "metric" in agg.columns.names, "Missing column metric level"
        levels = list(agg.columns.names)
        levels[-1] = "aggregate"
        agg.columns = agg.columns.set_names(levels)
        return agg

    @property
    def name(self) -> str:
        return str(self.dataset_config_name)

    @property
    def bench_config(self):
        """
        This needs to be dynamic to grab the up-to-date configuration of the benchmark.
        """
        return self.benchmark.config

    @property
    def has_qemu(self) -> bool:
        """
        Check whether we are using an auxiliary dataset that requires qemu tracing output
        """
        # if (self.benchmark.get_dataset(DatasetName.QEMU_STATS_BB_HIT) is not None
        #         or self.benchmark.get_dataset(DatasetName.QEMU_STATS_BB_ICOUNT) is not None
        #         or self.benchmark.get_dataset(DatasetName.QEMU_STATS_CALL_HIT) is not None
        #         or self.benchmark.get_dataset(DatasetName.QEMU_UMA_COUNTERS) is not None
        #         or self.benchmark.get_dataset(DatasetName.QEMU_DYNAMORIO) is not None):
        #     return True
        return False

    def input_fields(self) -> typing.Sequence[Field]:
        """
        Return a list of fields that we care about in the input data source.
        This will not contain any derived fields.
        """
        fields = [f for f in self._all_fields if not f.isderived]
        return fields

    def input_index_fields(self) -> typing.Sequence[Field]:
        """
        Return a list of fields that are the index levels in the input dataset.
        This will not include derived index fields.
        """
        fields = [f for f in self.input_fields() if f.isindex]
        return fields

    def input_non_index_columns(self) -> typing.Sequence[str]:
        """
        Return a list of column names in the input dataset that are not index columns,
        meaning that we return only data and metadata columns.
        """
        return [f.name for f in self.input_fields() if not f.isindex]

    def input_all_columns(self) -> typing.Sequence[str]:
        """
        Return a list of column names that we are interested in the input data source.
        """
        return [f.name for f in self.input_fields()]

    def input_base_index_columns(self) -> typing.Sequence[str]:
        """
        Return a list of column names that represent the index fields present in the
        input data source.
        """
        return [f.name for f in self.input_index_fields()]

    def input_index_columns(self) -> typing.Sequence[str]:
        """
        Return a list of column names that represent the index fields present in the
        input dataframe. Any implicit index column will be reported here as well.
        """
        return self.implicit_index_columns() + [f.name for f in self.input_index_fields()]

    def implicit_index_columns(self):
        cols = ["dataset_gid", "dataset_id", "iteration"]
        cols += self.parameter_index_columns()
        return cols

    def dataset_id_columns(self) -> typing.Sequence[str]:
        """
        Return the columns needed to identify a datase, these should be retained in all
        dataframes (except when intentionally aggregating them away).
        """
        return ["dataset_gid", "dataset_id"] + self.parameter_index_columns()

    def parameter_index_columns(self):
        if self.benchmark.config.parameters:
            return list(self.benchmark.config.parameters.keys())
        return []

    def dataset_id_values(self):
        """
        Values for the dataset_id_columns for the current benchmark instance.
        """
        return [self.benchmark.g_uuid, self.benchmark.uuid] + list(self.benchmark.config.parameters.values())

    def index_columns(self) -> typing.Sequence[str]:
        """
        All column names that are to be used as dataset index in the container dataframe.
        This will contain both input and derived index columns.
        """
        input_cols = self.input_index_columns()
        return input_cols + [f.name for f in self._all_fields if f.isderived and f.isindex]

    def all_columns(self) -> typing.Sequence[str]:
        """
        All columns (derived or not) in the pre-merge dataframe, including index columns.
        """
        return self.implicit_index_columns() + [f.name for f in self._all_fields]

    def data_columns(self) -> typing.Sequence[str]:
        """
        All data column names in the container dataframe.
        This, by default, includes derived data columns that are generated after importing the dataframe.
        """
        return [f.name for f in self._all_fields if f.isdata and not f.isindex]

    def iteration_output_file(self, iteration):
        """
        Generate the output file for this dataset for the current benchmark iteration.
        Any extension suffix should be added in subclasses.
        """
        return self.benchmark.get_benchmark_iter_data_path(iteration) / f"{self.name}-{self.benchmark.uuid}"

    def output_file(self):
        """
        Generate the iteration-independent output file for this dataset.
        Any extension suffix should be added in subclasses.
        """
        return self.benchmark.get_benchmark_data_path() / f"{self.name}-{self.benchmark.uuid}"

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
        return options

    def configure_iteration(self, script: "ShellScriptBuilder", iteration: int):
        """
        Update configuration for the current benchmark iteration, if any depends on it.
        This is called for each iteration, before pre_benchmark_iter()
        (e.g. to update the benchmark output file options)

        :param script: The shell script builder
        :param iteration: The current benchmark iteration
        """
        self.logger.debug("Configure iteration %d", iteration)

    def gen_pre_benchmark(self, script: "ShellScriptBuilder"):
        """
        Generate runner script content before the benchmark run phase.

        :param script: The shell script builder
        """
        self.logger.debug("Gen pre-benchmark")

    def gen_pre_benchmark_iter(self, script: "ShellScriptBuilder", iteration: int):
        """
        Generate runner script content before every benchmark iteration.

        :param script: The shell script builder
        :param iteration: The current benchmark iteration
        """
        self.logger.debug("Gen pre-benchmark iteration %d", iteration)

    def gen_benchmark(self, script: "ShellScriptBuilder", iteration: int):
        """
        Generate the runner script command for the given benchmark iteration;

        :param script: The shell script builder
        :param iteration: The current benchmark iteration
        """
        self.logger.debug("Gen benchmark iteration %d", iteration)

    def gen_post_benchmark_iter(self, script: "ShellScriptBuilder", iteration: int):
        """
        Generate the runner script contet after every benchmark iteration.

        :param script: The shell script builder
        :param iteration: The current benchmark iteration
        """
        self.logger.debug("Gen post-benchmark iteration %d", iteration)

    def gen_post_benchmark(self, script: "ShellScriptBuilder"):
        """
        Generate runner script content after the benchmark run phase.

        :param script: The shell script builder
        """
        self.logger.debug("Gen post-benchmark")

    def gen_pre_extract_results(self, script: "ShellScriptBuilder"):
        """
        Generate runner script content after everything else, but before
        the data files are extracted.

        :param script: The shell script builder
        """
        self.logger.debug("Gen pre-extract")

    def before_run(self):
        """
        Give a chance to run any extra collection/configuration step before the instance runs.
        """
        self.logger.debug("Before run hook")

    async def after_extract_results(self, script: "ShellScriptBuilder", instance: "InstanceInfo"):
        """
        Give a chance to run commands on the live instance after the benchmark has
        completed. Note that this should only be used to extract auxiliary information
        that are not part of a dataset main input file, or to post-process output files.

        :param script: The shell script builder that we used to generate the script
        :param instance: The connected instance
        """
        self.logger.debug("Run post-extraction hook")

    def load(self):
        """
        Load the dataset from the common output files.
        Note that this is always called after iteration data has been loaded.
        No-op by default
        """
        pass

    def load_iteration(self, iteration: int):
        """
        Load the dataset per-iteration data.
        No-op by default
        """
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
        self.logger.debug("Merge %s", other)
        assert self.merged_df is not None, "forgot to call init_merge()?"
        self.merged_df = pd.concat([self.merged_df, other.df])

    def post_merge(self):
        """
        After merging, this can be used to generate composite or relative metrics on the merged dataset.
        """
        self.logger.debug("Post-merge")
        # Setup the name for the first hierarchical column index level
        self.merged_df.columns.name = "metric"

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

    def init_cross_merge(self):
        """
        Initialize cross-benchmark merge.
        This creates the initial merged dataframe to accumulate all aggregated
        frames from all the different benchmark parameterizations.
        """
        if self.cross_merged_df is None:
            self.cross_merged_df = self.agg_df

    def cross_merge(self, other: "DataSetContainer"):
        """
        Merge another aggregated dataframe from a different benchmark parameterization.
        """
        self.logger.debug("Cross-merge")
        assert self.cross_merged_df is not None, "forgot to call init_cross_merge()?"
        self.cross_merged_df = pd.concat([self.cross_merged_df, other.agg_df])

    def post_cross_merge(self):
        """
        Perform any extra operation after cross merge.
        """
        self.logger.debug("Post-cross-merge")


@contextmanager
def dataframe_debug():
    """Helper context manager to print whole dataframes"""
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        yield


def make_index_key(df: pd.DataFrame) -> namedtuple:
    """
    Given a dataframe, generate a namedtuple to easily access index tuples without
    resorting to having to lookup the index level index in the dataframe names.
    Note: this is only as stable as your index, if you change the index level order,
    the tuple will not match correctly.
    """
    IndexKey = namedtuple("IndexKey", df.index.names)
    return IndexKey


def check_multi_index_aligned(df: pd.DataFrame, level: typing.List[str]):
    """
    Check that the given index level(s) are aligned.
    """
    if len(df) == 0:
        return True
    if not df.index.is_unique:
        return False

    grouped = df.groupby(level)
    # just grab the first group to compare
    first_index = list(grouped.groups.values())[0]
    match = first_index.to_frame().reset_index(drop=True).drop(level, axis=1)
    for _, g in grouped:
        g_match = g.index.to_frame().drop(level, axis=1).reset_index(drop=True)
        if match.shape != g_match.shape:
            # There is no hope of equality
            return False
        if not (match == g_match).all().all():
            return False
    return True


def align_multi_index_levels(df: pd.DataFrame, align_levels: typing.List[str], fill_value=np.nan):
    """
    Align a subset of the levels of a multi-index.
    This will generate the union of the sets of values in the align_levels parameter.
    The union set is then repeated for each other dataframe index level, so that every
    combination of the other levels, have the same set of aligned level combinations.
    If the propagate_columns list is given, the nan values filled during alignment will
    be replaced by the original value of the column for the existing index combination.
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
    new_index = pd.MultiIndex.from_frame(new_index).reorder_levels(df.index.names)
    return df.reindex(new_index, fill_value=fill_value).sort_index()


def pivot_multi_index_level(df: pd.DataFrame, level: str, rename_map: dict = None) -> pd.DataFrame:
    """
    Pivot a row multi index level into the last level of the columns multi index.
    If a rename_map is given, the resulting column index level values are mapped accordingly to
    transform them into the new column level values.

    Example:
    ID  name  |  value
    A   foo   |    0
    A   bar   |    1
    B   foo   |    2
    B   bar   |    3

    pivots into

    name  | 0:      value
          | ID:   A   |   B
          | ------------------
    foo   |       0   |   2
    bar   |       1   |   3
    """
    keep_index = [lvl for lvl in df.index.names if lvl != level]
    df = df.reset_index().pivot(index=keep_index, columns=[level])
    if rename_map is not None:
        level_index = df.columns.names.index(level)
        mapped_level = df.columns.levels[level_index].map(lambda value: rename_map[value])
        df.columns = df.columns.set_levels(mapped_level, level=level)
    return df


def rotate_multi_index_level(df: pd.DataFrame,
                             level: str,
                             suffixes: typing.Dict[str, str] = None,
                             fill_value=None) -> typing.Tuple[pd.DataFrame]:
    """
    Given a dataframe with multiple datasets indexed by one level of the multi-index, rotate datasets into
    columns so that the index level is removed and the column values related to each dataset are concatenated
    and renamed with the given suffix map.
    We also emit a dataframe for the level/column mappings as follows.
    XXX deprecate and remove in favor of pivot_multi_index_level

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


def subset_xs(df: pd.DataFrame, selector: pd.Series, complement=False):
    """
    Extract a cross section of the given levels of the dataframe, regarless of frame index ordering,
    where the values match the given set of values.
    """
    if selector.dtype != bool:
        raise TypeError("selector must be a bool series")

    l, _ = selector.align(df)
    l = l.reorder_levels(df.index.names).fillna(False).sort_index()
    assert l.index.equals(df.index), f"{l.index} != {df.index}"
    if complement:
        l = ~l
    return df.loc[l]


def broadcast_xs(df: pd.DataFrame, chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe and a cross-section from it, with some missing index levels, generate
    the complete series or frame with the cross-section aligned to the parent frame.
    This is useful to perform an intermediate operations on a subset (e.g. the baseline frame)
    and then replicate the values for the rest of the datasets.
    """
    if df.index.nlevels > 1 and set(chunk.index.names).issubset(df.index.names):
        # First reorder the levels so that the shared levels between df and chunk are at the
        # front of df index names lis
        _, r = df.align(chunk, axis=0)
        return r.reorder_levels(df.index.names)
    else:
        if chunk.index.nlevels > 1:
            raise TypeError("Can not broadcast multiindex into flat index")
        nrepeat = len(df) / len(chunk)
        if nrepeat != int(nrepeat):
            raise TypeError("Can not broadcast non-alignable chunk")
        # Just repeat the chunk along the frame
        df = df.copy()
        df.loc[:] = chunk.values.repeat(nrepeat, axis=0)
        return df


def reorder_columns(df: pd.DataFrame, ordered_cols: typing.Sequence[str]):
    """
    Reorder columns as the given column name list. Any remaining column is
    appended at the end.
    """
    extra_cols = list(set(df.columns) - set(ordered_cols))
    result_df = df.reindex(columns=np.append(ordered_cols, extra_cols))
    return result_df


def index_where(df: pd.DataFrame, level: str, cond: pd.Series, alt: pd.Series):
    """
    Operation that mirrors dataframe.where but operates on an index level.
    """
    idx_df = df.index.to_frame()
    idx_df[level] = idx_df[level].where(cond, alt)
    df = df.copy()
    df.index = pd.MultiIndex.from_frame(idx_df)
    return df


def stacked_histogram(df_in: pd.DataFrame, group: str, stack: str, data_col: str, bins: list):
    """
    Helper to compute a dataframe suitable for plotting stacked multi-group
    histograms.
    Currently this only supports a single 'group' and 'stack' levels.
    """
    df = df_in.reset_index()
    g_uniq = df[group].unique()
    s_uniq = df[stack].unique()
    boundaries = np.array(bins)
    b_start = boundaries[:-1]
    b_end = boundaries[1:]
    hidx = pd.MultiIndex.from_product([g_uniq, s_uniq, b_start], names=[group, stack, "bin_start"])
    # preallocate dataframe
    hdf = pd.DataFrame({"count": 0}, index=hidx)

    groups = df.groupby([group, stack])
    for (k_group, k_stack), chunk in groups:
        count, out_bins = np.histogram(chunk[data_col], bins=bins)
        hist_key = (k_group, k_stack, slice(None))
        hdf.loc[hist_key, "bin_end"] = b_end
        hdf.loc[hist_key, "count"] = count
    return hdf.set_index("bin_end", append=True).sort_index()


def quantile_slice(df: pd.DataFrame,
                   columns: typing.List[typing.Union[str, tuple]],
                   quantile: float,
                   max_entries: int = None,
                   level: typing.List[str] = None) -> pd.DataFrame:
    """
    Filter a dataset to select the values where the given columns are above/below the given quantile threshold.
    Care is taken to maintain the slice index aligned at the given level (dataset_id by default),
    for this reason if one entry satisfies the threshold for one dataset,
    the values for other datasets will be included as well.
    The max_entries option allows to limit the number of entries that we select for each dataset group.
    Returns the dataframe containing the entries above the given quantile threshold.
    """
    if level is None:
        level = ["dataset_id"]
    if isinstance(level, str):
        level = [level]
    if max_entries is None:
        max_entries = np.inf
    # preliminary checking
    assert check_multi_index_aligned(df, level)

    level_complement = df.index.names.difference(level)
    high_thresh = df[columns].quantile(quantile)

    # We split each level group to determine the top N entries for each group.
    # Then we slice each group at max_entries and realign the values across groups.
    # This will result in potentially more than max_entries per group, but maintains data integrity
    # without dropping interesting values. Note that we select entries based on the global
    # high_thresh, so there may be empty group selections.
    def handle_group(g):
        # Any column may be above
        cond = (g[columns] >= high_thresh).apply(np.any, axis=1)
        sel = pd.DataFrame({"quantile_slice_select": False}, index=g.index)
        if cond.sum() > max_entries:
            cut = g[cond].sort_values(columns, ascending=False).index[max_entries:]
            cond.loc[cut] = False
        sel[cond] = True
        return sel

    sel = df.groupby(level, group_keys=False).apply(handle_group)
    # Need to propagate True values in sel across `level` containing the
    # complementary key matching the high value, this is necessary to maintain
    # alignment of the frame groups.
    sel = sel.groupby(level_complement, group_keys=True).transform(lambda g: g.any())
    high_df = df[sel["quantile_slice_select"]]
    # Make sure we are still aligned
    assert check_multi_index_aligned(high_df, level)
    return high_df.copy()


def assign_sorted_coord(df: pd.DataFrame,
                        sort: typing.List[str],
                        group_by=typing.List[str],
                        **sort_kwargs) -> pd.Series:
    """
    Assign coordinates for plotting to dataframe groups, preserving the index mapping between groups.
    This assumes that the dataframe is aligned at the given level.

    df: the dataframe to operate on
    sort: columns to use for sorting
    group_by: grouping levels/columns
    **sort_kwargs: extra sort_values() parameters
    """
    assert check_multi_index_aligned(df, group_by)
    # Do not trash source df
    df = df.copy()
    # We now we find the max for each complementary group. This will be used for cross-group sorting
    index_complement = df.index.names.difference(group_by)
    sort_max_key = df.groupby(index_complement).max()[sort]
    # Generate temporary sort keys
    ngroups = len(df.groupby(group_by))
    tmp_sort_keys = [f"__sort_tmp_{i}" for i in range(len(sort))]
    for tmp_key, col in zip(tmp_sort_keys, sort):
        df[tmp_key] = np.tile(sort_max_key[col].values, ngroups)
    sorted_df = df.sort_values(tmp_sort_keys + index_complement, **sort_kwargs)
    coord_by_group = sorted_df.groupby(group_by).cumcount()
    return coord_by_group.sort_index()


def generalized_xs(df: pd.DataFrame, match: list, levels: list, complement=False, droplevels=False):
    """
    Generalized cross section that allows slicing on multiple named levels.
    Example:
    Given a dataframe, generaized_xs(df, [0, 1], levels=["k0", "k1"]) gives:

     k0 | k1 | k2 || V
     0  | 0  | 0  || 1
     0  | 0  | 1  || 2
     0  | 1  | 0  || 3  generalized_xs()   k0 | k1 | k2 || V
     0  | 1  | 1  || 4 ==================> 0  | 1  | 0  || 3
     1  | 0  | 0  || 5                     0  | 1  | 1  || 4
     1  | 0  | 1  || 6
     1  | 1  | 0  || 7
     1  | 1  | 1  || 8
    """
    assert len(match) == len(levels)
    nlevels = len(df.index.names)
    slicer = [slice(None)] * nlevels
    for m, level_name in zip(match, levels):
        level_idx = df.index.names.index(level_name)
        slicer[level_idx] = m
    sel = pd.Series(False, index=df.index)
    sel.loc[tuple(slicer)] = True
    if complement:
        sel = ~sel
    result = df[sel]
    if droplevels:
        if nlevels > len(levels):
            result = result.droplevel(levels)
        else:
            result = result.reset_index()
    return result


def filter_aggregate(df: pd.DataFrame, cond: pd.Series, by: list, how="all", complement=False):
    """
    Filter a dataframe with an aggregation function across a set of levels, where the
    aggregation function matches in all groups or in any group, depending on the "how" parameter.

    df: The dataframe to operate on
    cond: condition vector
    by: levels to check the condition on
    how: "all" or "any". If "all", match the rows where `cond` is True across all `by` groups.
    If "any", match the rows where `cond` is true in at least one `by` group.
    complement: If False, return the matched rows, if True return `df` without the matched rows.

    Example:
    Given a dataframe, filter_aggregate(df, df["k1"] == 0, ["k0"]) gives:

     k0 | k1 | k2 || V
     0  | 0  | 0  || 1
     0  | 0  | 1  || 2                     k0 | k1 | k2 || V
     0  | 1  | 0  || 3  filter_aggregate() 0  | 0  | 0  || 1
     0  | 1  | 1  || 4 ==================> 0  | 0  | 1  || 2
     1  | 0  | 0  || 5                     1  | 0  | 0  || 5
     1  | 0  | 1  || 6                     1  | 0  | 1  || 6
     1  | 1  | 0  || 7
     1  | 1  | 1  || 8
    """
    if cond.dtype != bool:
        raise TypeError("cond must be a boolean series")
    if isinstance(by, str):
        by = [by]
    if how == "all":
        agg_fn = lambda g: g.all()
    elif how == "any":
        agg_fn = lambda g: g.any()
    else:
        raise ValueError("how must be 'all' or 'any'")

    # Try to use the index complement first, if not available, dont know
    index_complement = df.index.names.difference(by)
    if len(index_complement) == 0:
        raise ValueError("Can not select across all index levels")
    match = cond.groupby(index_complement).transform(agg_fn)
    if complement:
        match = ~match
    return df[match]


def scale_to_std_notation(series: pd.Series) -> int:
    """
    Return the standard power of 10 for the magnitude of a given series
    """
    mag = np.log10(series.abs().max())
    if np.isnan(mag) or np.isinf(mag):
        return 0
    std_mag = 3 * int(mag / 3)
    return std_mag
