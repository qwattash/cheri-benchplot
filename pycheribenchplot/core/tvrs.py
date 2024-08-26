import enum
import inspect
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from functools import reduce
from typing import Any, Dict, List, Optional, Self, Type, Union

import polars as pl
import polars.selectors as cs
import seaborn as sns
from marshmallow import ValidationError, validates_schema

from .analysis import AnalysisTask
from .config import Config, ConfigPath, LazyNestedConfig, config_field
from .task import ExecutionTask
from .util import bytes2int


@dataclass
class TVRSExecConfig(Config):
    """
    Base class for TVRS execution task configuration.
    """
    scenario_path: Optional[ConfigPath] = config_field(
        None, desc="Path to the scenarios directory, used to generate execution scripts")
    scenarios: Dict[str, LazyNestedConfig] = config_field(
        dict, desc="Inline scenarios. The key must match the name given by the `scenario` parameter")


class TVRSExecTask(ExecutionTask):
    """
    Generic execution task that parameterizes benchmark runs with
    the common parameterization.

    target: the CHERI compilation target, e.g. purecap/hybrid
    variant: benchmark variant, e.g. package built with stack zero-init
    runtime: run-time configuration, e.g. with/without temporal memory safety
    scenario: benchmark-specific parameters

    Note that the target axis is already taken into account by the dataset_gid
    and instance configuration.
    We need to determine the setup/teardown hooks for the current parameter set.
    """

    script_template: str = None
    scenario_config_class: Type[Config] = None

    def __init__(self, benchmark, script, task_config=None):
        super().__init__(benchmark, script, task_config=task_config)
        # Resolve the lazy scenarios configs
        if self.scenario_config_class:
            parsed_scenarios = {}
            for key, scenario_spec in self.config.scenarios.items():
                if isinstance(scenario_spec, self.scenario_config_class):
                    parsed_scenarios[key] = scenario_spec
                else:
                    parsed_scenarios[key] = self.scenario_config_class.schema().load(scenario_spec)
            self.config.scenarios = parsed_scenarios

        pkeys = set(self.benchmark.parameters.keys())
        for p in ["variant", "runtime", "scenario"]:
            if p not in pkeys:
                self.logger.error("Invalid parameterization: '%s' is required", p)
                raise RuntimeError("Invalid configuration")

    @property
    def scenario(self) -> Config:
        """
        Return the scenario for the current benchmark instantiation.

        This attempts to find the scenario key in the inline scenarios and then in
        the scenario path.
        """
        scenario_key = self.benchmark.parameters["scenario"]
        if scenario_config := self.config.scenarios.get(scenario_key):
            return scenario_config
        else:
            assert self.scenario_config_class is not None, "Subclass must set TVRSExecTask.scenario_config_class"
            if self.config.scenario_path is None:
                self.logger.error("Scenario '%s' does not exist", scenario_key)
                raise RuntimeError("InvalidConfiguration")
            path = (self.config.scenario_path / scenario_key).with_suffix(".json")
            if not path.exists() or not path.is_file():
                self.logger.error("Scenario file '%s' does not exist or is not a regular file", path)
                raise RuntimeError("Invalid configuration")
            return self.scenario_config_class.load_json(path)

    def run(self):
        # Set the script template as specified by subclasses
        if self.script_template:
            self.script.set_template(self.script_template)

        self.script.extend_context({"scenario_config": self.scenario})


class WeightMode(enum.Enum):
    SortAscendingAsInt = "ascending_int"
    SortDescendingAsInt = "descending_int"
    SortAscendingAsBytes = "ascending_bytes"
    SortDescendingAsBytes = "descending_bytes"
    SortAscendingAsStr = "ascending_str"
    SortDescendingAsStr = "descending_str"
    Custom = "custom"

    def is_descending(self):
        if (self == WeightMode.SortDescendingAsInt or self == WeightMode.SortDescendingAsBytes
                or self == WeightMode.SortDescendingAsStr):
            return True
        return False

    def dtype(self):
        if self == WeightMode.SortAscendingAsStr or self == WeightMode.SortDescendingAsStr:
            return str
        if self == WeightMode.SortAscendingAsInt or self == WeightMode.SortDescendingAsInt:
            return int
        # Bytes
        return bytes2int


@dataclass
class TVRSParamWeight(Config):
    """
    Parameter weighting rule for stable output ordering.
    """
    mode: WeightMode = config_field(WeightMode.SortAscendingAsStr, desc="Strategy for assigning weights")
    base: Optional[int] = config_field(None, desc="Base weight, ignored for 'custom' strategy.")
    step: Optional[int] = config_field(None, desc="Weight increment, ignored for 'custom' strategy.")
    weights: Optional[Dict[str, int]] = config_field(None, desc="Custom mapping of parameter values to weights")

    @validates_schema
    def validate_mode(self, data, **kwargs):
        if data["mode"] == WeightMode.Custom:
            if data["weights"] is None:
                raise ValidationError("TVRSParamWeight.weights must be set when TVRSParamWeight.mode is 'custom'")
        else:
            if data["base"] is None:
                raise ValidationError("TVRSParamWeight.base must be set")
            if data["step"] is None:
                raise ValidationError("TVRSParamWeight.step must be set")


@dataclass
class TVRSPlotConfig(Config):
    """
    Shared TVRS analysis task configuration.

    This should be inherited by all analysis task configuration that use the TVRSParamsMixin.
    """
    parameter_weight: Optional[Dict[str, TVRSParamWeight]] = config_field(
        None, desc="Weight for determining the order of labels based on the parameters")
    parameter_names: Optional[Dict[str, str]] = config_field(
        None, desc="Relabel parameter keys specified in PipelineBenchmarkConfig.parameterize")
    metric_names: Optional[Dict[str, str]] = config_field(
        None, desc="Relabel metric columns, available names depend on the benchmark")
    parameter_labels: Optional[Dict[str, Dict[str, str]]] = config_field(
        None, desc="Relabel parameter values specified in PipelineBenchmarkConfig.parameterize")
    hue_parameters: Optional[List[str]] = config_field(None,
                                                       desc="List of parameter keys to combine and map to the plot hue")
    plot_params: Dict[str, Any] = config_field(
        dict, desc="Plot appearance configuration for tweaking, see the seaborn plotting_context documentation")
    parameter_filter: Optional[Dict[str, Any]] = config_field(None,
                                                              desc="Filter the data for the given set of parameters")


class TVRSParamsContext:
    """
    Helper that manages a dataframe and the data parameterization axes.

    In particular, this allows to remap parameterization axis
    labels and values for plotting.
    """
    TVRS_PARAMETERS = ["target", "variant", "runtime", "scenario"]

    class ColumnMapping:
        def __init__(self, rename: dict[str, str]):
            self._rename = rename

        def __getattr__(self, name: str) -> str:
            return self._rename.get(name, name)

        def __getitem__(self, name: str | None) -> str | None:
            return self._rename.get(name, name)

    def __init__(self, task: "TVRSParamsMixin", df: pl.DataFrame):
        assert isinstance(task, AnalysisTask), "Task must be an AnalysisTask"
        self.task = task
        self.df = df

        if filter_args := task.tvrs_config().parameter_filter:
            self.df = self.df.filter(**filter_args)

        self.params = []
        self._rename = {}
        self.base_params = self._ensure_parameterization()
        self.params = list(self.base_params)
        self.extra_params = set(self.base_params).difference(self.TVRS_PARAMETERS)
        self._rename.update({p: p for p in self.base_params})
        self._rename.update({m: m for m in df.columns if m not in self.params})

    def copy_context(self, df: pl.DataFrame) -> Self:
        cp = copy(self)
        cp.df = df
        cp.params = list(self.params)
        cp._rename = dict(self._rename)
        cp.extra_params = set(self.extra_params)
        return cp

    def _ensure_parameterization(self):
        """
        Ensure that the dataframe has all the required columns, if not
        generate a default column when possible.
        """
        if not set(self.TVRS_PARAMETERS).issubset(self.df.columns):
            self.logger.error("Invalid dataframe, the following columns are required: %s, found: %s",
                              self.TVRS_PARAMETERS, self.df.columns)
            raise RuntimeError("Invalid dataframe")

        all_bench = self.task.session.all_benchmarks()
        assert len(all_bench) > 0
        pkeys = list(all_bench[0].parameters.keys())
        for check_param in self.TVRS_PARAMETERS:
            if check_param not in pkeys:
                self.logger.error("Missing parameter %s, found %s", check_param, pkeys)
                raise RuntimeError("Configuration error")
        return pkeys

    def _get_baseline_selector(self) -> (dict[str, any], list[str]):
        """
        Return the selector that is used to identify the baseline slice and
        the selector that is used to join the baseline and the main dataframe.

        The columns may have been relabeled, so we apply the same transformation
        to the selector.

        The join selector must take into account the degrees of freedom that the
        baseline selector has with respect to the target/variant/runtime/scenario
        parameter axes.
        Note that the join selector may be empty if the baseline selector leaves
        no degrees of freedom.
        """
        # Get the selector for the baseline dataframe slice
        baseline_sel = self.task.baseline_selector()
        # Filter baseline selector for dropped axes, check that the selector includes
        # only parameter columns and optionally the dataset_id/gid
        df_baseline_sel = {}
        for k, v in baseline_sel.items():
            if k in self.base_params:
                if k not in self.params:
                    self.logger.warn(
                        "Overhead calculation will ignore baseline selector %s=%s "
                        "because it is not active in the parameterization context", k, v)
                else:
                    df_baseline_sel[self._rename[k]] = v  # XXX not resistent to re-labeling
            else:
                self.logger.error("Baseline selector %s does not match any of the "
                                  "parameterization axes %s", k, self.base_params)
                raise RuntimeError("Unexpected baseline selector")

        # Determine how to join the baseline slice. This is done using
        # the keys that identify the baseline selector to find the degrees of freedom
        # that the baseline slice has and join on those.
        join_sel = [self._rename[p] for p in self.TVRS_PARAMETERS if p in self.params and p not in baseline_sel]

        return df_baseline_sel, join_sel

    @property
    def r(self) -> ColumnMapping:
        """
        Helper to access the parameterization columns after renaming 
        """
        return self.ColumnMapping(self._rename)

    @property
    def logger(self):
        return self.task.logger

    def melt(self, extra_id_vars: list[str] | None = None, **kwargs):
        """
        Melt the context dataframe using the parameter columns as id_vars.
        This ensures that the resulting dataframe can be used with any of the
        other context operations.
        """
        assert "id_vars" not in kwargs, "'id_vars' is autogenerated, use extra_id_vars."
        if extra_id_vars is None:
            extra_id_vars = []
        id_vars = {self._rename[p] for p in self.params}
        for col in extra_id_vars:
            id_vars.add(self._rename[col])
        id_vars.add("dataset_id")
        self.df = self.df.melt(id_vars=id_vars, **kwargs)

    def suppress_const_params(self, keep: list | None = None):
        """
        Suppress parameter axes with no variation from the dataframe.
        The parameterization axes in keep are not suppressed.

        Note: this should occur before we relabel parameters.
        """
        hide_params = []
        for p in self.params:
            if len(self.df[p].unique()) == 1 and p not in keep:
                hide_params.append(p)
        self.df = self.df.drop(hide_params)
        self.params = [p for p in self.params if p not in hide_params]
        self._rename.update({p: None for p in hide_params})

    def derived_param(self, name: str, expr: pl.Expr):
        """
        Produce a new parameter column based on the given expression, which
        is applied to the dataframe.
        """
        assert name not in self.params, f"Duplicated parameter column {name}"
        self.df = self.df.with_columns(expr.alias(name))
        self.params.append(name)
        self._rename[name] = name

    def derived_param_strcat(self, name: str, to_combine: list[str], sep=" "):
        """
        Merge two or more columns into a single column by string concatenation.
        This is useful when there are too many dimensions to show and we need to
        assign a column to some plot feature (e.g. hue). If the `to_combine`
        argument contains a suppressed (missing) colum, it will be skipped.
        """
        to_combine = [c for c in to_combine if c in self.df.columns]
        expr = pl.concat_str([pl.col(p).cast(pl.String) for p in to_combine], separator=sep)
        self.derived_param(name, expr)

    def derived_hue_param(self, default: list[str] | None, sep=" "):
        """
        Produce the _hue derived parameter by combining the given set of parameter columns.

        If the TVRS configuration specifies hue_parameters, this will override the
        `default` hue setting.
        If the given parameters have been suppressed, the hue column is not generated
        and the rename map will contain an entry mapping to None.
        """
        config = self.task.tvrs_config()
        hue_params = config.hue_parameters if config.hue_parameters is not None else default
        hue_params = [self._rename[p] for p in hue_params]
        if hue_params is None or len(hue_params) == 0:
            self._rename["_hue"] = None
            return

        requested = set(hue_params)
        hue_params = [hp for hp in hue_params if hp in self.base_params]
        for name in requested.difference(hue_params):
            self.logger.warning("Hue parameter '%s' is invalid and will be ignored", name)
        hue_params = [hp for hp in hue_params if hp in self.params]
        for name in requested.difference(hue_params):
            self.logger.info("Hue parameter '%s' is suppressed and will be ignored", name)
        if hue_params:
            self.derived_param_strcat("_hue", hue_params, sep=sep)
        else:
            self._rename["_hue"] = None

    def map_by_param(self, axis: str, mapper):
        """
        Run the given function for every group of the given parameter axis.
        """
        assert axis in self.df.columns, f"{axis} missing from dataframe"
        for chunk_id, chunk_df in self.df.groupby(axis):
            mapper(chunk_id, chunk_df)

    def _compute_lf_overhead_median(self,
                                    metric: str,
                                    inverted: bool = False,
                                    extra_groupby: list[str] | None = None,
                                    overhead_scale=1) -> Self:
        ID_COLUMNS = ["dataset_id"]
        # All real parameter axes that are also active
        base_columns = [self._rename[c] for c in self.base_params if c in self.params]
        # All active parameter axes, including derived ones
        param_columns = [self._rename[c] for c in self.params]
        # Get the selector for the baseline dataframe slice
        df_baseline_sel, join_sel = self._get_baseline_selector()

        if extra_groupby:
            param_columns += extra_groupby
            join_sel += extra_groupby

        bs_val = f"{metric}_baseline"
        m_q25 = f"{metric}_q25"
        m_q75 = f"{metric}_q75"
        # Generate mean and std values for each group
        agg_selectors = [
            cs.by_name(metric).median(),
            cs.by_name(metric).quantile(0.25, "linear").name.suffix("_q25"),
            cs.by_name(metric).quantile(0.75, "linear").name.suffix("_q75")
        ]
        agg_rename = cs.ends_with("_q25", "_q75").name.suffix("_baseline")
        stats = self.df.group_by(param_columns).agg(cs.by_name(ID_COLUMNS).first(), *agg_selectors)
        # Find the baseline dataframe slice
        bs = stats.filter(**df_baseline_sel).with_columns(cs.by_name(metric).name.suffix("_baseline"), agg_rename)

        # Now we align the input and baseline median frames to compute delta and overhead
        if join_sel:
            join_bs = bs.select(cs.ends_with("_baseline") | cs.by_name(join_sel))
            join_df = self.df.join(join_bs, on=join_sel)
        else:
            # The selector may be empty if the baseline slice selects a specific
            # target/variant/runtime/scenario combination.
            # In this case, baseline is a single row which we replicate for every
            # parameter combination
            _, right_df = pl.align_frames(self.df, baseline, on=list(df_baseline_sel.keys()))
            join_df = right_df.with_columns(cs.by_name(metric).fill_null(strategy="forward"))
        assert join_df.shape[0] == self.df.shape[0], "Unexpected join result"

        # Use nonparametric bootstrapping to estimate difference of medians
        # We do this by repeating the bootstrap method for each metric we have.
        # XXX TODO scipy.stat.bootstrap()
        delta = (
            join_df.with_columns(pl.col(metric) - pl.col(bs_val), ).group_by(param_columns).agg(
                cs.by_name(ID_COLUMNS).first(), *agg_selectors).with_columns(
                    pl.lit(metric + "_delta").alias("_metric_type"),
                    # Mask the baseline slice with NaN
                    pl.when(**df_baseline_sel).then(float("nan")).otherwise(pl.col(metric)).alias(metric),
                    pl.when(**df_baseline_sel).then(float("nan")).otherwise(pl.col(m_q25)).alias(m_q25),
                    pl.when(**df_baseline_sel).then(float("nan")).otherwise(pl.col(m_q75)).alias(m_q75)))

        # XXX Use bootstrapping here as well
        ovh = (
            join_df.with_columns(
                (pl.col(metric) / pl.col(bs_val) - pl.lit(1)).mul(overhead_scale), ).group_by(param_columns).agg(
                    cs.by_name(ID_COLUMNS).first(), *agg_selectors).with_columns(
                        pl.lit(metric + "_overhead").alias("_metric_type"),
                        # Mask the baseline slice with NaN
                        pl.when(**df_baseline_sel).then(float("nan")).otherwise(pl.col(metric)).alias(metric),
                        pl.when(**df_baseline_sel).then(float("nan")).otherwise(pl.col(m_q25)).alias(m_q25),
                        pl.when(**df_baseline_sel).then(float("nan")).otherwise(pl.col(m_q75)).alias(m_q75)))

        # Combine the stats, delta and ovh frames to have long-form data layout
        stats = stats.with_columns(pl.lit(metric).alias("_metric_type"))
        lf_stats = pl.concat(
            [stats,
             delta.select(cs.all().exclude("^.*_baseline$")),
             ovh.select(cs.all().exclude("^.*_baseline$"))],
            how="vertical",
            rechunk=True)
        return self.copy_context(lf_stats)

    def compute_lf_overhead(self,
                            metric: str,
                            inverted: bool = False,
                            extra_groupby: list[str] | None = None,
                            overhead_scale=1,
                            how="mean") -> Self:
        """
        Generate overhead vs the common baseline for long-form data. The result
        is returned as a new parameter context.

        The new dataframe will have an additional derived parameter axis named
        _metric_type, which assumes the values of '<metric>', '<metric>_delta' and
        '<metric>_overhead'.
        The <metric> column change its meaning depending on the value of _metric_type.
        Note that in the resulting context, the baseline data is not suppressed.

        If how = "mean", data is aggregated using mean and standard deviation.
        The standard deviation is propagated for the delta and percent overhead as
        the _std columns.
        This makes assumptions about statistical independence of the measurements and
        normality of the distribution.

        If how = "median", data is aggregated using median and quartile range. The IQR is
        computed for the delta and percent overhead as _q25 and _q75 columns.
        Note that the relative measures are normalised with respect to the median,
        the quartiles will indicate spread but do not account the uncertainty on the
        baseline measure.
        """
        assert how == "mean" or how == "median"

        if how == "median":
            return self._compute_lf_overhead_median(metric, inverted, extra_groupby, overhead_scale)

        ID_COLUMNS = ["dataset_id"]
        # All real parameter axes that are also active
        base_columns = [self._rename[c] for c in self.base_params if c in self.params]
        # All active parameter axes, including derived ones
        param_columns = [self._rename[c] for c in self.params]
        # Get the selector for the baseline dataframe slice
        df_baseline_sel, join_sel = self._get_baseline_selector()

        if extra_groupby:
            param_columns += extra_groupby
            join_sel += extra_groupby

        metrics = [metric]
        # Generate mean and std values for each group
        agg_selectors = [cs.by_name(metric).mean(), cs.by_name(metric).std().name.suffix("_std")]
        agg_rename = cs.ends_with("_std").name.suffix("_baseline")
        stats = self.df.group_by(param_columns).agg(cs.by_name(ID_COLUMNS + param_columns).first(), *agg_selectors)
        # Find the baseline dataframe slice
        bs = stats.filter(**df_baseline_sel).with_columns(cs.by_name(metrics).name.suffix("_baseline"), agg_rename)

        # Now we align the stats and bs frames to compute delta and overhead
        if join_sel:
            join_bs = bs.select(cs.ends_with("_baseline") | cs.by_name(join_sel))
            join_df = stats.join(join_bs, on=join_sel)
        else:
            # The selector may be empty if the baseline slice selects a specific
            # target/variant/runtime/scenario combination.
            # In this case, baseline is a single row which we replicate for every
            # parameter combination
            _, right_df = pl.align_frames(self.df, baseline, on=list(df_baseline_sel.keys()))
            join_df = right_df.with_columns(cs.by_name(metrics).fill_null(strategy="forward"))
        assert join_df.shape[0] == stats.shape[0], "Unexpected join result"

        # Compute the absolute DELTA for each metric
        bs_val = f"{metric}_baseline"
        m_std = f"{metric}_std"
        bs_std = f"{metric}_std_baseline"
        delta = join_df.with_columns(
            pl.lit(metric + "_delta").alias("_metric_type"),
            pl.col(metric) - pl.col(bs_val), (pl.col(m_std).pow(2) + pl.col(bs_std).pow(2)).sqrt()).with_columns(
                # Mask the baseline slice with NaN
                pl.when(**df_baseline_sel).then(float("nan")).otherwise(pl.col(metric)).alias(metric),
                pl.when(**df_baseline_sel).then(float("nan")).otherwise(pl.col(m_std)).alias(m_std))

        # Compute the % Overhead for each metric
        rel_err_sum = (pl.col(m_std) / pl.col(metric)).pow(2) + (pl.col(bs_std) / pl.col(bs_val)).pow(2)
        ovh = delta.with_columns(
            pl.lit(metric + "_overhead").alias("_metric_type"),
            (pl.col(metric) / pl.col(bs_val)).mul(overhead_scale),
            ((pl.col(m_std) / pl.col(metric)).pow(2) +
             (pl.col(bs_std) / pl.col(bs_val)).pow(2)).alias("_tmp_sum_of_err_squared"),
        ).with_columns((pl.col(metric).abs() * pl.col("_tmp_sum_of_err_squared").sqrt()).alias(m_std)).with_columns(
            # Mask the baseline slice with NaN
            pl.when(**df_baseline_sel).then(float("nan")).otherwise(pl.col(metric)).alias(metric),
            pl.when(**df_baseline_sel).then(float("nan")).otherwise(pl.col(m_std)).alias(m_std))

        # Combine the stats, delta and ovh frame to have long-form data representation
        stats = stats.with_columns(pl.lit(metric).alias("_metric_type"))
        lf_stats = pl.concat([
            stats,
            delta.select(cs.all().exclude("^.*_baseline$")),
            ovh.select(cs.all().exclude("^.*_baseline$").exclude("^_tmp.*$"))
        ],
                             how="vertical",
                             rechunk=True)
        return self.copy_context(lf_stats)

    def compute_overhead(self,
                         metrics: list[str],
                         inverted: bool = False,
                         extra_groupby: list[str] | None = None) -> Self:
        """
        Generate the overhead vs the common baseline column and returns it as a
        new parameter context.
        """
        ID_COLUMNS = ["dataset_id"]
        # All real parameter axes that are also active
        base_columns = [self._rename[c] for c in self.base_params if c in self.params]
        # All active parameter axes, including derived ones
        param_columns = [self._rename[c] for c in self.params]
        # Get the selector for the baseline dataframe slice
        df_baseline_sel, join_sel = self._get_baseline_selector()

        if extra_groupby:
            param_columns += extra_groupby
            join_sel += extra_groupby

        # Generate the baseline dataframe slice.
        # We compute the mean of the baseline metrics columns to compute the overhead.
        # XXX We should deal with error propagation here...
        # yapf: disable
        baseline = (
            self.df.filter(**df_baseline_sel)
            .select(ID_COLUMNS + param_columns + metrics)
            .group_by(param_columns)
            .agg(
                cs.by_name(metrics).mean().name.suffix("_baseline"),
                cs.by_name(metrics).std().name.suffix("_std_baseline"),
                cs.by_name(ID_COLUMNS).first()
            )
        )
        # yapf: enable

        if join_sel:
            join_df = self.df.join(baseline, on=join_sel, suffix="__join_right")
            # Suppress any unwanted columns on the right
            join_df = join_df.drop(cs.ends_with("__join_right"))
        else:
            # The selector may be empty if the baseline slice selects a specific
            # target/variant/runtime/scenario combination.
            # In this case, baseline is a single row which we replicate for every
            # parameter combination
            _, right_df = pl.align_frames(self.df, baseline, on=list(df_baseline_sel.keys()))
            join_df = right_df.with_columns(cs.by_name(metrics).fill_null(strategy="forward"))

        assert join_df.shape[0] == self.df.shape[0], "Unexpected join result"
        # Create the overhead columns and optionally drop the baseline data
        sign = -1 if inverted else 1
        delta_col_expr = []
        for m in metrics:
            b_m = f"{m}_baseline"
            d_m = f"{m}_delta"
            delta = (pl.col(m) - pl.col(b_m)) * sign
            delta_col_expr.append(delta.alias(d_m))
            self._rename[d_m] = d_m
        stat_df = join_df.with_columns(delta_col_expr)

        ovh_col_expr = []
        for m in metrics:
            b_m = f"{m}_baseline"
            d_m = f"{m}_delta"
            o_m = f"{m}_overhead"
            ovh = pl.col(d_m) * 100 / pl.col(b_m)
            ovh_col_expr.append(ovh.alias(o_m))
            self._rename[o_m] = o_m
        stat_df = stat_df.with_columns(ovh_col_expr)

        # Filter out the baseline data
        baseline_eq_exprs = map(lambda i: pl.col(i[0]).eq(i[1]), df_baseline_sel.items())
        not_baseline_sel = reduce(lambda x, y: x & y, baseline_eq_exprs).not_()
        stat_df = stat_df.filter(not_baseline_sel)

        # Create the new context
        return self.copy_context(stat_df)

    def relabel(self, default: dict[str, str] = None):
        """
        Transform the dataframe to adjust displayed properties.

        This applies the plot configuration to rename parameter levels, axes and
        filters.
        Columns that have their values mapped through parameter_labels are cloned
        and retain the original value through generated _r_<column>.
        """
        config = self.task.tvrs_config()

        # Generate default mappings for the _r_<col> renames
        self._rename.update({f"_r_{col}": col for col in self.df.columns})

        # Parameter renames
        if config.parameter_labels:
            relabeling = []
            for name, mapping in config.parameter_labels.items():
                if name not in self.df.columns:
                    self.logger.info("Skipping re-labeling of parameter '%s', does not exist", name)
                    continue
                self._rename[f"_r_{name}"] = f"_r_{name}"
                relabeling.append(pl.col(name).alias(f"_r_{name}"))
                relabeling.append(pl.col(name).replace(mapping))
            self.df = self.df.with_columns(*relabeling)

        col_rename_map = default or dict()
        col_rename_map.update(config.parameter_names or {})
        col_rename_map.update(config.metric_names or {})

        # Ensure that we do not try to rename suppressed columns in the name mapping
        def _filter_suppressed(name):
            if name not in self.df.columns:
                self.logger.info(
                    "Skipping re-naming of column '%s' because it is not active in "
                    "the parameterization context", name)
                return False
            return True

        col_rename_map = {c: v for c, v in col_rename_map.items() if _filter_suppressed(c)}

        self.df = self.df.rename(col_rename_map)
        self._rename.update(col_rename_map)

    def build_palette_for(self, param_name: str, allow_empty: bool = True):
        """
        Generate a color palette with colors associated to the given parameterization axis
        """
        name = self._rename[param_name]
        if name:
            ncolors = len(self.df[name].unique())
        elif allow_empty:
            ncolors = 1
        else:
            return None
        return sns.color_palette(n_colors=ncolors)

    def sort(self, descending=False):
        """
        Sort the dataframe according to the configured param_weight.

        Note that this may occur before or after relabeling.
        """
        config = self.task.tvrs_config()
        if not config.parameter_weight:
            self.logger.info("Skipping sort(), not configured")
            return
        df = self.df.with_columns(pl.lit(0.0).alias("param_weight"))
        for name, weight_spec in config.parameter_weight.items():
            col_name = self._rename.get(name, name)
            if col_name not in df.columns:
                self.logger.info("Skipping weight for parameter '%s', does not exist", col_name)
                continue

            if weight_spec.mode == WeightMode.Custom:
                mapping = dict(weight_spec.weights)
                # We may have relabeled the parameter values through the `parameter_labels`
                # configuration, so we extend the mapping with the corresponding aliases,
                # if there are any defined in `parameter_labels`.
                if config.parameter_labels and name in config.parameter_labels:
                    alias_weights = {}
                    for old_p, new_p in config.parameter_labels[name].items():
                        alias_weights[new_p] = mapping.get(old_p, 0)
                    mapping.update(alias_weights)
            else:
                descending = weight_spec.mode.is_descending()
                coerce_dtype = weight_spec.mode.dtype()
                if inspect.isfunction(coerce_dtype):
                    unique_values = df[col_name].map_elements(coerce_dtype).unique()
                else:
                    unique_values = df[col_name].cast(coerce_dtype).unique()
                sorted_values = sorted(unique_values, reverse=descending)
                mapping = {v: weight_spec.base + i * weight_spec.step for i, v in enumerate(sorted_values)}
            # Update the weight for each row
            self.logger.debug("Set weight for %s => %s", col_name, mapping)
            df = df.with_columns(
                pl.col("param_weight") + pl.col(col_name).replace(mapping, default=0, return_dtype=pl.Decimal))

        self.df = df.sort(by="param_weight", descending=descending)


class TVRSParamsMixin:
    """
    Mixin for analysis tasks that use the common parameterization:
    target: the CHERI compilation target, e.g. purecap/hybrid
    variant: benchmark variant, e.g. package built with stack zero-init
    runtime: run-time configuration, e.g. with/without temporal memory safety
    scenario: benchmark-specific parameters
    """
    task_namespace = "analysis"
    task_config_class = TVRSPlotConfig

    def tvrs_config(self) -> TVRSPlotConfig:
        return self.config

    def make_param_context(self, df: pl.DataFrame) -> TVRSParamsContext:
        return TVRSParamsContext(self, df)

    def get_param_axis(self, name: str) -> pl.Series:
        """
        Fetch the unique values for a given parameterization axis.

        The `target` parameter is generated from dataset_gid if needed.
        """
        if name == "target":
            gids = self.session.parameterization_matrix["instance"].unique()
            values = [self.g_uuid_to_label(gid) for gid in gids]
        else:
            values = self.session.parameterization_matrix[name].unique()
        return values

    @contextmanager
    def config_plotting_context(self, **defaults):
        """
        Enter plotting context that overrides plot configuration
        using the plot_params overrides from the TVRSPlotConfig.
        """
        config = self.tvrs_config()
        defaults.update(config.plot_params)
        font_scale = defaults.pop("font_scale", 1)
        with sns.plotting_context(font_scale=font_scale, rc=defaults):
            yield
