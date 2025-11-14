import operator
import re
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from enum import Enum
from functools import reduce
from typing import Annotated, Any, Callable, Iterable, Self
from warnings import deprecated

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
from marshmallow import ValidationError
from marshmallow import fields as mf
from marshmallow import validate as mv
from marshmallow import validates_schema
from matplotlib.patches import Patch

from ..analysis import ParamFilterConfig
from ..artefact import Target
from ..config import Config, config_field
from ..error import ConfigurationError
from ..util import bytes2int

COLREF_PATTERN = r"^<[\w_-]+>$"
#: Custom type that expresses column references.
#: Column references are names delimited by <> that reference
#: a column in the grid dataframes.
#: These are used to sanitize the use dataframe columns in the plot
#: configurations, and provide a way to reference a column independently
#: of any rename / remap.
ColRef = Annotated[str,
                   mf.String(validate=mv.Regexp(COLREF_PATTERN, error="Invalid column ref {input}, must be '<ref>'"))]


@dataclass
class PlotConfigBase(Config):
    """
    Base class for all plot configs.

    This is used as the base class to absorb calls to common plot methods.
    """
    def uses_param(self, name: str) -> bool:
        """
        Check if the given parameter axis is used by the configuration
        """
        return False


class ColumnTransform(Enum):
    #: Create an alias column, without changing the contents.
    Identity = "identity"
    #: Map values from the column to a new value, non exhaustive.
    Map = "map"
    #: Concatenate columns (as strings)
    StrCat = "concat"


@dataclass
class DerivedColumnConfig(Config):
    """
    Describe a derived column.

    Derived columns are generated before sorting, so weights may be specificed in
    terms of derived columns as well.
    Similarly, tiling and other plot parameters can reference derived columns.

    Note the following conventions:
     - names starting with _ are reserved for internal use and for task-specific auxiliary columns.
    """
    ref: str = config_field(Config.REQUIRED,
                            desc="Unique identifier for the derived column. This will become available as <ref>.")
    transform: ColumnTransform = config_field(Config.REQUIRED,
                                              by_value=True,
                                              desc="How to generate transformed column.")
    src: str | list[str] = config_field(Config.REQUIRED, desc="Source column or columns. This must be a <ref>.")
    name: str | None = config_field(None, desc="Human readable name for the column, defaults to the ref identifier.")
    args: dict[str, Any] = config_field(dict, desc="Transformation arguments.")

    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = ref


class SortOrder(Enum):
    Ascending = "ascending"
    Descending = "descending"


@dataclass
class PlotGridConfig(PlotConfigBase):
    # Data manipulation configuration
    derived_columns: list[DerivedColumnConfig] = config_field(
        list,
        desc="Auxiliary columns. These can be used to arbitrarily rename headers and data, which then are "
        "used as part of the plot. The dictionary is keyed on the resulting derived column ColRef.")
    sort_descending: bool = config_field(False, desc="Sort the grid data in ascending or descending order.")
    sort_by: list[ColRef] = config_field(list, desc="Define sort keys for the grid data.")
    sort_order: dict[ColRef, list[Any]] = config_field(dict, desc="Custom categorical ordering for specific columns.")

    # General figure-level configuration
    title: str | None = config_field(None, desc="Override figure title.")
    size: tuple[float, float] | None = config_field(None, desc="Override figure size, in inches.")
    plot_params: dict[str, Any] = config_field(
        dict, desc="Plot appearance configuration for tweaking, see matplotlib rc_context documentation")

    # Tiling configuration
    tile_row: ColRef | None = config_field(None, desc="Set column ref to use for grid rows.")
    tile_col: ColRef | None = config_field(None, desc="Set column ref to use for grid cols.")
    tile_xlabel: ColRef | None = config_field(
        None, desc="Set column ref to use for X axis labels. Defaults to X data column name.")
    tile_ylabel: ColRef | None = config_field(
        None, desc="Set column ref to use for Y axis labels. Defaults to Y data column name.")
    tile_row_show_title: bool = config_field(True, desc="Show the row tiling title.")
    tile_col_show_title: bool = config_field(True, desc="Show the column tiling title.")
    tile_sharex: str | bool = config_field("col", desc="Override X axis sharing.")
    tile_sharey: str | bool = config_field("row", desc="Override Y axis sharing.")

    # Tile sizing configuration
    tile_aspect: float = config_field(1.0, desc="Aspect ratio of each tile.")
    tile_height_ratios: list[float] | None = config_field(None, desc="Height ratios for each row.")

    # Per-tile configuration
    tile_x_margin: float | tuple[float, float] | None = config_field(
        None, desc="X-axis margin within each tile, in normalised interval units range [0, 1].")
    tile_y_margin: float | tuple[float, float] | None = config_field(
        None, desc="Y-axis margin within each tile, in normalised interval untis range [0, 1].")
    hue: ColRef | list[ColRef] = config_field(
        None,
        desc="Set ColRef to use for the hue. If multiple levels are needed, consider creating a derived column. "
        "When no hue is given, the color is controlled by the plot_param axes.prop_cycle (See matplotlib rcParams).")
    hue_colors: dict[str | tuple[str, ...], str] | None = config_field(
        None, desc="Optional map that forces a specific color for each value of the 'hue' configuration.")

    # Legend configuration
    # This should be a separate object perhaps
    # Would be nice to support different strategies:
    # outer-top, outer-bottom, outer-left, outer-right and inner legend.
    legend_position: str = config_field("outer", desc="Control legend positioning.")
    legend_vspace: float = config_field(0.2, desc="Fraction of the vertical space reserved to legend and suptitle.")
    legend_hide: bool = config_field(False, desc="Disable the legend.")
    legend_columns: int = config_field(
        4, desc="Number of columns in the legend, this affects the necessary vspace and tile_aspect.")

    @property
    def hue_levels(self):
        """
        Number of hue levels to group on.
        """
        if isinstance(self.hue, tuple):
            return len(self.hue)
        else:
            return 1

    def with_config_default(self, **kwargs):
        """
        Clone the configuration with the given default values.
        """
        defaults = {k: v for k, v in kwargs.items() if hasattr(self, k) and getattr(self, k) is None}
        config = replace(self, **defaults)
        return config

    @deprecated("Use PlotGridConfig.with_config_default() instead")
    def set_default(self, **kwargs):
        return self.with_config_default(**kwargs)

    @deprecated("Use PlotGridConfig.with_config_default() instead")
    def setdefault(self, **kwargs):
        return self.with_config_default(**kwargs)

    def set_fixed(self, **kwargs) -> Self:
        """
        Force a parameter to a fixed value.
        If the user specified an override, generate a warning.
        """
        for key, value in kwargs.items():
            if getattr(self, key) is not None:
                self.logger.warning("Configuration %s is disabled in this task, forced to %s", key, value)
        config = replace(self, **kwargs)
        return config

    def uses_param(self, name: str) -> bool:
        ref = f"<{name}>"
        if super().uses_param(ref) or self.tile_row == ref or self.tile_col == ref:
            return True
        if self.hue and isinstance(self.hue, list):
            for hue_key in self.hue:
                if hue_key == ref:
                    return True
        if self.hue == ref:
            return True
        return False


class WeightMode(Enum):
    AscendingAsInt = "ascending_int"
    DescendingAsInt = "descending_int"
    AscendingAsBytes = "ascending_bytes"
    DescendingAsBytes = "descending_bytes"
    AscendingAsStr = "ascending_str"
    DescendingAsStr = "descending_str"
    AscendingCat = "ascending_cat"
    DescendingCat = "descending_cat"
    Custom = "custom"


class SimpleWeightStrategy:
    """
    The weights generated by this strategy use the base
    Strategy object that generates weights for a series of unique values.
    """
    def __init__(self, cast_dtype: type, base: float = 0.0, step: float = 1.0, order=SortOrder.Ascending):
        self.cast_dtype = cast_dtype
        self.base = base
        self.step = step
        self.order = order

    def _transform_series(self, values: pl.Series):
        return values.cast(self.cast_dtype)

    def __call__(self, values: pl.Series) -> dict[any, float]:
        is_descending = self.order == SortOrder.Descending
        values = sorted(self._transform_series(values), reverse=is_descending)
        mapping = {v: self.base + i * self.step for i, v in enumerate(values)}
        return mapping


class BytesWeightStrategy(SimpleWeightStrategy):
    """
    Map a string with K/M/G suffixes to the corresponding bytes value,
    then sort as integer.
    """
    def __init__(self, base: float = 0.0, step: float = 1.0, order=SortOrder.Ascending):
        super().__init__(pl.Int64, base, step, order)

    def _transform_series(self, values: pl.Series):
        return values.map_elements(bytes2int, return_dtype=pl.Int64)


class CustomWeightStrategy:
    """
    Apply custom weights to each value.
    """
    def __init__(self, weights):
        self.weights = weights

    def __call__(self, values: pl.Series) -> dict[any, float]:
        return self.weights


class CategoricalWeightStrategy:
    """
    Use given sort order to assign weights.
    """
    def __init__(self, elements, base: float = 0.0, step: float = 1.0, order=SortOrder.Ascending):
        self.elements_order = elements
        self.base = base
        self.step = step
        self.order = order

    def __call__(self, values: pl.Series) -> dict[any, float]:
        nvalues = len(self.elements_order)
        weights = self.base + np.arange(nvalues) * self.step
        if self.order == SortOrder.Descending:
            # Reverse array
            weights = weights[::-1]
        mapping = dict(zip(self.elements_order, weights))
        return mapping


@dataclass
class ParamWeight(Config):
    """
    Parameter weighting rule for stable output ordering.
    """
    mode: WeightMode = config_field(WeightMode.AscendingAsStr, desc="Strategy for assigning weights")
    base: int | None = config_field(None, desc="Base weight, ignored for 'custom' strategy.")
    step: int | None = config_field(None, desc="Weight increment, ignored for 'custom' strategy.")
    weights: dict[str, int] | None = config_field(
        None, desc="Custom mapping of parameter values to weights, only valid for 'custom' mode.")
    order: list[str] | None = config_field(None, desc="Custom order of parameter values, only valid for '*_cat' modes.")

    @validates_schema
    def validate_mode(self, data, **kwargs):
        mode_name = data["mode"].value

        def _required(field_name):
            if data[field_name] is None:
                raise ValidationError(f"ParamWeight.{field_name} must be set when ParamWeight.mode = '{mode_name}'", )

        def _ignored(field_name):
            if data[field_name] is not None:
                raise ValidationError(f"ParamWeight.{field_name} is ignored when ParamWeight.mode = '{mode_name}'")

        match data["mode"]:
            case WeightMode.Custom:
                _required("weights")
                _ignored("order")
                _ignored("base")
                _ignored("step")
            case WeightMode.AscendingCat | WeightMode.DescendingCat:
                _required("order")
                _required("base")
                _required("step")
                _ignored("weights")
            case _:
                _required("base")
                _required("step")
                _ignored("weights")
                _ignored("order")

    def get_strategy(self) -> Callable[[pl.Series], dict[any, float]]:
        """
        Build the weight generation strategy
        """
        def _simple_strategy(dtype, order):
            return SimpleWeightStrategy(dtype, self.base, self.step, order)

        match self.mode:
            case WeightMode.AscendingAsStr:
                return _simple_strategy(pl.String, SortOrder.Ascending)
            case WeightMode.DescendingAsStr:
                return _simple_strategy(pl.String, SortOrder.Descending)
            case WeightMode.AscendingAsInt:
                return _simple_strategy(pl.Int64, SortOrder.Ascending)
            case WeightMode.DescendingAsInt:
                return _simple_strategy(pl.Int64, SortOrder.Descending)
            case WeightMode.AscendingAsBytes:
                return BytesWeightStrategy(self.base, self.step, SortOrder.Ascending)
            case WeightMode.DescendingAsBytes:
                return BytesWeightStrategy(self.base, self.step, SortOrder.Descending)
            case WeightMode.AscendingCat:
                return CategoricalWeightStrategy(self.order, self.base, self.step, SortOrder.Ascending)
            case WeightMode.DescendingCat:
                return CategoricalWeightStrategy(self.order, self.base, self.step, SortOrder.Descending)
            case WeightMode.Custom:
                return CustomWeightStrategy(self.weights)
            case _:
                raise ConfigurationError("Invalid WeightMode")


@dataclass
class DisplayGridConfig(PlotGridConfig):
    """
    Dataframe context renaming and sorting configuration

    This should be inherited by all analysis task configurations that use renaming.
    Note that the display renaming/mapping is not allowed to change the behaviour of
    group_by over parameterization axes.
    """
    param_sort_weight: dict[str, ParamWeight] | None = config_field(
        None,
        desc="Weight for determining the order of labels based on the parameters. "
        "The dictionary key must be a parameter name, the value is a weight descriptor.")

    param_sort_order: SortOrder = config_field(SortOrder.Ascending,
                                               desc="Sort order by ascending or descending weight.")

    param_names: dict[str, str] | None = config_field(
        None, desc="Relabel parameter keys specified in PipelineBenchmarkConfig.parameterize")

    param_values: dict[str, dict[str, Any]] | None = config_field(
        None,
        desc="Relabel parameter values. The key is a parameter name, the value "
        "is a mapping of the form { <parameter value> => <value alias> }.")

    param_filter: ParamFilterConfig | None = config_field(
        None, desc="Further filter the data for the given set of parameters.")

    derived_columns: list[DerivedColumnConfig] = config_field(
        list, desc="List of derived columns. These can be used to rename columns or elements within a column.")

    @deprecated("Use with_default_axis_rename and with_default_axis_remap")
    def set_display_defaults(self,
                             param_names: dict[str, any] | None = None,
                             param_values: dict[str, dict[str, any]] | None = None) -> Self:
        """
        Set default renames
        """
        config = self
        if param_names:
            config = config.with_default_axis_rename(param_names)
        if param_values:
            config = config.with_default_axis_remap(param_values)
        return config

    def with_default_axis_rename(self, rename_map: dict[str, any]) -> Self:
        """
        Clone the configuration with the given default column renaming.
        """
        config = replace(self)
        rename_map.update(config.param_names or {})
        config.param_names = rename_map
        return config

    def with_default_axis_remap(self, rename_map: dict[str, dict[str, any]]) -> Self:
        """
        Clone the configuration with the given column values renaming.
        """
        config = replace(self)
        if config.param_values is None:
            config.param_values = {}
        for key, rename in rename_map.items():
            rename.update(config.param_values.get(key, {}))
            config.param_values[key] = rename
        return config


class ColumnNameMapping:
    """
    Map internal raw column names to a dataframe column.

    This may be used to rename columns or contents to human-readable values.
    This allows deeper configuration of the plot appearence when drawing the
    grid tiles.
    """
    def __init__(self, rename: dict[str, str]):
        self._rename = rename

    def __getattr__(self, name: str) -> str:
        return self._rename.get(name, name)

    def __getitem__(self, name: str | None) -> str | None:
        return self._rename.get(name, name)


class PlotTile:
    def __init__(self, grid: "PlotGrid", ax: "Axes", hue: str | None, palette: list | None, coords: tuple[int, int],
                 loc: tuple[any, any]):
        self.grid = grid
        self.ax = ax
        self.hue = hue
        # Note that the palette maps hue labels to colors
        self.palette = palette
        self.coords = coords
        self.loc = loc

    @property
    def row_index(self):
        return self.coords[0]

    @property
    def col_index(self):
        return self.coords[1]

    @property
    def row(self):
        return self.loc[0]

    @property
    def col(self):
        return self.loc[1]

    def ref_to_col(self, ref: str) -> str:
        return self.grid.ref_to_col(ref)


class DisplayTile(PlotTile):
    def __init__(self, grid, ax, hue, palette, coords, loc, raw_map: ColumnNameMapping, display_map: ColumnNameMapping):
        super().__init__(grid, ax, hue, palette, coords, loc)
        # Mapping from the standard column name to the raw data column name
        self.raw_map = raw_map
        # Mapping from the standard column name to the display column name
        self.display_map = display_map

    @property
    def d(self):
        return self.display_map

    @property
    def r(self):
        return self.raw_map


class DataFrameInfo:
    """
    Documents and specifies columns available for processing in a dataframe.

    XXX separate data and metadata columns
    """
    def __init__(self, desc: dict[str, str]):
        self.columns = desc

    @classmethod
    def from_session_params(cls, task: "AnalysisTask", **extra_columns) -> Self:
        desc = {}
        for param in task.param_columns:
            desc[param] = f"User-defined parameterization axis '{param}'"
        desc.update(extra_columns)
        return cls(desc)

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame) -> Self:
        desc = {c: f"Unknown column" for c in df.columns}
        return cls(desc)


class PlotGrid(AbstractContextManager):
    """
    Abstraction to generate grid plots based on polars dataframes.

    This aims to have some similarity to seaborn FacetGrid while being more
    efficient with polars dataframes.
    """
    def __init__(self, target: Target, data: pl.DataFrame, config: PlotGridConfig, info: DataFrameInfo = None):
        assert isinstance(config, PlotGridConfig)
        self._target = target
        self._df = data
        self._config = config
        # May override height if needed
        self._height = 3
        self._figure = None
        self._grid = None
        self._margin_titles = []
        # This will be populated after the frame sort order is known
        # dictionary mapping hue column value to a color
        self._color_palette = None
        self._rc_context = None
        # This is set when legend is enabled, kwargs may be an empty dict
        self._legend_kwargs = None

        # Map column reference keys (<name>) to the current column name in the
        # dataframe.
        # If a ref does not exist, the column can not be named in the configuration.
        # This ensures that internal columns are not manipulated.
        self._column_refs_map = {}
        if info is None:
            info = DataFrameInfo.from_dataframe(data)
        self._init_column_refs(info)

    def __enter__(self):
        self._gen_derived_columns()
        self._sort()
        # self._filter()

        # XXX verify that sorting is unique across the free axes.

        # Determine tiling row and column parameters
        nrows, ncols = self._grid_shape()
        # Compute the figure size based on the configured rows and columns
        # and initialize the figure
        if self._config.size:
            fig_size = self._config.size
        else:
            fig_size = (ncols * self._height * self._config.tile_aspect, (nrows + 1) * self._height)
        self._rc_context = plt.rc_context(rc=self._config.plot_params)
        self._rc_context.__enter__()

        self._figure = plt.figure(figsize=fig_size, layout="constrained")
        # Initialize the grid
        kwargs = {
            "squeeze": False,
            "sharex": self._config.tile_sharex,
            "sharey": self._config.tile_sharey,
        }

        if ratios := self._config.tile_height_ratios:
            if len(ratios) != nrows:
                self.logger.error("Invalid height ratios configuration, expected %d rows", nrows)
                raise ConfigurationError("Invalid configuration")
            kwargs["height_ratios"] = ratios

        self._grid = self._figure.subplots(nrows, ncols, **kwargs)

        if self._config.title:
            self._figure.suptitle(self._config.title)
        self._gen_color_palette()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.map(self._set_titles)
        self._finalize_layout()
        for path in self._target.paths():
            self._figure.savefig(path, bbox_inches="tight")
        plt.close(self._figure)
        self._rc_context.__exit__(exc_type, exc_value, traceback)

    def _grid_shape(self) -> tuple[int, int]:
        nrows, ncols = 1, 1
        if row_param := self.tile_row:
            nrows = self._df[row_param].n_unique()
        if col_param := self.tile_col:
            ncols = self._df[col_param].n_unique()
        return nrows, ncols

    def _grid_rows(self, df: pl.DataFrame) -> Iterable[tuple[any, pl.DataFrame]]:
        if row_param := self.tile_row:
            for (name, ), chunk in df.group_by(row_param, maintain_order=True):
                yield (name, chunk)
        else:
            yield (None, df)

    def _grid_cols(self, df: pl.DataFrame) -> Iterable[tuple[any, pl.DataFrame]]:
        if col_param := self.tile_col:
            for (name, ), chunk in df.group_by(col_param, maintain_order=True):
                yield (name, chunk)
        else:
            yield (None, df)

    def _init_column_refs(self, info: DataFrameInfo):
        for col in info.columns.keys():
            self._register_ref(col, col)

    def _register_ref(self, ref: str, name: str):
        if ref in self._column_refs_map:
            self.logger.error("Attempt to re-define column ref %s to %s", ref, name)
            raise ConfigurationError("Configuration error")
        self._column_refs_map[f"<{ref}>"] = name

    def _gen_derived_identity(self, spec: DerivedColumnConfig):
        src = self.ref_to_col(spec.src)
        if spec.name in self._df.columns:
            self.logger.error("Invalid name %s for target column, column exists", spec.name)
            raise ConfigurationError("Invalid configuration")
        self._df = self._df.with_columns(pl.col(src).alias(spec.name))
        self._register_ref(spec.ref, spec.name)

    def _gen_derived_map(self, spec: DerivedColumnConfig):
        src = self.ref_to_col(spec.src)
        if spec.name in self._df.columns:
            self.logger.error("Invalid name %s for target column, column exists", spec.name)
            raise ConfigurationError("Invalid configuration")
        dtypes = {type(v) for v in spec.args.values()}
        if len(dtypes) > 1:
            self.logger.error("Derived column %s map to inconsistent dtypes", spec.name, dtypes)
            raise ConfigurationError("Invalid configuration")
        dtype = dtypes.pop()
        if dtype is str:
            src_expr = pl.col(src).cast(pl.String)
        else:
            src_expr = pl.col(src)
        self._df = self._df.with_columns(src_expr.replace(spec.args).alias(spec.name))
        self._register_ref(spec.ref, spec.name)

    def _gen_derived_columns(self):
        """
        Generate derived columns according to the configuration.

        Note that new columns will be made available in the column_refs_map.
        """
        for spec in self._config.derived_columns:
            match spec.transform:
                case ColumnTransform.Identity:
                    self._gen_derived_identity(spec)
                case ColumnTransform.Map:
                    self._gen_derived_map(spec)
                case _:
                    self.logger.error("Invalid column transform %s", spec.transform)
                    raise RuntimeError("Unsupported column transformation")

    def _sort(self):
        """
        Sort the dataframe according to the configuration.
        """
        if not self._config.sort_by:
            self.logger.warning("Sorting is not configured, use grid tiling order, this may not be stable.")
            sort_by = [self._config.tile_row, self._config.tile_col, self._config.hue]
            sort_by = [v for v in sort_by if v is not None]
        else:
            sort_by = self._config.sort_by
        sort_by = [self.ref_to_col(ref) for ref in sort_by]

        # If there is no custom ordering of categorical column values
        # we can just sort the frame and rely on the internal dtype ordering support.
        # otherwise, create a custom weight column.
        if not self._config.sort_order:
            self._df = self._df.sort(by=sort_by, descending=self._config.sort_descending)
            return

        sort_order = {self.ref_to_col(ref): order for ref, order in self._config.sort_order.items()}
        df = self._df.with_columns(pl.lit(0.0).alias("_sort_weight"))
        weight_base = 0
        for col in reversed(sort_by):
            data_order = sort_order.get(col)
            data_values = df[col].unique()
            n_weights = len(data_values)
            if not data_order:
                data_order = data_values.sort()
            else:
                # Verify that the data order is a superset of the data
                # in the column.
                if not set(data_values).issubset(data_order):
                    self.logger.error("Invalid sort weights mapping for %s, requires %s", col, data_values)
                    raise ConfigurationError("Invalid sort_order configuration")
            n_weights = max(len(data_order), n_weights)
            weights = weight_base + np.arange(n_weights)
            self.logger.debug("Assign weights to column %s: %s", col, list(zip(data_order, weights)))
            for value, weight in zip(data_order, weights):
                df = df.with_columns(pl.col("_sort_weight") + pl.when(pl.col(col) == value).then(weight).otherwise(0))
            weight_base += n_weights
        self._df = df.sort(by="_sort_weight", descending=self._config.sort_descending)

    def _gen_color_palette(self):
        """
        Generate the color palette for the grid tiles.

        When hue is a string, this is interpreted as a single column and we will
        associate a color for each unique value in the column.
        When hue is a tuple, this is interpreted as multiple columns and we will
        associate a color for each unique combination of the values over the
        given columns in the tuple.

        In the latter case, when the color map is explicit, the hue_colors must
        be keyed on compatible tuples containing the matching values for each column
        combination.

        Note that the _color_palette dictionary is always keyed on a tuple, this
        removes complexity on the plot implementations, because when grouping on
        the hue level(s) the group label is always going to be a tuple.
        """
        if self._config.hue is None:
            # Generate synthetic hue column with an uniform value
            self._df = self._df.with_columns(pl.lit(0).alias("_default_hue"))
            self._color_palette = {0: plt.rcParams["axes.prop_cycle"].by_key()["color"][0]}
        else:
            # Keep the order so that the user has control on the mapping between
            # data points and hue colors even when not using explicit color mapping.
            hue_keys = self._df[self.tile_hue].unique(maintain_order=True)
            ncolors = len(hue_keys)
            if self._config.hue_colors is not None:
                if len(self._config.hue_colors) != ncolors:
                    self.logger.error("Hue level %s: invalid number of colors in configuration: expected %d, found %d",
                                      self._config.hue, ncolors, len(self._config.hue_colors))
                    raise ConfigurationError("Configuration error")
                self._color_palette = {h: self._config.hue_colors[h] for h in hue_keys}
            else:
                colors = sns.color_palette(n_colors=ncolors)
                self._color_palette = dict(zip(hue_keys, colors))

    def _set_titles(self, tile, chunk):
        """
        Set row and column labels.
        Suppress unwanted X / Y axis labels and handle labeling overrides.
        """
        nrows, ncols = self._grid_shape()
        if col_param := self.tile_col:
            # Only add the tile header to the first row of the grid.
            if tile.row_index == 0 and self._config.tile_col_show_title:
                tile.ax.set_title(f"{col_param} = {tile.col}")

        if row_param := self.tile_row:
            # Clear shared Y axis labels if this isn't the first column
            if self._config.tile_sharey and tile.col_index > 0:
                tile.ax.set_ylabel(None).set_visible(False)
            # Annotate the row group at the end of the row
            if tile.col_index == ncols - 1 and self._config.tile_row_show_title:
                text = tile.ax.annotate(f"{row_param} = {tile.row}",
                                        xy=(1.02, .5),
                                        xycoords="axes fraction",
                                        rotation=270,
                                        ha="left",
                                        va="center")
                self._margin_titles.append(text)

        # Handle X and Y labels override
        def _sanitize_label_override(label_ref):
            label_col = self.ref_to_col(label_ref)
            if chunk[label_col].n_unique() > 1:
                self.logger.warning(
                    "Can not determine Y axis label for tile (%s, %s) from %s; non-unique label in chunk", tile.row,
                    tile.col, label_ref)
                return "unknown"
            return chunk[label_col].first()

        if self._config.tile_sharex and tile.row_index != nrows - 1:
            tile.ax.set_xlabel(None)
        elif label_ref := self._config.tile_xlabel:
            label = _sanitize_label_override(label_ref)
            tile.ax.set_xlabel(label)
        if self._config.tile_sharey and tile.col_index != 0:
            tile.ax.set_ylabel(None)
        elif label_ref := self._config.tile_ylabel:
            label = _sanitize_label_override(label_ref)
            tile.ax.set_ylabel(label)

    def _make_tile(self, ax: "Axes", ri: int, ci: int, row_value: str, col_value: str):
        return PlotTile(grid=self,
                        ax=ax,
                        hue=self.tile_hue,
                        palette=self._color_palette,
                        coords=(ri, ci),
                        loc=(row_value, col_value))

    def _finalize_margins(self):
        # Helper internal function to set margin along an axis
        def _set_margins(tile, _, margins: tuple[float, float] = None, dimension: str = None):
            # Note we conver from data to display coordinates and from display to axes.
            # Axes coordinates are in the space [0, 1] x [0, 1].
            axes_space_tf = tile.ax.transData + tile.ax.transAxes.inverted()
            data_space_tf = tile.ax.transAxes + tile.ax.transData.inverted()

            match dimension:
                case "x":
                    get_lim_fn = tile.ax.get_xlim
                    set_lim_fn = tile.ax.set_xlim
                    to_axes_space = lambda x: axes_space_tf.transform([x, 0])[0]
                    to_data_space = lambda x: data_space_tf.transform([x, 0])[0]
                    # We manage margins manually, so reset the margin here
                    # this should be redundant
                    tile.ax.set_xmargin(0)
                case "y":
                    get_lim_fn = tile.ax.get_ylim
                    set_lim_fn = tile.ax.set_ylim
                    to_axes_space = lambda y: axes_space_tf.transform([0, y])[1]
                    to_data_space = lambda y: data_space_tf.transform([0, y])[1]
                    tile.ax.set_ymargin(0)
                case _:
                    assert False, "Invalid margins axis name"
            # Compute the new axis limits according to the margins
            m_low, m_high = margins
            assert m_low >= 0 and m_low <= 1
            assert m_high >= 0 and m_high <= 1
            d_low, d_high = get_lim_fn()
            # Move to axis coordinates
            t_low = to_axes_space(d_low)
            t_high = to_axes_space(d_high)
            delta = t_high - t_low
            delta_low, delta_high = delta * m_low, delta * m_high
            t_low, t_high = t_low - delta_low, t_high + delta_high
            # Go back to data coordinates
            d_low = to_data_space(t_low)
            d_high = to_data_space(t_high)
            set_lim_fn(d_low, d_high)

        # Obey tile margin overrides
        if xm := self._config.tile_x_margin:
            if not isinstance(xm, tuple):
                xm = (xm, xm)
            self.map(_set_margins, margins=xm, dimension="x")
        if ym := self._config.tile_y_margin:
            if not isinstance(ym, tuple):
                ym = (ym, ym)
            self.map(_set_margins, margins=ym, dimension="y")

    def _finalize_layout(self):
        """
        Finalize the figure layout.

        This is called after all plotting is done to handle margins and legend.
        """
        self._finalize_margins()

        # Generate the legend, if enabled
        if self._legend_kwargs is None or self._config.legend_hide:
            return
            self._generate_legend()
        if not self._config.hue:
            return

        if self._config.legend_position == "inner":
            self.map(lambda tile, chunk: tile.ax.legend())
        elif self._config.legend_position == "outer":
            hue_keys = self._df[self.tile_hue].unique(maintain_order=True)
            labels = self._df[self.tile_hue].unique(maintain_order=True)
            patches = [Patch(color=self._color_palette[hue_key]) for hue_key in hue_keys]

            legend_handles = {}

            def _merge_legend_handles(tile, _chunk):
                handles, labels = tile.ax.get_legend_handles_labels()
                for h, l in zip(handles, labels):
                    if l not in legend_handles:
                        legend_handles[l] = h

            self.map(_merge_legend_handles)

            # DEPRECATED Make space between the title and the subplot axes
            # reserved_y_fraction = 1 - self._config.legend_vspace
            # self._figure.subplots_adjust(top=reserved_y_fraction, bottom=self._config.legend_vspace)
            # legend_anchor = (0., reserved_y_fraction, 1., self._config.legend_vspace)
            self._figure.legend(legend_handles.values(),
                                legend_handles.keys(),
                                ncols=self._config.legend_columns,
                                loc="outside upper center",
                                **self._legend_kwargs)
        else:
            raise RuntimeError(f"Invalid legend_position='{self._config.legend_position}'")

    @property
    def tile_row(self) -> str | None:
        if ref := self._config.tile_row:
            return self.ref_to_col(ref)
        return None

    @property
    def tile_col(self) -> str | None:
        if ref := self._config.tile_col:
            return self.ref_to_col(ref)
        return None

    @property
    def tile_hue(self) -> str:
        if ref := self._config.hue:
            return self.ref_to_col(ref)
        return "_default_hue"

    @property
    def logger(self):
        return self._target.task.logger

    def ref_to_col(self, ref: str | None) -> str | None:
        if ref is None:
            return None
        if ref not in self._column_refs_map:
            self.logger.error("Column ref %s is not defined", ref)
            raise ConfigurationError("Undefined column reference")
        return self._column_refs_map[ref]

    def try_ref_to_col(self, ref: str) -> str:
        """Try to resolve ref as a ColRef, if not, return it unchanged."""
        if re.match(COLREF_PATTERN, ref):
            return self.ref_to_col(ref)
        return ref

    def map(self, tile_plotter: Callable[[PlotTile, pl.DataFrame], None], *args, **kwargs):
        for i, (ax_stride, (row_param, row_chunk)) in enumerate(zip(self._grid, self._grid_rows(self._df))):
            for j, (ax, (col_param, chunk)) in enumerate(zip(ax_stride, self._grid_cols(row_chunk))):
                tile = self._make_tile(ax, i, j, row_param, col_param)
                self.logger.debug("PlotGrid: tile callback (row=%s, col=%s) %s", row_param, col_param, chunk)
                tile_plotter(tile, chunk, *args, **kwargs)

    def get_grid_df(self) -> pl.DataFrame:
        return self._df

    def add_legend(self, **kwargs):
        """
        Add the legend on top of the axes.

        The legend is generated lazily on contextmanager exit.
        """
        self._legend_kwargs = kwargs


class DisplayGrid(PlotGrid):
    """
    A display grid is a more advanced plot grid that allows renaming and filtering.

    The configuration allows renaming parameters and columns as well as sorting weight
    configuration and filtering.

    The filtering and renaming occurs when entering the context.
    """
    def __init__(self, target: Target, data: pl.DataFrame, config: DisplayGridConfig):
        assert isinstance(config, DisplayGridConfig)
        super().__init__(target, data, config)
        self._raw_map = dict()
        self._display_map = dict()

    def __enter__(self):
        self._filter()
        self._sort2()
        self._gen_display_columns()
        super().__enter__()
        return self

    def _filter(self):
        """
        Filter the dataframe according to the configuration.
        """
        if self._config.param_filter is None:
            return

        if rules := self._config.param_filter.keep:
            conditions = []
            for rule in rules:
                cond = [pl.col(k) == v for k, v in rule.matches.items()]
                conditions.append(reduce(operator.and_, cond))
            cond = reduce(operator.or_, conditions)
            self._df = self._df.filter(cond)

        if rules := self._config.param_filter.drop:
            conditions = []
            for rule in rules:
                cond = [pl.col(k) == v for k, v in rule.matches.items()]
                conditions.append(reduce(operator.and_, cond))
            cond = reduce(operator.or_, conditions)
            self._df = self._df.filter(~cond)

    def _gen_display_columns(self):
        """
        Generate human-readable columns according to the mapping policy in the
        configuration.

        This applies the plot configuration to rename parameter levels.
        Columns that have their values mapped through param_values are cloned
        and retain the original value through generated _r_<column>.

        This function populates the raw_map and display map accordingly, so
        that the raw_map always accesses columns containing the raw values and
        the display_map always accesses columns containing the display contents.
        """
        # Column contents renaming
        if self._config.param_values:
            exprs = []
            for name, mapping in self._config.param_values.items():
                if name not in self._df.columns:
                    self.logger.debug("Skipping display mapping for '%s', does not exist", name)
                    continue
                self._raw_map[name] = f"_r_{name}"
                exprs.append(pl.col(name).alias(f"_r_{name}"))
                # Mapping can either make the column a string type or map it to
                # values of the same numeric type
                mapping_dtypes = {type(v) for v in mapping.values()}
                if len(mapping_dtypes) > 1:
                    self.logger.error("param_values mapping for %s does not map to a consistent dtype: found %s", name,
                                      mapping_dtypes)
                    raise RuntimeError("Invalid configuration")
                mapped_dtype = mapping_dtypes.pop()
                if mapped_dtype is str:
                    expr = pl.col(name).cast(pl.String).replace(mapping)
                else:
                    expr = pl.col(name).replace(mapping)
                exprs.append(expr)
            self._df = self._df.with_columns(*exprs)

        # Column name renaming from columns to display column names
        # There is no point in duplicating columns here, the contents will be
        # the same for both the raw and display maps.
        if self._config.param_names:
            self._raw_map = self._config.param_names | self._raw_map
            self._display_map = dict(self._config.param_names)
            self._df = self._df.rename(self._config.param_names)

    def _generate_sort_weights(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate mapping between values in different parameterisation axes to
        a sort weight.
        This determines the ordering of data in the dataframe used for plotting.
        Note that this is critical to ensure stability and consistency of the output.
        """
        assert "_param_weight" in df.columns

        for name, weight_spec in self._config.param_sort_weight.items():
            if name not in self._df.columns:
                self.logger.info("Skipping weight for parameter '%s', does not exist", name)
                continue

            if weight_spec.mode == WeightMode.Custom:
                # Use custom weights
                mapping = dict(weight_spec.weights)
            else:
                # Use weight strategy
                strategy = weight_spec.get_strategy()
                unique_values = df[name].unique()
                mapping = strategy(unique_values)

            self.logger.debug("Assign sort weights %s => %s", name, mapping)
            df = df.with_columns(
                pl.col("_param_weight") + pl.col(name).replace(mapping, default=0.0, return_dtype=pl.Float32))
        return df

    def _sort2(self):
        """
        Sort the dataframe according to the configured param_sort_weight.

        Note that this always operates before mapping columns to the
        human-readable display view.
        """
        if not self._config.param_sort_weight:
            self.logger.info("Skipping sort(), not configured")
            return
        df = self._df.with_columns(pl.lit(0.0).alias("_param_weight"))
        df = self._generate_sort_weights(df)

        is_descending = self._config.param_sort_order == SortOrder.Descending
        self._df = df.sort(by="_param_weight", descending=is_descending)

    def _make_tile(self, ax, ri, ci, row, col):
        tile = DisplayTile(grid=self,
                           ax=ax,
                           hue=self.tile_hue,
                           palette=self._color_palette,
                           coords=(ri, ci),
                           loc=(row, col),
                           raw_map=ColumnNameMapping(self._raw_map),
                           display_map=ColumnNameMapping(self._display_map))
        return tile


def grid_debug(tile: PlotTile, chunk: pl.DataFrame, x: str, y: str, logger: "Logger"):
    """
    Helper for debugging plot generation.

    This dumps the output sorted dataframe for each tile.
    """
    logger.debug("Dump tile (%s, %s)\n%s", tile.row, tile.col, chunk)
