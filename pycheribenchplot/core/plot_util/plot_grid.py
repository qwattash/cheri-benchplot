import operator
import re
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from enum import Enum
from functools import reduce
from logging import Logger
from typing import Annotated, Any, Callable, Iterable, Self, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from marshmallow import fields as mf
from marshmallow import validate as mv

from ..analysis import ParamFilterConfig
from ..artefact import Target
from ..config import Config, config_field
from ..error import ConfigurationError

if TYPE_CHECKING:
    from matplotlib.axes import Axes

type AnalysisTask = "AnalysisTask"
type MplAxes = "Axes"

COLREF_PATTERN = r"^<[\w_-]+>$"
#: Custom type that expresses column references.
#: Column references are names delimited by <> that reference
#: a column in the grid dataframes.
#: These are used to sanitize the use dataframe columns in the plot
#: configurations, and provide a way to reference a column independently
#: of any rename / remap.
ColRef = Annotated[
    str,
    mf.String(
        validate=mv.Regexp(
            COLREF_PATTERN, error="Invalid column ref {input}, must be '<ref>'"
        )
    ),
]


@dataclass
class ColRefMap(Config):
    """
    Mapping using source values from a given column ref.
    """

    src: ColRef = config_field(Config.REQUIRED, desc="ColRef of source values.")
    values: dict[str, Any] = config_field(
        Config.REQUIRED, desc="Values associated to each element in <src>."
    )


@dataclass
class PlotConfigBase(Config):
    """
    Base class for all plot configs.
    """

    pass


class ColumnTransform(Enum):
    #: Create an alias column, without changing the contents.
    Identity = "identity"
    #: Map values from the column to a new value, non exhaustive.
    Map = "map"
    #: Map values to byte size (with suffix)
    MapToBytes = "map2bytes"
    #: Concatenate columns (as strings)
    StrCat = "concat"


def int2bytes(value: int) -> str:
    """
    Coerce bytes value into a string with optional K/M/G/T suffix corresponding
    to the bytes size.
    """
    SUFFIXES = ["B", "KiB", "MiB", "GiB", "TiB"]

    idx = 0
    while value >= 2**10:
        idx += 1
        value /= 2**10
    if idx >= len(SUFFIXES):
        return f"{int(value)} 2^{10 * idx}"
    return f"{int(value)}{SUFFIXES[idx]}"


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

    transform: ColumnTransform = config_field(
        Config.REQUIRED, by_value=True, desc="How to generate transformed column."
    )
    src: str | list[str] = config_field(
        Config.REQUIRED, desc="Source column or columns. This must be a <ref>."
    )
    name: str | None = config_field(None, desc="Human readable name for the column.")
    args: dict[str, Any] = config_field(dict, desc="Transformation arguments.")


@dataclass
class PlotGridConfig(PlotConfigBase):
    # Data manipulation configuration
    derived_columns: dict[str, DerivedColumnConfig] = config_field(
        dict,
        desc="Auxiliary columns. These can be used to arbitrarily rename headers and data, which then are "
        "used as part of the plot. The dictionary is keyed on the resulting derived column ColRef name.",
    )
    sort_descending: bool = config_field(
        False, desc="Sort the grid data in ascending or descending order."
    )
    sort_by: list[ColRef] = config_field(
        list, desc="Define sort keys for the grid data."
    )
    sort_order: dict[ColRef, list[Any]] = config_field(
        dict, desc="Custom categorical ordering for specific columns."
    )
    data_filter: ParamFilterConfig | None = config_field(
        None,
        desc="Further filter the data for the given set of parameters. Note that this expects ColRef keys.",
    )

    # General figure-level configuration
    title: str | None = config_field(None, desc="Set figure title.")
    size: tuple[float, float] | None = config_field(
        None, desc="Override figure size, in inches."
    )
    style: dict[str, Any] = config_field(
        dict,
        desc="Plot appearance configuration for tweaking, see matplotlib rc_context documentation.",
    )

    # Tiling configuration
    tile_row: ColRef | None = config_field(
        None, desc="Set column ref to use for grid rows."
    )
    tile_col: ColRef | None = config_field(
        None, desc="Set column ref to use for grid cols."
    )
    tile_xlabel: ColRef | None = config_field(
        None,
        desc="Set column ref to use for X axis labels. Defaults to X data column name.",
    )
    tile_ylabel: ColRef | None = config_field(
        None,
        desc="Set column ref to use for Y axis labels. Defaults to Y data column name.",
    )
    tile_row_show_title: bool = config_field(True, desc="Show the row tiling title.")
    tile_col_show_title: bool = config_field(True, desc="Show the column tiling title.")
    tile_sharex: str | bool = config_field("col", desc="Override X axis sharing.")
    tile_sharey: str | bool = config_field("row", desc="Override Y axis sharing.")

    # Tile sizing configuration
    tile_aspect: float = config_field(1.0, desc="Aspect ratio of each tile.")
    tile_height_ratios: ColRefMap | ColRef | dict[str, float] | None = config_field(
        None,
        desc="Height ratios for each row. A ColRef, or a mapping of <tile_row> values to a float.",
    )

    # Per-tile configuration
    tile_x_margin: float | tuple[float, float] | None = config_field(
        None,
        desc="X-axis margin within each tile, in normalised interval units range [0, 1].",
    )
    tile_y_margin: float | tuple[float, float] | None = config_field(
        None,
        desc="Y-axis margin within each tile, in normalised interval untis range [0, 1].",
    )
    hue: ColRef | list[ColRef] = config_field(
        None,
        desc="Set ColRef to use for the hue. If multiple levels are needed, consider creating a derived column. "
        "When no hue is given, the color is controlled by the plot_param axes.prop_cycle (See matplotlib rcParams).",
    )
    hue_colors: dict[str | tuple[str, ...], str] | None = config_field(
        None,
        desc="Optional map that forces a specific color for each value of the 'hue' configuration.",
    )

    # Legend configuration
    # Would be nice to support different strategies:
    # outer-top, outer-bottom, outer-left, outer-right and inner legend.
    legend_hide: bool = config_field(False, desc="Disable the legend.")
    legend_position: str = config_field("outer", desc="Control legend positioning.")
    legend_columns: int = config_field(
        4,
        desc="Number of columns in the legend, this affects the necessary vspace and tile_aspect.",
    )

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
        defaults = {
            k: v
            for k, v in kwargs.items()
            if hasattr(self, k) and getattr(self, k) is None
        }
        config = replace(self, **defaults)
        return config

    def set_fixed(self, **kwargs) -> Self:
        """
        Force a parameter to a fixed value.
        If the user specified an override, generate a warning.
        """
        for key, value in kwargs.items():
            if getattr(self, key) is not None:
                self.logger.warning(
                    "Configuration %s is disabled in this task, forced to %s",
                    key,
                    value,
                )
        config = replace(self, **kwargs)
        return config


class PlotTile:
    def __init__(
        self,
        grid: "PlotGrid",
        ax: MplAxes,
        hue: str | None,
        palette: list | None,
        coords: tuple[int, int],
        loc: tuple[any, any],
    ):
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


class DataFrameInfo:
    """
    Documents and specifies columns available for processing in a dataframe.

    XXX separate data and metadata columns
    """

    def __init__(self, desc: dict[str, str]):
        self.columns = desc

    @classmethod
    def from_session_params(cls, task: AnalysisTask, **extra_columns) -> Self:
        desc = {}
        for param in task.param_columns:
            desc[param] = f"User-defined parameterization axis '{param}'"
        desc.update(extra_columns)
        return cls(desc)

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame) -> Self:
        desc = {c: "Unknown column" for c in df.columns}
        return cls(desc)


class PlotGrid(AbstractContextManager):
    """
    Abstraction to generate grid plots based on polars dataframes.

    This aims to have some similarity to seaborn FacetGrid while being more
    efficient with polars dataframes.
    """

    def __init__(
        self,
        target: Target,
        data: pl.DataFrame,
        config: PlotGridConfig,
        info: DataFrameInfo = None,
    ):
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
        self._filter()

        # XXX verify that sorting is unique across the free axes.

        # Determine tiling row and column parameters
        nrows, ncols = self._grid_shape()
        # Compute the figure size based on the configured rows and columns
        # and initialize the figure
        if self._config.size:
            fig_size = self._config.size
        else:
            fig_size = (
                ncols * self._height * self._config.tile_aspect,
                (nrows + 1) * self._height,
            )
        self._rc_context = plt.rc_context(rc=self._config.style)
        self._rc_context.__enter__()

        self._figure = plt.figure(figsize=fig_size, layout="constrained")
        # Initialize the grid
        kwargs = {
            "squeeze": False,
            "sharex": self._config.tile_sharex,
            "sharey": self._config.tile_sharey,
        }

        if height_spec := self._config.tile_height_ratios:
            if isinstance(height_spec, str):
                ratio_col = self.ref_to_col(height_spec)
                values = self._df[ratio_col].unique(maintain_order=True)
            else:
                if isinstance(height_spec, ColRefMap):
                    ratio_col = self.ref_to_col(height_spec.src)
                    ratio_map = height_spec.values
                else:
                    assert isinstance(height_spec, dict)
                    ratio_col = self.tile_row
                    ratio_map = height_spec
                if len(ratio_map) < nrows:
                    self.logger.error(
                        "Invalid height ratios mapping, expected %d values got %s",
                        nrows,
                        ratio_map,
                    )
                    raise ConfigurationError("Invalid configuration")
                values = (
                    self._df[ratio_col]
                    .unique(maintain_order=True)
                    .replace(ratio_map)
                    .cast(pl.Float32)
                )
            kwargs["height_ratios"] = values

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
            for (name,), chunk in df.group_by(row_param, maintain_order=True):
                yield (name, chunk)
        else:
            yield (None, df)

    def _grid_cols(self, df: pl.DataFrame) -> Iterable[tuple[any, pl.DataFrame]]:
        if col_param := self.tile_col:
            for (name,), chunk in df.group_by(col_param, maintain_order=True):
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

    def _gen_derived_identity(self, ref: str, spec: DerivedColumnConfig):
        src = self.ref_to_col(spec.src)
        self._df = self._df.with_columns(pl.col(src).alias(spec.name))
        self._register_ref(ref, spec.name)

    def _gen_derived_map(self, ref: str, spec: DerivedColumnConfig):
        src = self.ref_to_col(spec.src)
        dtypes = {type(v) for v in spec.args.values()}
        if len(dtypes) > 1:
            self.logger.error(
                "Derived column %s map to inconsistent dtypes", spec.name, dtypes
            )
            raise ConfigurationError("Invalid configuration")
        dtype = dtypes.pop()
        if dtype is str:
            src_expr = pl.col(src).cast(pl.String)
        else:
            src_expr = pl.col(src)
        self._df = self._df.with_columns(src_expr.replace(spec.args).alias(spec.name))
        self._register_ref(ref, spec.name)

    def _gen_derived_map_fn(
        self, ref: str, spec: DerivedColumnConfig, fn: Callable[str, [Any]]
    ):
        src = self.ref_to_col(spec.src)
        self._df = self._df.with_columns(
            pl.col(src).map_elements(fn, return_dtype=str).alias(spec.name)
        )
        self._register_ref(ref, spec.name)

    def _gen_derived_concat(self, ref: str, spec: DerivedColumnConfig):
        src = [self.ref_to_col(src_ref) for src_ref in spec.src]
        sep = spec.args.get("separator", ",")
        self._df = self._df.with_columns(
            pl.concat_str(src, separator=sep).alias(spec.name)
        )
        self._register_ref(ref, spec.name)

    def _gen_derived_columns(self):
        """
        Generate derived columns according to the configuration.

        Note that new columns will be made available in the column_refs_map.
        """
        for ref, spec in self._config.derived_columns.items():
            if spec.name in self._df.columns:
                self.logger.error(
                    "Invalid name %s for target column, column exists", spec.name
                )
                raise ConfigurationError("Invalid configuration")
            match spec.transform:
                case ColumnTransform.Identity:
                    self._gen_derived_identity(ref, spec)
                case ColumnTransform.Map:
                    self._gen_derived_map(ref, spec)
                case ColumnTransform.MapToBytes:
                    self._gen_derived_map_fn(ref, spec, int2bytes)
                case ColumnTransform.StrCat:
                    self._gen_derived_concat(ref, spec)
                case _:
                    self.logger.error("Invalid column transform %s", spec.transform)
                    raise RuntimeError("Unsupported column transformation")

    def _sort(self):
        """
        Sort the dataframe according to the configuration.
        """
        if not self._config.sort_by:
            self.logger.warning(
                "Sorting is not configured, use grid tiling order, this may not be stable."
            )
            sort_by = [self._config.tile_row, self._config.tile_col, self._config.hue]
            sort_by = [v for v in sort_by if v is not None]
        else:
            sort_by = self._config.sort_by
        sort_by = [self.ref_to_col(ref) for ref in sort_by]

        # If there is no custom ordering of categorical column values
        # we can just sort the frame and rely on the internal dtype ordering support.
        # otherwise, create a custom weight column.
        if not self._config.sort_order:
            self._df = self._df.sort(
                by=sort_by, descending=self._config.sort_descending
            )
            return

        sort_order = {
            self.ref_to_col(ref): order
            for ref, order in self._config.sort_order.items()
        }
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
                    self.logger.error(
                        "Invalid sort weights mapping for %s, requires %s",
                        col,
                        data_values,
                    )
                    raise ConfigurationError("Invalid sort_order configuration")
            n_weights = max(len(data_order), n_weights)
            weights = weight_base + np.arange(n_weights)
            self.logger.debug(
                "Assign weights to column %s: %s", col, list(zip(data_order, weights))
            )
            for value, weight in zip(data_order, weights):
                df = df.with_columns(
                    pl.col("_sort_weight")
                    + pl.when(pl.col(col) == value).then(weight).otherwise(0)
                )
            weight_base += n_weights
        self._df = df.sort(by="_sort_weight", descending=self._config.sort_descending)

    def _filter(self):
        """
        Filter the dataframe according to the configuration.
        """
        if self._config.data_filter is None:
            return

        if rules := self._config.data_filter.keep:
            conditions = []
            for rule in rules:
                cond = [
                    pl.col(self.ref_to_col(k)) == v for k, v in rule.matches.items()
                ]
                conditions.append(reduce(operator.and_, cond))
            cond = reduce(operator.or_, conditions)
            self._df = self._df.filter(cond)

        if rules := self._config.data_filter.drop:
            conditions = []
            for rule in rules:
                cond = [
                    pl.col(self.ref_to_col(k)) == v for k, v in rule.matches.items()
                ]
                conditions.append(reduce(operator.and_, cond))
            cond = reduce(operator.or_, conditions)
            self._df = self._df.filter(~cond)

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
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        if self._config.hue is None:
            # Generate synthetic hue column with an uniform value
            self._df = self._df.with_columns(pl.lit(0).alias("_default_hue"))
            self._color_palette = {0: color_cycle[0]}
        else:
            # Keep the order so that the user has control on the mapping between
            # data points and hue colors even when not using explicit color mapping.
            hue_keys = self._df[self.tile_hue].unique(maintain_order=True)
            ncolors = len(hue_keys)
            if self._config.hue_colors is not None:
                if len(self._config.hue_colors) != ncolors:
                    self.logger.error(
                        "Hue level %s: invalid number of colors in configuration: expected %d, found %d",
                        self._config.hue,
                        ncolors,
                        len(self._config.hue_colors),
                    )
                    raise ConfigurationError("Configuration error")
                self._color_palette = {h: self._config.hue_colors[h] for h in hue_keys}
            else:
                self._color_palette = dict(zip(hue_keys, color_cycle))

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
                text = tile.ax.annotate(
                    f"{row_param} = {tile.row}",
                    xy=(1.02, 0.5),
                    xycoords="axes fraction",
                    rotation=270,
                    ha="left",
                    va="center",
                )
                self._margin_titles.append(text)

        # Handle X and Y labels override
        def _sanitize_label_override(label_ref):
            label_col = self.ref_to_col(label_ref)
            if chunk[label_col].n_unique() > 1:
                self.logger.warning(
                    "Can not determine Y axis label for tile (%s, %s) from %s; non-unique label in chunk",
                    tile.row,
                    tile.col,
                    label_ref,
                )
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

    def _make_tile(self, ax: MplAxes, ri: int, ci: int, row_value: str, col_value: str):
        return PlotTile(
            grid=self,
            ax=ax,
            hue=self.tile_hue,
            palette=self._color_palette,
            coords=(ri, ci),
            loc=(row_value, col_value),
        )

    def _finalize_margins(self):
        # Helper internal function to set margin along an axis
        def _set_margins(
            tile, _, margins: tuple[float, float] = None, dimension: str = None
        ):
            # Note we conver from data to display coordinates and from display to axes.
            # Axes coordinates are in the space [0, 1] x [0, 1].
            axes_space_tf = tile.ax.transData + tile.ax.transAxes.inverted()
            data_space_tf = tile.ax.transAxes + tile.ax.transData.inverted()

            match dimension:
                case "x":
                    get_lim_fn = tile.ax.get_xlim
                    set_lim_fn = tile.ax.set_xlim

                    def to_axes_space(x):
                        return axes_space_tf.transform([x, 0])[0]

                    def to_data_space(x):
                        return data_space_tf.transform([x, 0])[0]

                    # We manage margins manually, so reset the margin here
                    # this should be redundant
                    tile.ax.set_xmargin(0)
                case "y":
                    get_lim_fn = tile.ax.get_ylim
                    set_lim_fn = tile.ax.set_ylim

                    def to_axes_space(y):
                        return axes_space_tf.transform([0, y])[1]

                    def to_data_space(y):
                        return data_space_tf.transform([0, y])[1]

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
            # hue_keys = self._df[self.tile_hue].unique(maintain_order=True)
            # labels = self._df[self.tile_hue].unique(maintain_order=True)
            # patches = [Patch(color=self._color_palette[hue_key]) for hue_key in hue_keys]

            legend_handles = {}

            def _merge_legend_handles(tile, _chunk):
                handles, labels = tile.ax.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    if label not in legend_handles:
                        legend_handles[label] = handle

            self.map(_merge_legend_handles)

            # DEPRECATED Make space between the title and the subplot axes
            # reserved_y_fraction = 1 - self._config.legend_vspace
            # self._figure.subplots_adjust(top=reserved_y_fraction, bottom=self._config.legend_vspace)
            # legend_anchor = (0., reserved_y_fraction, 1., self._config.legend_vspace)
            self._figure.legend(
                legend_handles.values(),
                legend_handles.keys(),
                ncols=self._config.legend_columns,
                loc="outside upper center",
                **self._legend_kwargs,
            )
        else:
            raise RuntimeError(
                f"Invalid legend_position='{self._config.legend_position}'"
            )

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

    def map(
        self, tile_plotter: Callable[[PlotTile, pl.DataFrame], None], *args, **kwargs
    ):
        for i, (ax_stride, (row_param, row_chunk)) in enumerate(
            zip(self._grid, self._grid_rows(self._df))
        ):
            for j, (ax, (col_param, chunk)) in enumerate(
                zip(ax_stride, self._grid_cols(row_chunk))
            ):
                tile = self._make_tile(ax, i, j, row_param, col_param)
                self.logger.debug(
                    "PlotGrid: tile callback (row=%s, col=%s) %s",
                    row_param,
                    col_param,
                    chunk,
                )
                tile_plotter(tile, chunk, *args, **kwargs)

    def get_grid_df(self) -> pl.DataFrame:
        return self._df

    def add_legend(self, **kwargs):
        """
        Add the legend on top of the axes.

        The legend is generated lazily on contextmanager exit.
        """
        self._legend_kwargs = kwargs


def grid_debug(tile: PlotTile, chunk: pl.DataFrame, x: str, y: str, logger: Logger):
    """
    Helper for debugging plot generation.

    This dumps the output sorted dataframe for each tile.
    """
    logger.debug("Dump tile (%s, %s)\n%s", tile.row, tile.col, chunk)
