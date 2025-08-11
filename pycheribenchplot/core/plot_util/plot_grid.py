import enum
import inspect
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from functools import reduce
from typing import Any, Callable, Dict, Iterable, Optional, Self
from warnings import deprecated

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from marshmallow import ValidationError, validates_schema
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from ..artefact import Target
from ..config import Config, config_field
from ..util import bytes2int


def default(value, default):
    if value is None:
        return default
    return value


@dataclass
class PlotGridConfig(Config):
    title: Optional[str] = config_field(None, desc="Override figure title.")
    tile_row: Optional[str] = config_field(None, desc="Override parameter for grid rows.")
    tile_col: Optional[str] = config_field(None, desc="Override parameter for grid cols.")
    tile_row_as_ylabel: bool | None = config_field(
        None, desc="Use value of the tile_row parameter as the Y axis label for the row.")
    tile_col_as_xlabel: bool | None = config_field(
        None, desc="Use value of the tile_col parameter as the X axis label for the colum.")
    tile_sharex: Optional[str | bool] = config_field(None, desc="Override X axis sharing.")
    tile_sharey: Optional[str | bool] = config_field(None, desc="Override Y axis sharing.")
    plot_params: Dict[str, Any] = config_field(
        dict, desc="Plot appearance configuration for tweaking, see matplotlib rc_context documentation")
    hue: Optional[str] = config_field(
        None,
        desc="Override parameter for hue. When no hue is given, the color is controlled by the "
        "plot_param axes.prop_cycle (See matplotlib rcParams).")
    hue_colors: Optional[dict[str, str]] = config_field(
        None, desc="Optional map that forces a specific color for each value of the 'hue' configuration.")
    tile_aspect: float = config_field(1.0, desc="Aspect ratio of each tile.")
    legend_vspace: float = config_field(0.2, desc="Fraction of the vertical space reserved to legend and suptitle.")
    legend_hide: bool = config_field(False, desc="Disable the legend.")
    legend_columns: int = config_field(
        4, desc="Number of columns in the legend, this affects the necessary vspace and tile_aspect.")

    def set_default(self, **kwargs):
        defaults = {k: v for k, v in kwargs.items() if hasattr(self, k) and getattr(self, k) is None}
        config = replace(self, **defaults)
        return config

    @deprecated("Use PlotGridConfig.set_default() instead")
    def setdefault(self, **kwargs):
        return self.set_default(**kwargs)

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

    def weight_transform(self):
        if self == WeightMode.SortAscendingAsStr or self == WeightMode.SortDescendingAsStr:
            return lambda series: series.cast(pl.String)
        if self == WeightMode.SortAscendingAsInt or self == WeightMode.SortDescendingAsInt:
            return lambda series: series.cast(pl.Int64)
        if self == WeightMode.SortAscendingAsBytes or self == WeightMode.SortDescendingAsBytes:
            return lambda series: series.map_elements(bytes2int, return_dtype=pl.Int64)
        return None


class WeightOrder(enum.Enum):
    Ascending = "ascending"
    Descending = "descending"


@dataclass
class ParamWeight(Config):
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
                raise ValidationError("ParamWeight.weights must be set when TVRSParamWeight.mode is 'custom'")
        else:
            if data["base"] is None:
                raise ValidationError("ParamWeight.base must be set")
            if data["step"] is None:
                raise ValidationError("ParamWeight.step must be set")


@dataclass
class DisplayGridConfig(PlotGridConfig):
    """
    Dataframe context renaming and sorting configuration

    This should be inherited by all analysis task configurations that use renaming.
    Note that the display renaming/mapping is not allowed to change the behaviour of
    group_by over parameterization axes.
    """
    param_sort_weight: Optional[Dict[str, ParamWeight]] = config_field(
        None,
        desc="Weight for determining the order of labels based on the parameters. "
        "The dictionary key must be a parameter name, the value is a weight descriptor.")

    param_sort_order: WeightOrder = config_field(WeightOrder.Ascending,
                                                 desc="Sort order by ascending or descending weight.")

    param_names: Optional[Dict[str, str]] = config_field(
        None, desc="Relabel parameter keys specified in PipelineBenchmarkConfig.parameterize")

    param_values: Optional[Dict[str, Dict[str, Any]]] = config_field(
        None,
        desc="Relabel parameter values. The key is a parameter name, the value "
        "is a mapping of the form { <parameter value> => <value alias> }.")

    param_filter: Optional[Dict[str, Any]] = config_field(
        None,
        desc="Filter the data for the given set of parameters. Specify constraints as key=value pairs, "
        "multiple constraints on the same key are not supported. Note that depending on the "
        "tiling setup, this may result in an unaligned dataframe and cause errors.")

    def set_display_defaults(self,
                             param_names: dict | None = None,
                             param_values: dict[str, dict[str, any]] | None = None) -> Self:
        """
        Set default renames
        """
        config = replace(self)
        if param_names:
            config.param_names = dict(param_names, **default(config.param_names, {}))
        if param_values:
            config.param_values = dict(param_values, **default(config.param_values, {}))
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
    def __init__(self, ax: Axes, hue: str | None, palette: list | None, coords: tuple[int, int], loc: tuple[any, any]):
        self.ax = ax
        self.hue = hue
        # Note that the palette colors are in the sort order of the hue column
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


class DisplayTile(PlotTile):
    def __init__(self, ax, hue, palette, coords, loc, raw_map: ColumnNameMapping, display_map: ColumnNameMapping):
        super().__init__(ax, hue, palette, coords, loc)
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


class PlotGrid(AbstractContextManager):
    """
    Abstraction to generate grid plots based on polars dataframes.

    This aims to have some similarity to seaborn FacetGrid while being more
    efficient with polars dataframes.
    """
    def __init__(self, target: Target, data: pl.DataFrame, config: PlotGridConfig):
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
        self._color_palette = None
        self._rc_context = None
        # This is set when legend is enabled, kwargs may be an empty dict
        self._legend_kwargs = None

    def __enter__(self):
        # Determine tiling row and column parameters
        nrows, ncols = self._grid_shape()
        # Compute the figure size based on the configured rows and columns
        # and initialize the figure
        fig_size = (ncols * self._height * self._config.tile_aspect, (nrows + 1) * self._height)
        self._figure = plt.figure(figsize=fig_size)
        # Initialize the grid
        kwargs = {
            "squeeze": False,
            "sharex": default(self._config.tile_sharex, "col"),
            "sharey": default(self._config.tile_sharey, "row"),
        }

        self._grid = self._figure.subplots(nrows, ncols, **kwargs)

        if self._config.title:
            self._figure.suptitle(self._config.title)
        self._gen_color_palette()
        self._rc_context = plt.rc_context(rc=self._config.plot_params).__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.map(self._set_titles)
        if self._rc_context:
            self._rc_context.__exit__(exc_type, exc_value, traceback)
        if self._legend_kwargs is not None:
            self._generate_legend()
            # XXX provide a way to adjust these
            # self._figure.subplots_adjust(bottom=0.15)
        for path in self._target.paths():
            self._figure.savefig(path)
        plt.close(self._figure)

    def _grid_shape(self) -> tuple[int, int]:
        nrows, ncols = 1, 1
        if self._config.tile_row:
            nrows = self._df[self.tile_row_param].n_unique()
        if self._config.tile_col:
            ncols = self._df[self.tile_col_param].n_unique()
        return nrows, ncols

    def _grid_rows(self, df=None) -> Iterable[tuple[any, pl.DataFrame]]:
        df = default(df, self._df)
        if self._config.tile_row is None:
            yield (None, df)
        else:
            for (name, ), chunk in df.group_by(self.tile_row_param, maintain_order=True):
                yield (name, chunk)

    def _grid_cols(self, df=None) -> Iterable[tuple[any, pl.DataFrame]]:
        df = default(df, self._df)
        if self._config.tile_col is None:
            yield (None, df)
        else:
            for (name, ), chunk in df.group_by(self.tile_col_param, maintain_order=True):
                yield (name, chunk)

    def _gen_color_palette(self):
        if self._config.hue is None:
            self._color_palette = None
        else:
            hue_count = self._df.select(pl.n_unique(self.hue_param).alias("count"))
            ncolors = hue_count["count"][0]
            if self._config.hue_colors is not None:
                if len(self._config.hue_colors) != ncolors:
                    self.logger.error(
                        "Invalid number of colors in configuration: expected %d, found %d for hue level %s", ncolors,
                        len(self._config.hue_colors), self._config.hue)
                    raise ValueError("Configuration error")
                hues = self._df[self.hue_param].unique(maintain_order=True)
                self._color_palette = [self._config.hue_colors[h] for h in hues]
            else:
                self._color_palette = sns.color_palette(n_colors=ncolors)

    def _set_titles(self, tile, chunk):
        """
        Set row and column labels
        """
        nrows, ncols = self._grid_shape()
        if self._config.tile_col_as_xlabel:
            tile.ax.set_xlabel(tile.col, loc="center")
        else:
            if tile.row_index == 0 and self._config.tile_col:
                tile.ax.set_title(f"{self.tile_col_param} = {tile.col}")

        if self._config.tile_row_as_ylabel:
            tile.ax.set_ylabel(tile.row, loc="center")
        else:
            if self._config.tile_sharey and tile.col_index > 0:
                # Clear the Y axis label if this isn't the first column
                tile.ax.set_ylabel("").set_visible(False)
            if tile.col_index == ncols - 1 and self._config.tile_row:
                text = tile.ax.annotate(f"{self.tile_row_param} = {tile.row}",
                                        xy=(1.02, .5),
                                        xycoords="axes fraction",
                                        rotation=270,
                                        ha="left",
                                        va="center")
                self._margin_titles.append(text)

    def _make_tile(self, ax, ri, ci, row, col):
        return PlotTile(ax=ax, hue=self._config.hue, palette=self._color_palette, coords=(ri, ci), loc=(row, col))

    def _generate_legend(self):
        if not self._config.hue:
            return
        self._figure.tight_layout()
        if self._config.legend_hide:
            return
        labels = self._df[self.hue_param].unique(maintain_order=True)
        patches = [Patch(color=color) for color in self._color_palette]
        reserved_y_fraction = 1 - self._config.legend_vspace

        # Make space between the title and the subplot axes
        self._figure.subplots_adjust(top=reserved_y_fraction, bottom=self._config.legend_vspace)
        self._figure.legend(patches,
                            labels,
                            bbox_to_anchor=(0., reserved_y_fraction, 1., self._config.legend_vspace),
                            ncols=self._config.legend_columns,
                            loc="center",
                            **self._legend_kwargs)

    @property
    def tile_row_param(self):
        return self._config.tile_row

    @property
    def tile_col_param(self):
        return self._config.tile_col

    @property
    def hue_param(self):
        return self._config.hue

    @property
    def logger(self):
        return self._target.task.logger

    def map(self, tile_plotter: Callable[[PlotTile, pl.DataFrame], None], *args, **kwargs):
        for i, (ax_stride, (row_param, row_chunk)) in enumerate(zip(self._grid, self._grid_rows(self._df))):
            for j, (ax, (col_param, chunk)) in enumerate(zip(ax_stride, self._grid_cols(row_chunk))):
                tile = self._make_tile(ax, i, j, row_param, col_param)
                self.logger.debug("PlotGrid: tile callback (row=%s, col=%s) %s", row_param, col_param, chunk)
                tile_plotter(tile, chunk, *args, **kwargs)

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
        self._sort()
        self._gen_display_columns()
        super().__enter__()
        return self

    def _filter(self):
        """
        Filter the dataframe according to the configuration.
        """
        if filter_args := self._config.param_filter:
            self._df = self._df.filter(**filter_args)

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

    def _sort(self):
        """
        Sort the dataframe according to the configured param_sort_weight.

        Note that this always operates before mapping columns to the
        human-readable display view.
        """
        if not self._config.param_sort_weight:
            self.logger.info("Skipping sort(), not configured")
            return
        df = self._df.with_columns(pl.lit(0.0).alias("_param_weight"))
        for name, weight_spec in self._config.param_sort_weight.items():
            if name not in self._df.columns:
                self.logger.info("Skipping weight for parameter '%s', does not exist", name)
                continue

            if weight_spec.mode == WeightMode.Custom:
                mapping = dict(weight_spec.weights)
            else:
                descending = weight_spec.mode.is_descending()
                unique_values = df[name].unique()
                if transform := weight_spec.mode.weight_transform():
                    unique_values = transform(unique_values)
                sorted_values = sorted(unique_values, reverse=descending)
                mapping = {v: weight_spec.base + i * weight_spec.step for i, v in enumerate(sorted_values)}
            # Update the weight for each row
            self.logger.debug("Set weight for %s => %s", name, mapping)
            df = df.with_columns(
                pl.col("_param_weight") + pl.col(name).replace(mapping, default=0, return_dtype=pl.Decimal))

        match self._config.param_sort_order:
            case WeightOrder.Ascending:
                descending = False
            case WeightOrder.Descending:
                descending = True
            case _:
                assert False, "Invalid WeightOrder value"
        self._df = df.sort(by="_param_weight", descending=descending)

    def _make_tile(self, ax, ri, ci, row, col):
        tile = DisplayTile(ax=ax,
                           hue=self._config.hue,
                           palette=self._color_palette,
                           coords=(ri, ci),
                           loc=(row, col),
                           raw_map=ColumnNameMapping(self._raw_map),
                           display_map=ColumnNameMapping(self._display_map))
        return tile

    @property
    def tile_row_param(self):
        return self._display_map.get(self._config.tile_row, self._config.tile_row)

    @property
    def tile_col_param(self):
        return self._display_map.get(self._config.tile_col, self._config.tile_col)

    @property
    def hue_param(self):
        return self._display_map.get(self._config.hue, self._config.hue)
