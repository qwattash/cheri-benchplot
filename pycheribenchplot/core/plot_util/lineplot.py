from dataclasses import dataclass
from enum import Enum

import polars as pl

from ..config import Config, config_field
from .plot_grid import ColRef, PlotConfigBase, PlotTile


class Scale(Enum):
    Linear = "linear"
    Log2 = "log2"
    Log10 = "log10"


@dataclass
class LinePlotConfig(PlotConfigBase):
    """
    Display grid configuration extension specific to line plots.
    """
    tile_xaxis: ColRef = config_field(Config.REQUIRED, desc="ColRef to use for the X axis of each tile.")
    tile_xscale: Scale = config_field(Scale.Linear, desc="Scale for the X axis.")
    tile_yscale: Scale = config_field(Scale.Linear, desc="Scale for the Y axis.")
    line_width: float | None = config_field(None, desc="Width of the lines.")
    marker_fill: bool = config_field(True, desc="When false, only draw the marker outline.")
    marker_size: float | None = config_field(None, desc="Size of the markers.")
    marker: ColRef | None = config_field(None, desc="ColRef containing the markers. Should align on the hue.")
    linestyle: ColRef | None = config_field(None, desc="ColRef containing the linestyles. Should align on the hue.")

    def uses_param(self, name: str) -> bool:
        return (super().uses_param(name) or self.tile_xaxis == name)

    def lineplot_kwargs(self) -> dict:
        style_kwargs = {}
        if w := self.line_width:
            style_kwargs.update({"linewidth": w, "markeredgewidth": w})
        if not self.marker_fill:
            style_kwargs.update({"markerfacecolor": "none"})
        if size := self.marker_size:
            style_kwargs.update({"markersize": size})
        return style_kwargs


def generate_x_coords(df: pl.DataFrame, x: str) -> pl.DataFrame:
    pass


def grid_lineplot(tile: PlotTile,
                  chunk: pl.DataFrame,
                  config: LinePlotConfig,
                  x: str,
                  y: str,
                  err: tuple[str, str] | None = None):
    """
    Create a line plot with error bars.
    """
    # Note: here all the column refs should have been resolved
    x = tile.ref_to_col(x)
    assert not y.startswith("<")
    assert tile.hue is not None
    assert not tile.hue.startswith("<")

    if chunk[x].dtype.is_numeric():
        view = chunk.with_columns(pl.col(x).alias("_x"))
    else:
        # Use coord allocator to produce coordinates or this is
        # just a very simple case in which we don't have to stack/shift?
        raise NotImplementedError("TODO")

    match config.tile_xscale:
        case Scale.Linear:
            pass
        case Scale.Log2:
            tile.ax.set_xscale("log", base=2)
        case Scale.Log10:
            tile.ax.set_xscale("log", base=10)

    match config.tile_yscale:
        case Scale.Linear:
            pass
        case Scale.Log2:
            tile.ax.set_yscale("log", base=2)
        case Scale.Log10:
            tile.ax.set_yscale("log", base=10)

    # Add knob to force xticks vs custom xticks vs automatic?
    x_vals = view["_x"].unique(maintain_order=True)
    tile.ax.set_xticks(x_vals)
    tile.ax.set_xlim(x_vals.min(), x_vals.max())
    tile.ax.set_xlabel(x)
    tile.ax.set_ylabel(y)

    for (hue_label, ), hue_group in view.group_by(tile.hue, maintain_order=True):
        color = tile.palette[hue_label]
        plot_kwargs = config.lineplot_kwargs()
        if marker_ref := config.marker:
            plot_kwargs["marker"] = hue_group[tile.ref_to_col(marker_ref)].first()
        if line_ref := config.linestyle:
            plot_kwargs["linestyle"] = hue_group[tile.ref_to_col(line_ref)].first()
        if err:
            lower, upper = err
            err_low = (hue_group[y] - hue_group[lower]).abs()
            err_high = (hue_group[upper] - hue_group[y]).abs()
            plot_kwargs.update({"yerr": (err_low, err_high), "capsize": 4})
            tile.ax.errorbar(hue_group["_x"], hue_group[y], color=color, label=hue_label, **plot_kwargs)
        else:
            tile.ax.plot(hue_group["_x"], hue_group[y], color=color, label=hue_label, **plot_kwargs)
