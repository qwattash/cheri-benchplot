from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ..config import Config, config_field
from .plot_grid import PlotConfigBase, PlotTile


class Scale(Enum):
    Linear = "linear"
    Log2 = "log2"
    Log10 = "log10"


@dataclass
class LinePlotConfig(PlotConfigBase):
    """
    Display grid configuration extension specific to line plots.
    """
    tile_xaxis: str = config_field(Config.REQUIRED, desc="Parameter to use for the X axis of each tile.")
    tile_xscale: Scale = config_field(Scale.Linear, desc="Scale for the x axis.")
    line_width: float | None = config_field(None, desc="Width of the lines.")
    marker_fill: bool = config_field(True, desc="When false, only draw the marker outline.")
    marker_size: float | None = config_field(None, desc="Size of the markers.")
    hue_marker: dict[str, str] = config_field(dict, desc="Set the marker style for each hue.")
    hue_linestyle: dict[str, str] = config_field(dict, desc="Set the line style for each hue.")

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
                  x: str,
                  y: str,
                  err: tuple[str, str] | None = None,
                  config: LinePlotConfig | None = None):
    """
    Create a line plot with error bars.
    """
    try:
        # Handle column renaming if we have a DisplayTile
        x = tile.d[x]
        y = tile.d[y]
        hue = tile.d[tile.hue]
    except AttributeError:
        hue = tile.hue

    # XXX this is a common operation and should probably be part of the PlotGrid
    if hue is None:
        # Create a fake hue column with a non-null value
        chunk = chunk.with_columns(pl.lit("__no_hue__").alias("__gen_hue"))
        hue = "__gen_hue"
        tile.palette = plt.rcParams["axes.prop_cycle"].by_key()["color"][:1]

    if chunk[x].dtype.is_numeric():
        view = chunk.with_columns(pl.col(x).alias("__gen_coord"))
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

    for (hue_label, ), hue_group in view.group_by(hue, maintain_order=True):
        color = tile.palette[hue_label]
        plot_kwargs = config.lineplot_kwargs()
        # XXX this is broken by renaming, there should be a better way to do this, perhaps
        # via the base plot config
        if marker := config.hue_marker.get(hue_label):
            plot_kwargs["marker"] = marker
        if linestyle := config.hue_linestyle.get(hue_label):
            plot_kwargs["linestyle"] = linestyle
        if err:
            lower, upper = err
            err_low = (hue_group[y] - hue_group[lower]).abs()
            err_high = (hue_group[upper] - hue_group[y]).abs()
            plot_kwargs.update({"yerr": (err_low, err_high), "capsize": 4})
            tile.ax.errorbar(hue_group["__gen_coord"], hue_group[y], color=color, label=hue_label, **plot_kwargs)
        else:
            tile.ax.plot(hue_group["__gen_coord"], hue_group[y], color=color, label=hue_label, **plot_kwargs)
        xval = hue_group["__gen_coord"]
        tile.ax.set_xticks(xval)
        tile.ax.set_xlim(xval.min(), xval.max())

    tile.ax.set_ylabel(y)
    tile.ax.set_xlabel(x)
