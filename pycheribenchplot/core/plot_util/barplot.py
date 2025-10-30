from dataclasses import dataclass

import matplotlib.pyplot as plt
import polars as pl

from ..config import Config, config_field
from .coords import CoordGenConfig, CoordGenerator
from .plot_grid import PlotConfigBase


@dataclass
class BarPlotConfig(PlotConfigBase):
    """
    Display grid configuration extension specific to bar plots.
    """
    tile_xaxis: str = config_field(Config.REQUIRED, desc="Parameter to use for the X axis of each tile")
    stack_by: str | None = config_field(None, desc="Stack bars by the given parameter")
    shift_by: str | None = config_field(None, desc="Shift bars by the given parameter")

    def uses_param(self, name: str) -> bool:
        return (super().uses_param(name) or self.tile_xaxis == name or self.stack_by == name or self.shift_by == name)


def grid_barplot(tile: "PlotTile",
                 chunk: pl.DataFrame,
                 x: str,
                 y: str,
                 err: tuple[str, str] | None = None,
                 orient: str = "x",
                 stack: bool = False,
                 coordgen_kwargs: dict | None = None,
                 config: BarPlotConfig | None = None):
    """
    Produce a grouped bar plot on the given plot grid tile.
    # XXX add stacked + shifted version
    # XXX add twin-axis versions

    :param tile: The plot grid tile
    :param chunk: Dataframe containing the data to plot
    :param x: Canonical name of the dataframe column to use for the X axis values
    :param y: Canonical name of the dataframe column to use for the Y axis values
    :param err: 2-tuple of canonical names for columns containing the lower and upper bounds of the
    confidence interval, or None to disable errorbars.
    :param config: Common configuration for grouping and stacking bars.
    :param orient: Orientation of the plot, the `orient` axis is used as the categorical axis and
    the other axis must be numeric.
    """
    if coordgen_kwargs is None:
        coordgen_kwargs = {}

    # Attempt column renaming
    if hasattr(tile, "d"):
        x = tile.d[x]
        y = tile.d[y]
        hue = tile.d[tile.hue]
    else:
        hue = tile.hue
    palette = tile.palette

    if orient != "x" and orient != "y":
        raise ValueError("Invalid `orient` value, must be 'x' or 'y'")
    if orient == "x" and not chunk[y].dtype.is_numeric():
        raise TypeError("Y axis values must be numeric when orient='x'")
    if orient == "y" and not chunk[x].dtype.is_numeric():
        raise TypeError("X axis values must be numeric when orient='y'")
    orthogonal_orient = "x" if orient == "y" else "y"

    if orient == "x":
        catcol, metric = x, y
        bar_plot = tile.ax.bar
        set_ticks = tile.ax.set_xticks
        set_catlabel = tile.ax.set_xlabel
        set_mlabel = tile.ax.set_ylabel
    else:
        catcol, metric = y, x
        bar_plot = tile.ax.barh
        set_ticks = tile.ax.set_yticks
        set_catlabel = tile.ax.set_ylabel
        set_mlabel = tile.ax.set_xlabel
    if hue is None:
        # Create a fake hue colum set to null
        chunk = chunk.with_columns(pl.lit("__no_hue__").alias("__gen_hue"))
        hue = "__gen_hue"
        tile.palette = plt.rcParams["axes.prop_cycle"].by_key()["color"][:1]

    # XXX provide a way to propagate this to plot configuration
    cgen = CoordGenerator(tile.ax, orient=orient)
    # XXX handle stacked + shifted
    if stack:
        cgen_config = CoordGenConfig(stack_by=hue, **coordgen_kwargs)
    else:
        cgen_config = CoordGenConfig(shift_by=hue, **coordgen_kwargs)

    view = cgen.compute_coordinates(chunk, independent_var=catcol, dependent_vars=[metric], config=cgen_config)

    # Assign categorical axis ticks and labels
    ticks = view.select("__gen_coord", catcol).unique(maintain_order=True)
    set_ticks(ticks=ticks["__gen_coord"], labels=ticks[catcol])
    set_catlabel(catcol)
    set_mlabel(metric)
    # Draw the plot
    for color, (hue_label, hue_group) in zip(tile.palette, view.group_by(hue, maintain_order=True)):
        error_kwargs = {}
        if err:
            lower, upper = err
            err_low = (hue_group[metric] - hue_group[lower]).abs()
            err_high = (hue_group[upper] - hue_group[metric]).abs()
            error_kwargs.update({f"{orthogonal_orient}err": (err_low, err_high), "capsize": 4})
        coord = hue_group["__gen_coord"] + hue_group[f"__gen_offset"]
        bar_plot(coord,
                 hue_group[metric],
                 hue_group["__gen_width"],
                 hue_group["__gen_stack"],
                 color=color,
                 **error_kwargs)
