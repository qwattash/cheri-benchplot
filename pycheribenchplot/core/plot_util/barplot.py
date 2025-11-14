from dataclasses import dataclass

import matplotlib.pyplot as plt
import polars as pl
from marshmallow import ValidationError, validates

from ..config import Config, config_field
from .coords import CoordGenConfig, CoordGenerator
from .plot_grid import ColRef, PlotConfigBase


@dataclass
class BarPlotConfig(PlotConfigBase):
    """
    Display grid configuration extension specific to bar plots.
    """
    tile_xaxis: ColRef = config_field(Config.REQUIRED, desc="Column ref to use for the tile X axis.")
    stack_by: ColRef | None = config_field(None, desc="Stack bars along the given column ref.")
    shift_by: ColRef | None = config_field(None, desc="Shift bars along the given column ref.")
    orient: str = config_field("x", desc="Plot orientation (x or y).")

    @validates("orient")
    def check_orientation(self, data, **kwargs):
        if data != "x" and data != "y":
            raise ValidationError("Orientation must be 'x' or 'y'.")

    def uses_param(self, name: str) -> bool:
        ref = f"<{name}>"
        return (super().uses_param(name) or self.tile_xaxis == ref or self.stack_by == ref or self.shift_by == ref)


def grid_barplot(tile: "PlotTile",
                 chunk: pl.DataFrame,
                 config: BarPlotConfig,
                 x: str,
                 y: str,
                 err: tuple[str, str] | None = None):
    """
    Produce a grouped bar plot on the given plot grid tile.
    # XXX add stacked + shifted version
    # XXX add twin-axis versions

    :param tile: The plot grid tile.
    :param chunk: Dataframe containing the data to plot.
    :param x: ColRef of the dataframe column to use for the X axis values.
    :param y: Sanitized name of the dataframe column to use for the Y axis values.
    :param err: 2-tuple of canonical names for columns containing the lower and upper bounds of the
    confidence interval, or None to disable errorbars.
    :param config: Bar plot configuration object.
    """
    # Note: here all the column refs should have been resolved
    x = tile.ref_to_col(x)
    assert not y.startswith("<")
    assert tile.hue is not None
    assert not tile.hue.startswith("<")

    if config.orient == "x" and not chunk[y].dtype.is_numeric():
        raise TypeError("Y axis values must be numeric when orient='x'")
    if config.orient == "y" and not chunk[x].dtype.is_numeric():
        raise TypeError("X axis values must be numeric when orient='y'")
    orthogonal_orient = "x" if config.orient == "y" else "y"

    if config.orient == "x":
        i_var, d_var = x, y
        bar_plot = tile.ax.bar
        set_ticks = tile.ax.set_xticks
        tile.ax.set_xlabel(x)
        tile.ax.set_ylabel(y)
    else:
        i_var, d_var = y, x
        bar_plot = tile.ax.barh
        set_ticks = tile.ax.set_yticks
        tile.ax.set_xlabel(y)
        tile.ax.set_ylabel(x)

    # XXX provide a way to propagate this to plot configuration
    cgen = CoordGenerator(tile.ax, orient=config.orient)
    # XXX handle stacked + shifted
    coordgen_kwargs = {}
    if config.stack_by:
        coordgen_kwargs["stack_by"] = tile.hue
    elif config.shift_by:
        coordgen_kwargs["shift_by"] = tile.hue
    else:
        coordgen_kwargs["shift_by"] = tile.hue

    cgen_config = CoordGenConfig(**coordgen_kwargs)
    view = cgen.compute_coordinates(chunk, independent_var=i_var, dependent_vars=[d_var], config=cgen_config)

    # Assign categorical axis ticks and labels
    ticks = view.select("__gen_coord", i_var).unique(maintain_order=True)
    set_ticks(ticks=ticks["__gen_coord"], labels=ticks[i_var])
    # Draw the plot
    for (hue_label, ), hue_group in view.group_by(tile.hue, maintain_order=True):
        color = tile.palette[hue_label]
        error_kwargs = {}
        if err:
            lower, upper = err
            err_low = (hue_group[d_var] - hue_group[lower]).abs()
            err_high = (hue_group[upper] - hue_group[d_var]).abs()
            error_kwargs.update({f"{orthogonal_orient}err": (err_low, err_high), "capsize": 4})
        coord = hue_group["__gen_coord"] + hue_group[f"__gen_offset"]
        bar_plot(coord,
                 hue_group[d_var],
                 hue_group["__gen_width"],
                 hue_group["__gen_stack"],
                 color=color,
                 label=hue_label,
                 **error_kwargs)
