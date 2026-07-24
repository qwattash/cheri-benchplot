from dataclasses import dataclass
from typing import Any

import polars as pl
from marshmallow import ValidationError, validates

from ..config import config_field
from .coords import CoordGenConfig, CoordGenerator
from .plot_grid import check_colref_pattern, OptColRef, PlotConfigBase, PlotTile


@dataclass
class BarPlotConfig(PlotConfigBase):
    """
    Display grid configuration extension specific to bar plots.
    """

    stack_by: OptColRef = config_field(
        None, desc="Stack bars along the given column ref."
    )
    shift_by: OptColRef = config_field(
        None, desc="Shift bars along the given column ref."
    )
    orient: str = config_field("x", desc="Plot orientation (x or y).")
    coordgen: dict[str, Any] = config_field(
        dict, desc="Coordinate generator extra arguments. (internal)"
    )
    show_bar_label: bool = config_field(
        False, desc="Show the actual value of the bar on top of each bar."
    )
    bar_label_orient: str = config_field(
        "h", desc="Orientation of the bar labels (h or v)."
    )
    bar_label: OptColRef = config_field(
        None,
        desc="Optional ColRef to select an alternative column for the label (permits arbitrary formatting).",
    )
    bar_label_size: float | None = config_field(
        None, desc="Label text size (or None for default font size)."
    )

    @validates("bar_label_orient")
    def check_bar_label_orientation(self, data, **kwargs):
        if data not in ["h", "v"]:
            raise ValidationError("Bar label orientation must be 'h' or 'v'")

    @validates("orient")
    def check_orientation(self, data, **kwargs):
        if data != "x" and data != "y":
            raise ValidationError("Orientation must be 'x' or 'y'.")

    def uses_param(self, name: str) -> bool:
        ref = f"<{name}>"
        return (
            super().uses_param(name)
            or self.tile_xaxis == ref
            or self.stack_by == ref
            or self.shift_by == ref
            or self.bar_label == ref
        )


def _draw_bar_labels(
    tile: PlotTile,
    df: pl.DataFrame,
    coord: pl.Series,
    config: BarPlotConfig,
    d_var: str,
    err: tuple[str, str] | None,
):
    """
    Helper function to draw value labels on top of (or next to) each bar.

    :param tile: The tile descriptor
    :param df: The hue_group dataframe
    :param coord: Series of coordinates on the independent-variable orientation axis
    :param config: Bar plot configuration
    :param d_var: Dependent variable column name
    :param err: Tuple with the error bar column names, if configured
    """
    label_col = d_var
    if config.bar_label:
        label_col = tile.ref_to_col(config.bar_label)

    # Format the label values according to the label configuration
    if df.schema[label_col].is_numeric():
        bar_label = df.select(
            pl.col(label_col)
            .cast(pl.Float32)
            .fill_nan(None)
            .map_elements(lambda v: f"{v:.3g}", skip_nulls=True)
            .fill_null("")
        ).to_series()
    else:
        bar_label = df[label_col]

    sign = df[d_var].sign()
    if err:
        # the 'top' Y coordinate depends on the sign of the data, when negative, we need to
        # pick the lower error bar position.
        text_coord = df.select(
            pl.when(pl.col(d_var) >= 0).then(pl.col(err[1])).otherwise(pl.col(err[0]))
            + pl.col("__gen_stack")
        ).to_series()
    else:
        text_coord = df.select(pl.col(d_var) + pl.col("__gen_stack")).to_series()

    # Produce offset, alignment and location vectors
    text_offset = 3 * sign  # 3 points offset
    align_lr_vector = df.select(
        pl.when(pl.col(d_var) >= 0).then(pl.lit("left")).otherwise(pl.lit("right"))
    ).to_series()
    align_tb_vector = df.select(
        pl.when(pl.col(d_var) >= 0).then(pl.lit("bottom")).otherwise(pl.lit("top"))
    ).to_series()
    if config.orient == "x":
        text_x = coord
        text_y = text_coord
        if config.bar_label_orient == "h":
            text_ha = "center"
            text_align = align_tb_vector  # va
        elif config.bar_label_orient == "v":
            text_align = align_lr_vector  # ha
            text_va = "center"
    else:
        text_x = text_coord
        text_y = coord
        if config.bar_label_orient == "h":
            text_align = align_lr_vector  # ha
            text_va = "center"
        elif config.bar_label_orient == "v":
            text_ha = "center"
            text_align = align_tb_vector  # va

    annotate_kwargs = {}
    if config.bar_label_size is not None:
        annotate_kwargs["fontsize"] = config.bar_label_size

    if config.bar_label_orient == "h":
        annotate_kwargs["rotation"] = 0
        annotate_kwargs["rotation_mode"] = "anchor"
    else:
        annotate_kwargs["rotation"] = 90
        annotate_kwargs["rotation_mode"] = "anchor"

    for x, y, label, off, align in zip(
        text_x, text_y, bar_label, text_offset, text_align
    ):
        if config.orient == "x":
            offset = (0, off)
            if config.bar_label_orient == "h":
                ha, va = text_ha, align
            else:
                ha, va = align, text_va
        else:
            offset = (off, 0)
            if config.bar_label_orient == "h":
                ha, va = align, text_va
            else:
                ha, va = text_ha, align

        tile.ax.annotate(
            label,
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            **annotate_kwargs,
        )


def grid_barplot(
    tile: PlotTile,
    chunk: pl.DataFrame,
    config: BarPlotConfig,
    x: str,
    y: str,
    err: tuple[str, str] | None = None,
):
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
    # Resolve ColRef for the X axis
    x = tile.ref_to_col(x)
    # The Y axis is never allowed a ColRef
    assert not check_colref_pattern(y)

    # The error columns may contain NaN, but not NoneType, verify this
    if err:
        assert chunk[err[0]].null_count() == 0, (
            f"Unexpected NoneType in error column {err[0]}"
        )
        assert chunk[err[1]].null_count() == 0, (
            f"Unexpected NoneType in error column {err[1]}"
        )

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
    view = cgen.compute_coordinates(
        chunk, independent_var=i_var, dependent_vars=[d_var], config=cgen_config
    )

    # Assign categorical axis ticks and labels
    ticks = view.select("__gen_coord", i_var).unique(maintain_order=True)
    set_ticks(ticks=ticks["__gen_coord"], labels=ticks[i_var])
    # Draw the plot
    for (hue_label,), hue_group in view.group_by(tile.hue, maintain_order=True):
        color = tile.palette[hue_label]
        error_kwargs = {}
        if err:
            lower, upper = err
            err_low = (hue_group[d_var] - hue_group[lower]).abs()
            err_high = (hue_group[upper] - hue_group[d_var]).abs()
            error_kwargs.update(
                {f"{orthogonal_orient}err": (err_low, err_high), "capsize": 4}
            )
        coord = hue_group["__gen_coord"] + hue_group["__gen_offset"]
        bar_plot(
            coord,
            hue_group[d_var],
            hue_group["__gen_width"],
            hue_group["__gen_stack"],
            color=color,
            label=hue_label,
            **error_kwargs,
        )

        if config.show_bar_label:
            _draw_bar_labels(tile, hue_group, coord, config, d_var, err)
