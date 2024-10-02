import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def grid_pointplot(tile, chunk, x, y, err=None, shift=True, padding=0.1):
    """
    Create a simple point plot with error bars.

    Currently there is no orientation support, vertical plots are supported.
    """
    # Attempt column renaming
    try:
        x = tile.d[x]
        y = tile.d[y]
        hue = tile.d[tile.hue]
    except AttributeError:
        hue = tile.hue

    if chunk[x].dtype.is_numeric():
        view = chunk.with_columns(pl.col(x).alias("_xcoord_"))
    else:
        # Categorical X axis point plot
        coord_expr = pl.col(x).cum_count()
        if hue:
            coord_expr = coord_expr.over(hue)
        view = chunk.with_columns(coord_expr.alias("_xcoord_"))
        tile.ax.set_xticks(ticks=view["_xcoord_"].unique(maintain_order=True),
                           labels=view[x].unique(maintain_order=True))

    if hue:
        n_hue = len(tile.palette)
        if shift and not chunk[x].dtype.is_numeric():
            # Compute shift offsets for each hue, this is only sensible for categorical plots.
            width = 1 / n_hue
            center_shift = -0.5 - width / 2
            hue_index = view.with_columns(pl.col(hue).cum_count().over("_xcoord_").alias("_offset_"))
            # We shift the offsets so that each block is centered around offset = 0
            # and the tick falls at the center of the per-hue space width.
            # For instance, given 3 hues, we will have the following
            # offsets = [-1/3, 0, 1/3]
            # width = 1/3
            view = hue_index.with_columns(pl.col("_offset_") * width + center_shift)
            error_cap_width = 5
        else:
            view = view.with_columns(pl.lit(0).alias("_offset_"))
            error_cap_width = 5

        for color, (hue_label, hue_group) in zip(tile.palette, view.group_by(hue, maintain_order=True)):
            x = hue_group["_xcoord_"] + hue_group["_offset_"]
            if err:
                lower, upper = err
                err_low = (hue_group[y] - hue_group[lower]).abs()
                err_high = (hue_group[upper] - hue_group[y]).abs()
                tile.ax.errorbar(x,
                                 hue_group[y],
                                 yerr=(err_low, err_high),
                                 fmt="none",
                                 capsize=error_cap_width,
                                 ecolor="black")
            tile.ax.plot(x, hue_group[y], marker="o", linestyle="none", color=color, label=hue_label)
    else:
        tile.ax.plot(view["_xcoord_"], view[y], marker="o", linestyle="none")

    tile.ax.set_ylabel(y)
