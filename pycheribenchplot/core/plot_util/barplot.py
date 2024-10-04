import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from .coords import CoordGenConfig, CoordGenerator


def grid_barplot(tile, chunk, x, y, err: tuple[str, str] | None = None, orient="x"):
    """
    Produce a grouped bar plot on the given plot grid tile.
    # XXX add stacked version
    # XXX add stacked and shifted version
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
        chunk = chunk.with_columns(pl.lit(None).alias("__gen_hue"))
        hue = "__gen_hue"
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"][:1]

    cgen = CoordGenerator(tile.ax, orient=orient)
    cgen_config = CoordGenConfig(shift_by=hue)
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
        bar_plot(coord, hue_group[metric], hue_group["__gen_width"], color=color, **error_kwargs)


def extended_barplot(data, **kwargs):
    """
    Bar plot with a similar interface to the seaborn barplot function, with support for custom
    pre-computed error bars.
    """
    df = pl.from_pandas(data) if not isinstance(data, pl.DataFrame) else data
    errorbar = kwargs.pop("errorbar", None)
    errorbar_kwargs = None
    if type(errorbar) == tuple and errorbar[0] == "custom":
        # Make sure this is a copy
        errorbar_kwargs = dict(errorbar[1])

    sns.barplot(df, **kwargs)
    if not errorbar_kwargs:
        return
    ax = kwargs.get("ax", plt.gca())
    dodge = kwargs.get("dodge", False)
    hue = kwargs.get("hue", None)
    palette = kwargs.get("palette", [None])
    width = kwargs.get("width", 0.8)
    x = kwargs["x"]
    y = kwargs["y"]
    yerr = errorbar_kwargs.pop("yerr", None)

    assert not errorbar_kwargs.get("xerr"), "Unsupported horizontal orientation"
    assert not kwargs.get("native_scale"), "Unsupported"
    assert yerr is not None, "yerr errorbar keyword argument is required"

    # Map the categorical X axis into coordinates, this must match seaborn
    x_cat = df[x].unique(maintain_order=True)
    x_points = np.arange(0, len(x_cat))
    df = df.with_columns(pl.col(x).replace({xc: xp for xc, xp in zip(x_cat, x_points)}).cast(float).alias("_xcoord"))

    hues = df[hue].unique(maintain_order=True) if hue else []
    if dodge:
        assert hue is not None, "Hue is required for dodge"
        bar_width = width / len(hues)
        dodge_steps = np.linspace(-width / 2, width / 2, 2 * len(hues) + 1)[1::2]
        hue_chunks = df.group_by(hue)
        dodge_map = {hue_val: offset for hue_val, offset in zip(hues, dodge_steps)}
    else:
        hue_chunks = [(None, df)]
        dodge_map = dict()

    err_color = errorbar_kwargs.pop("color", 0.78)
    errorbar_kwargs.setdefault("fmt", "none")
    errorbar_kwargs.setdefault("capsize", 5.0)
    errorbar_kwargs.setdefault("elinewidth", 2.0)
    errorbar_kwargs.setdefault("capthick", 2.0)

    # XXX handle orientation
    # Need to use maps for color and offsets because polars groupby does
    # not preserver group order by default.
    for hue_val, chunk in hue_chunks:
        # Normalize, iterating the groupby returns tuples
        if type(hue_val) == tuple:
            hue_val = hue_val[0]

        if yerr and type(yerr) == str:
            chunk_yerr = chunk[yerr] if yerr else None
        elif yerr:
            ymin, ymax = yerr
            err_lo = chunk[y] - chunk[ymin]
            err_hi = chunk[ymax] - chunk[y]
            chunk_yerr = np.asarray([err_lo.to_numpy(), err_hi.to_numpy()])
        else:
            chunk_yerr = None
        offset = dodge_map.get(hue_val, 0)
        ax.errorbar(chunk["_xcoord"] + offset, chunk[y], yerr=chunk_yerr, ecolor=err_color, **errorbar_kwargs)
