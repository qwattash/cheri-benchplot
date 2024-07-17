
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

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
    df = df.with_columns(
        pl.col(x).replace({xc: xp for xc, xp in zip(x_cat, x_points) }).cast(float).alias("_xcoord")
    )

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
        chunk_yerr = chunk[yerr] if yerr else None
        offset = dodge_map.get(hue_val, 0)
        ax.errorbar(chunk["_xcoord"] + offset, chunk[y], yerr=chunk_yerr,
                    ecolor=err_color, **errorbar_kwargs)
