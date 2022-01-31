import functools as ft
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from pandas.api.types import is_numeric_dtype

from .plot import BarPlotDataView, CellData, Style, Surface, ViewRenderer


def build_style_args(style: Style) -> dict:
    """
    Convert a Style wrapper object to matplotlib style arguments
    """
    args = {}
    if style.line_style is not None:
        args["linestyle"] = style.line_style
    if style.line_width:
        args["linewidth"] = style.line_width
    return args


class Legend:
    """
    Helper to build the legend
    """
    def __init__(self):
        self.handles = {}

    def set_item(self, label: str, handle: "matplotlib.artist.Artist"):
        self.handles[label] = handle

    def set_group(self, labels: list[str], handles: list["matplotlib.artist.Artist"]):
        for l, h in zip(labels, handles):
            self.set_item(l, h)

    def build_legend(self, ctx):
        if self.handles:
            ctx.ax.legend(handles=self.handles.values(), labels=self.handles.keys())


class SimpleLineRenderer(ViewRenderer):
    """
    Render vertical or horizontal lines
    """
    def render(self, view, cell, surface, ctx):
        legend_key = cell.get_legend_col(view)
        legend_map = cell.get_legend_map(view)
        style_args = build_style_args(view.style)

        for c in view.horizontal:
            for k, y in zip(legend_key, view.df[c]):
                color = legend_map.get_color(k) if legend_map else None
                line = ctx.ax.axhline(y, color=color, **style_args)
                ctx.legend.set_item(legend_map.get_label(k), line)
        for c in view.vertical:
            for k, x in zip(legend_key, view.df[c]):
                color = legend_map.get_color(k) if legend_map else None
                line = ctx.ax.axvline(x, color=color, **style_args)
                ctx.legend.set_item(legend_map.get_label(k), line)


class BarRenderer(ViewRenderer):
    """
    Render a bar plot.
    """
    def render(self, view, cell, surface, ctx):
        # Normalize X axis to be numeric and assign bar labels
        xcol = view.get_x()
        if not is_numeric_dtype(xcol.dtype):
            xcol_labels = xcol
            xcol = range(len(xcol))
        else:
            xcol_labels = xcol

        left_bar = view.get_yleft()
        right_bar = view.get_yright()
        if left_bar is not None:
            ctx.ax.bar(xcol, height=left_bar)
        # if right_bar is not None:
        #     ctx.rax.bar(xcol, height=right_bar)

        if view.x_scale:
            ctx.ax.set_xscale(view.x_scale.name, base=view.x_scale.base)


class HistRenderer(ViewRenderer):
    """
    Render an histogram plot.
    """
    def render(self, view, cell, surface, ctx):
        xcol = view.get_x()

        # Use the bucket group as legend level
        # This guarantees that the legend_level is always in the index of
        # the groups after groupby()
        # XXX we may have some more complex condition if we do not wish to use
        # the bucket group but for now this covers all use cases
        view.legend_level = view.bucket_group

        legend_map = cell.get_legend_map(view)
        legend_col = cell.get_legend_col(view)
        idx_values = legend_col.unique()

        # Note: xvec retains the bucket_group index (or more generally the legend_level)
        if view.bucket_group:
            groups = xcol.groupby(view.bucket_group)
            xvec = [chunk for _, chunk in groups]
            keys = [k for k, _ in groups]
            colors = legend_map.get_color(keys)
            labels = legend_map.get_label(keys)
            assert len(colors) == len(xvec), f"#colors({len(colors)}) does not match histogram #groups({len(xvec)})"
        else:
            assert len(xcol.shape) == 1 or xcol.shape[1] == 1
            xvec = xcol
            colors = None

        if view.x_scale:
            ctx.ax.set_xscale(view.x_scale.name, base=view.x_scale.base)
        n, bins, patches = ctx.ax.hist(xvec, bins=view.buckets, rwidth=0.5, color=colors)
        ctx.legend.set_group(labels, patches)
        ctx.ax.set_xticks(view.buckets)


class MatplotlibSurface(Surface):
    """
    Draw plots using matplotlib
    """
    @dataclass
    class DrawContext(Surface.DrawContext):
        """
        Arguments:
        figure: the matplotlib figure
        axes: axes matrix, mirroring the surface layout
        ax: current left axis to use
        rax: current right axis to use
        """
        figure: Figure
        axes: list[list[Axes]]
        legend: Legend
        ax: typing.Optional[Axes] = None
        rax: typing.Optional[Axes] = None

    def __init__(self):
        super().__init__()
        self._renderers = {
            "bar": BarRenderer,
            "hist": HistRenderer,
            "axline": SimpleLineRenderer,
        }

    def _make_draw_context(self, title, dest, **kwargs):
        r, c = self._layout.shape
        fig, axes = plt.subplots(r, c, tight_layout=True, figsize=(10 * c, 5 * r))
        axes = np.array(axes)
        axes = axes.reshape((r, c))
        fig.suptitle(title)
        ctx = super()._make_draw_context(title, dest, figure=fig, axes=axes, legend=Legend())
        return ctx

    def _finalize_draw_context(self, ctx):
        ctx.figure.savefig(ctx.dest.with_suffix(".pdf"))

    def make_cell(self, **kwargs):
        return MatplotlibPlotCell(**kwargs)


class MatplotlibPlotCell(CellData):
    """
    A cell in the matplotlib plot. At render time, this will be associated
    to a unique set of axes.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.figure = None
        self.ax = None

    def draw(self, ctx):
        ctx.ax = ctx.axes[ctx.row][ctx.col]
        if ft.reduce(lambda a, v: hasattr(v, "yleft") and len(v.yleft) > 0, self.views, False):
            # Some view uses the twin right axis, so create it
            ctx.rax = ctx.ax.twinx()
        else:
            ctx.rax = None
        for view in self.views:
            r = self.surface.get_renderer(view)
            r.render(view, self, self.surface, ctx)
        ctx.legend.build_legend(ctx)
        ctx.ax.set_title(self.title)
        ctx.ax.set_xlabel(self.x_label)
        ctx.ax.set_ylabel(self.yleft_label)
