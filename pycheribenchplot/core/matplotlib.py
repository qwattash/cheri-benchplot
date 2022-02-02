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

from .plot import (BarPlotDataView, CellData, Scale, Style, Surface, ViewRenderer)


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


def build_scale_args(scale: Scale) -> tuple[list, dict]:
    """
    Convert a Scale wrapper into scale arguments
    """
    args = [scale.name]
    kwargs = {}
    if scale.name == "log" or scale.name == "symlog":
        kwargs["base"] = scale.base
    if scale.name == "symlog":
        kwargs["linthresh"] = scale.lintresh
        kwargs["linscale"] = scale.linscale
    return (args, kwargs)


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

        legend_map = cell.get_legend_map(view)
        legend_col = cell.get_legend_col(view)
        if view.bar_group:
            groups = view.df.groupby(view.bar_group)
            data_groups = [g for _, g in groups]
            keys = [k for k, _ in groups]
        else:
            # single bar plot
            assert len(legend_col.unique()) == 1
            data_groups = [view.df]
            keys = [legend_col.unique()]

        # Generate x centers for the bars and the
        # relative offsets for each group
        if view.yleft is not None and view.yright is not None:
            naxes = 2
        else:
            naxes = 1
        ngroups = len(data_groups)
        width = (ngroups * naxes) * view.bar_width + 2 * view.bar_pad
        offsets = np.arange(0, width, naxes * view.bar_width)
        assert len(offsets) == ngroups * naxes
        # shift offsets to represent the bar centers
        offsets = offsets - width / 2 + view.bar_width / 2 + view.bar_pad
        # Normalize X axis to be numeric and assign bar labels
        xcol = view.get_x()
        x_labels = xcol
        x = np.arange(0, width * len(xcol), width) + width / 2

        left_patches = []
        right_patches = []
        for off, grp in zip(offsets, data_groups):
            if view.yleft is not None:
                values = view.get_col(view.yleft, grp)
                grp_x = x + off
                patches = ctx.ax.bar(grp_x, height=values, color="b")
                # ctx.legend.set_item("test", patches)
                # next offset
                off += view.bar_width
            if view.yright is not None:
                values = view.get_col(view.yright, grp)
                grp_x = x + off
                patches = ctx.rax.bar(grp_x, height=values, color="r")

            # ctx.legend.set_item("test", patches)
        # Fixup cell parameters


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
        fig, axes = plt.subplots(r, c, constrained_layout=True, figsize=(10 * c, 5 * r))
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

    def _config_x(self, cfg, ax):
        ax.set_xlabel(cfg.label)
        if cfg.scale:
            args, kwargs = build_scale_args(cfg.scale)
            ax.set_xscale(*args, **kwargs)
        if cfg.limits:
            ax.set_xlim(cfg.limits[0], cfg.limits[1])
        if cfg.ticks is not None:
            ax.set_xticks(cfg.ticks)

    def _config_y(self, cfg, ax):
        ax.set_ylabel(cfg.label)
        if cfg.scale:
            args, kwargs = build_scale_args(cfg.scale)
            ax.set_yscale(*args, **kwargs)
        if cfg.limits:
            ax.set_ylim(cfg.limits[0], cfg.limits[1])
        if cfg.ticks:
            ax.set_yticks(cfg.ticks)

    def draw(self, ctx):
        ctx.ax = ctx.axes[ctx.row][ctx.col]
        ctx.legend = Legend()
        # Auto enable yright axis if it is used by any view
        auto_yright = ft.reduce(lambda a, v: hasattr(v, "yright") and len(v.yright) > 0, self.views, False)
        if auto_yright:
            self.yright_config.enable = True
        if self.yright_config:
            ctx.rax = ctx.ax.twinx()
        else:
            ctx.rax = None
        # Render all the views in the cell
        for view in self.views:
            r = self.surface.get_renderer(view)
            r.render(view, self, self.surface, ctx)
        ctx.legend.build_legend(ctx)
        # Set all remaining cell parameters
        ctx.ax.set_title(self.title)
        if self.x_config:
            self._config_x(self.x_config, ctx.ax)
        if self.yleft_config:
            self._config_y(self.yleft_config, ctx.ax)
        if self.yright_config:
            assert ctx.rax is not None, "Missing twin axis"
            self._config_y(self.yright_config, ctx.rax)
