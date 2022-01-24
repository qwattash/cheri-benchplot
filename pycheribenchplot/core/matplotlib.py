import functools as ft
import typing
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .plot import BarPlotDataView, CellData, Surface, ViewRenderer


class MatplotlibBarRenderer(ViewRenderer):
    """
    Render a bar plot.
    """
    def render(self, cell, surface, ctx):
        pass


class MatplotlibHistRenderer(ViewRenderer):
    """
    Render an histogram plot.
    """
    def render(self, view, cell, surface, ctx):
        xcol = view.df[view.x]
        if view.bucket_group:
            groups = xcol.groupby(view.bucket_group)
            xvec = [chunk for _, chunk in groups]
        else:
            xvec = xcol
        if view.x_scale:
            ctx.ax.set_xscale(view.x_scale.name, base=view.x_scale.base)
        ctx.ax.hist(xvec, bins=view.buckets, rwidth=0.5)
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
        ax: typing.Optional[Axes] = None
        rax: typing.Optional[Axes] = None

    def __init__(self):
        super().__init__()
        self._renderers = {
            "bar": MatplotlibBarRenderer,
            "hist": MatplotlibHistRenderer,
        }

    def _make_draw_context(self, title, dest, **kwargs):
        r, c = self._layout.shape
        fig, axes = plt.subplots(r, c)
        # Normalize axes matrix to always be a list of lists
        if c == 1:
            axes = [axes]
        if r == 1:
            axes = [[ax] for ax in axes]
        fig.suptitle(title)
        ctx = super()._make_draw_context(title, dest, figure=fig, axes=axes)
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
        if ft.reduce(lambda a, v: len(v.yleft) > 0, self.views, False):
            # Some view uses the twin right axis, so create it
            ctx.rax = ctx.ax.twinx()
        else:
            ctx.rax = None
        for view in self.views:
            r = self.surface.get_renderer(view)
            r.render(view, self, self.surface, ctx)
        ctx.ax.set_title(self.title)
        ctx.ax.set_xlabel(self.x_label)
        ctx.ax.set_ylabel(self.yleft_label)
