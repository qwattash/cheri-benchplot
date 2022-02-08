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


def align_y_at(ax1, p1, ax2, p2):
    """
    Align p1 on axis ax1 to p2 on axis ax2.
    The aligment takes care to avoid clipping data.
    Note: The ylim should have been set on the axes prior to
    calling this function, otherwise the limits will not be
    picked up correctly.
    """
    ymin_ax1, ymax_ax1 = ax1.get_ylim()
    ymin_ax2, ymax_ax2 = ax2.get_ylim()
    t_ax1 = ax1.get_yaxis_transform()
    inv_ax1 = t_ax1.inverted()
    t_ax2 = ax2.get_yaxis_transform()
    inv_ax2 = t_ax2.inverted()

    # First align the p1/p2 points
    _, t_p1 = t_ax1.transform((0, p1))
    _, t_p2 = t_ax2.transform((0, p2))
    t_dy = t_p1 - t_p2
    _, dy = inv_ax2.transform((0, 0)) - inv_ax2.transform((0, t_p1 - t_p2))
    ymin_ax2, ymax_ax2 = ymin_ax2 + dy, ymax_ax2 + dy
    ax2.set_ylim(ymin_ax2, ymax_ax2)

    # Now we restore the initial range that we had on ax2 to avoid any
    # clipping. ax1 is unchanged so we assume that there is no clipping there
    _, ext_ax1 = inv_ax1.transform((0, 0)) - inv_ax1.transform((0, t_dy))
    ext_ax2 = dy
    if dy > 0:
        # We moved up so we need to extend the low ylim
        ymin_ax1 -= ext_ax1
        ymin_ax2 -= ext_ax2
    elif dy < 0:
        # We moved down so we need to extend the high ylim
        ymax_ax1 += ext_ax1
        ymax_ax2 += ext_ax2
    ax1.set_ylim(ymin_ax1, ymax_ax1)
    ax2.set_ylim(ymin_ax2, ymax_ax2)


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
        legend_info = cell.get_legend_info(view)
        style_args = build_style_args(view.style)

        for c in view.horizontal:
            for k, y in zip(legend_key, view.df[c]):
                color = legend_info.color(k) if legend_info else None
                line = ctx.ax.axhline(y, color=color, **style_args)
                ctx.legend.set_item(legend_info.label(k), line)
        for c in view.vertical:
            for k, x in zip(legend_key, view.df[c]):
                color = legend_info.color(k) if legend_info else None
                line = ctx.ax.axvline(x, color=color, **style_args)
                ctx.legend.set_item(legend_info.label(k), line)


class BarRenderer(ViewRenderer):
    """
    Render a bar plot.
    """
    def _resolve_groups(self, cell, view):
        """
        Compute the bar groups. The resulting group keys will
        have either 1 or 2 index levels, depending on the bar_group and
        stack_group settings
        """
        by = []

        if view.bar_group:
            by.append(view.bar_group)
        if view.stack_group:
            by.append(view.stack_group)

        if len(by) == 0:
            by = cell.get_legend_level(view)
            if by is None:
                by = view.df.index.names
        return by

    def _compute_bar_x(self, cell, view, ax, group_keys):
        """
        Compute the group center x for the X axis and then
        the x for each bar group.
        Returns the center x vector and the modified view dataframe.

        Note: this assumes that the input dataframe is aligned on the
        bar_group and stack_group levels
        """
        if view.has_yleft and view.has_yright:
            naxes = 2
        else:
            naxes = 1
        if view.bar_group:
            ngroups = len(view.get_col(view.bar_group).unique())
        else:
            ngroups = 1
        if view.stack_group:
            nstacks = len(view.get_col(view.stack_group).unique())
        else:
            nstacks = 1

        def _transform_helper(xform, v):
            tmp_points = np.column_stack([v, np.repeat(0, len(v))])
            tmp_tx = xform.transform(tmp_points)
            result = tmp_tx[:, 0]
            return result.ravel()

        x_trans = ax.get_xaxis_transform()
        x_inv = x_trans.inverted()
        transform_helper = ft.partial(_transform_helper, x_trans)
        transform_inv_helper = ft.partial(_transform_helper, x_inv)

        xcol = sorted(view.get_x().unique())
        # Find the space between each x tick, we scale the bars to fit into the interval
        # Assume that each x in xcol is the center of the bar group.
        # Since the x axis may be subject to a transformation, we need to take that into
        # account when computing widths so that the bars have uniform size.
        # 1. find the distance between each x value
        # 2. transform each width into axis coordinates
        # 3. split the intervals to fit the bars
        # 4. transform back the offsets
        # transform() requires an Nx2 array, we ignore the Y axis here
        tx_xcol = transform_helper(xcol)
        tx_space = np.diff(tx_xcol)
        assert len(tx_space) >= 1, "need to handle the special case of only 1 X coordinate"
        tx_space = np.append(tx_space, tx_space[-1])
        assert tx_space.shape == tx_xcol.shape

        # allocate bar width
        assert view.bar_width > 0 and view.bar_width <= 1
        tx_width = (tx_space * view.bar_width) / (ngroups * naxes)
        # relative offsets in width units, starting from the related x position
        # if centering, this will need a translation
        rel_offsets = np.arange(0, (ngroups * naxes))
        # generate transformed offset matrix via column * row vector multiplication
        # Note that half of the groups are for the yright axis if it exists
        tx_width_vector = tx_width.reshape(len(tx_width), 1)
        rel_offsets_vector = rel_offsets.reshape(1, len(rel_offsets))
        # the row is indexed by x, the colum is indexed by group
        tx_offset_matrix = tx_width_vector * rel_offsets_vector
        assert tx_offset_matrix.shape == (len(tx_xcol), ngroups * naxes)

        # apply the offsets to the transformed x column
        # shift the positions to center the groups at the given x column
        tx_position_matrix = tx_offset_matrix + tx_xcol[:, np.newaxis]
        if view.bar_group_location == "center":
            align_offset = tx_width / 2 - (tx_space * view.bar_width) / 2
            tx_position_matrix += align_offset[:, np.newaxis]
        elif view.bar_group_location == "left":
            # XXX TODO
            pass
        else:
            assert False, "Not Reached"
        # compute bar widths, these are taken from the position matrix that gives the center of each bar
        tx_bar_start = tx_position_matrix - tx_width[:, np.newaxis] / 2
        tx_bar_end = tx_position_matrix + tx_width[:, np.newaxis] / 2
        assert tx_bar_start.shape == tx_position_matrix.shape
        assert tx_bar_end.shape == tx_position_matrix.shape

        # transform back the offsets, as usual we need to add an extra useless dimension
        # and drop it later
        position_matrix = np.apply_along_axis(transform_inv_helper, 1, tx_position_matrix)
        assert position_matrix.shape == tx_position_matrix.shape
        bar_start = np.apply_along_axis(transform_inv_helper, 1, tx_bar_start)
        assert bar_start.shape == tx_bar_start.shape
        bar_end = np.apply_along_axis(transform_inv_helper, 1, tx_bar_end)
        assert bar_end.shape == tx_bar_end.shape
        bar_width = bar_end - bar_start

        # set bar width so that the size stays constant regardless of the X scale
        # for each x we have bar width differences, these are repeated for each stack.
        # We need to flatten the bar_width matrix in column-major order.
        # We do the same for the position matrix.
        axis_split = position_matrix.shape[1] / naxes
        assert axis_split == int(axis_split)
        axis_split = int(axis_split)

        bar_x_left = position_matrix[:, 0:axis_split]
        bar_x_flat = bar_x_left.ravel(order="F")
        view.df["__bar_x_left"] = np.tile(bar_x_flat, nstacks)
        bar_width_left = bar_width[:, 0:axis_split]
        bar_width_flat = bar_width_left.ravel(order="F")
        view.df["__bar_width_left"] = np.tile(bar_width_flat, nstacks)

        if view.has_yright:
            bar_x_right = position_matrix[:, axis_split:]
            bar_x_flat = bar_x_right.ravel(order="F")
            view.df["__bar_x_right"] = np.tile(bar_x_flat, nstacks)
            bar_width_right = bar_width[:, axis_split:]
            bar_width_flat = bar_width_right.ravel(order="F")
            view.df["__bar_width_right"] = np.tile(bar_width_flat, nstacks)

        # And the stacking Y bases
        view.df["__bar_y_left_base"] = 0
        view.df["__bar_y_right_base"] = 0
        if view.stack_group:
            non_stack_idx = view.df.index.names.difference([view.stack_group])
            grouped = view.df.groupby(non_stack_idx)
            view.df["__bar_y_left_base"] = grouped[view.yleft].apply(
                lambda stack_slice: stack_slice.cumsum().shift(fill_value=0))
            if view.has_yright:
                view.df["__bar_y_right_base"] = grouped[view.yright].apply(
                    lambda stack_slice: stack_slice.cumsum().shift(fill_value=0))
        return view.df

    def render(self, view, cell, surface, ctx):
        """
        For the input dataframe we split the column groups
        and stack groups. Each grouping have the associated
        x, yleft and yright columns, which are used to plot
        the data.
        """

        legend_info = cell.get_legend_info(view)
        by = self._resolve_groups(cell, view)
        df = self._compute_bar_x(cell, view, ctx.ax, by)
        groups = df.groupby(by)

        left_patches = []
        right_patches = []
        for key, chunk in groups:
            legend_key = cell.get_legend_col(view, chunk).unique()
            # There can be only one color for each group/stack
            assert len(legend_key) == 1
            legend_key = legend_key[0]
            l_color = legend_info.color(legend_key, axis="left")
            r_color = legend_info.color(legend_key, axis="right")
            l_label = legend_info.label(legend_key, axis="left")
            r_label = legend_info.label(legend_key, axis="right")

            if view.has_yleft:
                values = view.get_col(view.yleft, chunk)
                x = view.get_col("__bar_x_left", chunk)
                base = view.get_col("__bar_y_left_base", chunk)
                width = view.get_col("__bar_width_left", chunk)
                patches = ctx.ax.bar(x, height=values, bottom=base, color=l_color, width=width)
                ctx.legend.set_item(l_label, patches)
            if view.has_yright:
                values = view.get_col(view.yright, chunk)
                x = view.get_col("__bar_x_right", chunk)
                base = view.get_col("__bar_y_right_base", chunk)
                width = view.get_col("__bar_width_right", chunk)
                patches = ctx.rax.bar(x, height=values, bottom=base, color=r_color, width=width)
                ctx.legend.set_item(r_label, patches)

        if view.has_yleft and view.has_yright:
            # Need to realign the Y axes 0 coordinates
            align_y_at(ctx.ax, 0, ctx.rax, 0)


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

        legend_info = cell.get_legend_info(view)
        legend_col = cell.get_legend_col(view)
        idx_values = legend_col.unique()

        # Note: xvec retains the bucket_group index (or more generally the legend_level)
        if view.bucket_group:
            groups = xcol.groupby(view.bucket_group)
            xvec = [chunk for _, chunk in groups]
            keys = [k for k, _ in groups]
            colors = legend_info.color(keys)
            labels = legend_info.label(keys)
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
        if self.config.split_subplots:
            r = c = 1
        else:
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
        if cfg.tick_labels is not None:
            ax.set_xticklabels(cfg.tick_labels, rotation=cfg.tick_rotation)

    def _config_y(self, cfg, ax):
        ax.set_ylabel(cfg.label)
        if cfg.scale:
            args, kwargs = build_scale_args(cfg.scale)
            ax.set_yscale(*args, **kwargs)
        if cfg.limits:
            ax.set_ylim(cfg.limits[0], cfg.limits[1])
        if cfg.ticks:
            ax.set_yticks(cfg.ticks)
        if cfg.tick_labels is not None:
            ax.set_yticklabels(cfg.tick_labels, rotation=cfg.tick_rotation)

    def draw(self, ctx):
        ctx.ax = ctx.axes[ctx.row][ctx.col]
        ctx.legend = Legend()
        # Auto enable yright axis if it is used by any view
        for view in self.views:
            if hasattr(view, "yright") and len(view.yright) > 0:
                self.yright_config.enable = True
                break
        if self.yright_config:
            ctx.rax = ctx.ax.twinx()
        else:
            ctx.rax = None

        # If ylimits are not defined, pick up the limits from the views here.
        # TODO

        # Configure axes before rendering, this allows the renderers to grab
        # any axis scale transform that may be set, so that any scaling is
        # generic.
        # The drawback is that we can not change the axis configuration safely
        # after/during rendering (especially scaling factors). This was not
        # intended to happen in the first place, so it should be ok.
        # It may still be possible to perform some adjustment of the viewport
        # if needed.
        if self.x_config:
            self._config_x(self.x_config, ctx.ax)
        if self.yleft_config:
            self._config_y(self.yleft_config, ctx.ax)
        if self.yright_config:
            assert ctx.rax is not None, "Missing twin axis"
            self._config_y(self.yright_config, ctx.rax)

        # Render all the views in the cell
        for view in self.views:
            r = self.surface.get_renderer(view)
            r.render(view, self, self.surface, ctx)
        # Always render an horizontal line at origin
        ctx.ax.axhline(0, linestyle="--", linewidth=0.5, color="black")

        ctx.legend.build_legend(ctx)
        ctx.ax.set_title(self.title)
