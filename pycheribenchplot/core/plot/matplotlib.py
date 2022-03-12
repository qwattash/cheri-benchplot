import functools as ft
import itertools as it
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

from .backend import FigureManager, Mosaic, ViewRenderer
from .data_view import BarPlotDataView, CellData, Scale, Style


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


def transform_x_helper(xform, v):
    """
    Matplotlib transform helper to apply transformation on the X axis
    """
    tmp_points = np.column_stack([v, np.repeat(0, len(v))])
    tmp_tx = xform.transform(tmp_points)
    result = tmp_tx[:, 0]
    return result.ravel()


def transform_y_helper(xform, v):
    """
    Matplotlib transform helper to apply transformation on the Y axis
    """
    tmp_points = np.column_stack([np.repeat(0, len(v)), v])
    tmp_tx = xform.transform(tmp_points)
    result = tmp_tx[:, 1]
    return result.ravel()


def align_y_at(ax1, p1, ax2, p2):
    """
    Align p1 on axis ax1 to p2 on axis ax2.
    The aligment takes care to avoid clipping data.
    Note: The ylim should have been set on the axes prior to
    calling this function, otherwise the limits will not be
    picked up correctly.
    """
    ymin_ax1, ymax_ax1 = ax1.get_ylim()
    pre_ymin_ax1, pre_ymax_ax1 = ymin_ax1, ymax_ax1
    ymin_ax2, ymax_ax2 = ax2.get_ylim()
    pre_ymin_ax2, pre_ymax_ax2 = ymin_ax2, ymax_ax2
    t_ax1 = ax1.get_yaxis_transform()
    inv_ax1 = t_ax1.inverted()
    t_ax2 = ax2.get_yaxis_transform()
    inv_ax2 = t_ax2.inverted()

    # First align the p1/p2 points
    _, t_p1 = t_ax1.transform((0, p1))
    _, t_p2 = t_ax2.transform((0, p2))
    t_dy = t_p1 - t_p2
    _, dy = inv_ax2.transform((0, 0)) - inv_ax2.transform((0, t_dy))
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
        ymax_ax1 -= ext_ax1
        ymax_ax2 -= ext_ax2
    ax1.set_ylim(ymin_ax1, ymax_ax1)
    ax2.set_ylim(ymin_ax2, ymax_ax2)
    # The limits we set must be monotonic, this function is not allowed to ever
    # shrink the viewport
    assert ymin_ax1 <= pre_ymin_ax1, f"{ymin_ax1} <= {pre_ymin_ax1}"
    assert ymax_ax1 >= pre_ymax_ax1, f"{ymax_ax1} >= {pre_ymax_ax1}"
    assert ymin_ax2 <= pre_ymin_ax2, f"{ymin_ax2} <= {pre_ymin_ax2}"
    assert ymax_ax2 >= pre_ymax_ax2, f"{ymax_ax2} >= {pre_ymax_ax2}"


class Legend:
    """
    Helper to build the legend
    """
    def __init__(self, legend_position="top"):
        self.handles = {}
        self.legend_position = legend_position

    def set_item(self, label: str, handle: "matplotlib.artist.Artist"):
        self.handles[label] = handle

    def set_group(self, labels: list[str], handles: list["matplotlib.artist.Artist"]):
        for l, h in zip(labels, handles):
            self.set_item(l, h)

    def _compute_legend_position(self):
        """
        Legend position can be top, bottom, left, right.
        The legend is always generated outside the axes
        """
        nentries = len(self.handles)
        if self.legend_position == "top":
            loc = "lower left"
            bbox = (0, 1.03, 1, 0.2)
            # Empirically allocate 3 columns. This should be computed
            # based on the pixel width of the axes in the figure
            ncol = 3
        elif self.legend_position == "bottom":
            pass
        elif self.legend_position == "left":
            pass
        elif self.legend_position == "right":
            pass
        else:
            assert False, "Invalid legend position"
        self.bbox = bbox
        legend_args = {
            "loc": loc,
            "bbox_to_anchor": bbox,
            "ncol": ncol,
            "fontsize": "small",
            "mode": "expand",
            "borderaxespad": 0.0,
        }
        return legend_args

    def build_legend(self, cell):
        if self.handles:
            legend_kwargs = self._compute_legend_position()
            cell.ax.legend(handles=self.handles.values(), labels=self.handles.keys(), **legend_kwargs)


class SimpleLineRenderer(ViewRenderer):
    """
    Render vertical or horizontal lines
    """
    def _find_legend_entry(self, view):
        """
        For each column, there are N line coordines. Each coordinate is in a different
        row and should have a value associated in the legend_level column(s)
        """
        # get_col always returns a Dataframe, if we have only one level we need to squeeze
        legend_key = view.get_col(view.legend_level)
        color = view.legend_info.color(legend_key)
        label = view.legend_info.label(legend_key)
        return color, label

    def render(self, view, cell):
        # legend key is always a dataframe because legend_level is normalized to a list
        style_args = build_style_args(view.style)
        color_set, label_set = self._find_legend_entry(view)

        for c in view.horizontal:
            for coord, color, label in zip(view.df[c], color_set, label_set):
                line = cell.ax.axhline(coord, color=color, **style_args)
                cell.legend.set_item(label, line)
        for c in view.vertical:
            for coord, color, label in zip(view.df[c], color_set, label_set):
                line = cell.ax.axvline(coord, color=color, **style_args)
                cell.legend.set_item(label, line)


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
            by = view.legend_level
        return by

    def _iter_group_slices(self, view, ngroups, naxes):
        """
        Generate the position and bar width matrix column slices associated
        to each group of bars to plot. In the simple case, each group is associated to an offset
        for each X point. If multiple columns are present there will be more than 1 matrix colum
        that will be flattened when assigning X position data.
        """
        if view.bar_axes_ordering == "sequential":
            col_set_size = ngroups
            col_index = 0
            for i in range(len(view.yleft)):
                group_slice = slice(col_index * col_set_size, (col_index + 1) * col_set_size)
                yield (group_slice, i, "l")
                col_index += 1
            for i in range(len(view.yright)):
                group_slice = slice(col_index * col_set_size, (col_index + 1) * col_set_size)
                yield (group_slice, i, "r")
                col_index += 1
        elif view.bar_axes_ordering == "interleaved":
            n_ycols = max(len(view.yleft), len(view.yright))
            col_index = 0
            for i in range(n_ycols):
                if i < len(view.yleft):
                    group_slice = slice(col_index, ngroups * naxes + 1, naxes)
                    yield (group_slice, i, "l")
                    col_index += 1
                if i < len(view.yright):
                    group_slice = slice(col_index, ngroups * naxes + 1, naxes)
                    yield (group_slice, i, "r")
                    col_index += 1
        else:
            raise ValueError(f"Invalid {view}::bar_axes_ordering: only 'sequential' and 'interleaved' allowed")

    def _compute_bar_x(self, cell, view, ax, group_by):
        """
        Compute the group center x for the X axis and then
        the x for each bar group.
        Returns the center x vector and the modified view dataframe.

        Note: this assumes that the input dataframe is aligned on the
        bar_group and stack_group levels.
        Operate on the assumption that waterver sorting was applied to the input
        data, it is reflected on the X axis ordering.
        """
        assert view.df.index.is_unique, "Require unique index"
        naxes = 0
        if view.has_yleft and view.has_yright:
            # naxes = 2
            nleft_cols = len(view.yleft) if isinstance(view.yleft, list) else 1
            nright_cols = len(view.yright) if isinstance(view.yright, list) else 1
            naxes = nleft_cols + nright_cols
        else:
            # naxes = 1
            nleft_cols = len(view.yleft) if isinstance(view.yleft, list) else 1
            naxes = nleft_cols
        if view.bar_group:
            ngroups = len(view.df.groupby(view.bar_group))
        else:
            ngroups = 1
        if view.stack_group:
            nstacks = len(view.df.groupby(view.stack_group))
        else:
            nstacks = 1
        assert naxes > 0 and ngroups >= 1 and nstacks >= 1

        x_trans = ax.get_xaxis_transform()
        x_inv = x_trans.inverted()
        transform_helper = ft.partial(transform_x_helper, x_trans)
        transform_inv_helper = ft.partial(transform_x_helper, x_inv)

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

        # 1. Apply the offsets to the transformed x column.
        # 2. Shift the positions to center the groups at the given x column.
        # Assuming that we have ngroups * naxes = 4 and 4 X points in xcol,
        # the position matrix will look like:
        #    _                     _     _    _     _      _     _      _
        #   |  W0*0 W0*1 W0*2 W0*3  |   |  X0  |   |  W0/2  |   |  S0/2  |
        #   |  W1*0 W1*1 W1*2 W1*3  | + |  X1  | + |  W1/2  | - |  S1/2  |
        #   |  W2*0 W2*1 W2*2 W2*3  |   |  X2  |   |  W2/2  |   |  S2/2  |
        #   |_ W3*0 W3*1 W3*2 W3*3 _|   |_ X3 _|   |_ W3/2 _|   |_ S3/2 _|
        # Where Wi are the transformed (linear space) single-bar width for each X group
        # 0..N are the bar indexes in each X group, Xi are the X point center coordinates.
        # The + W/2 - S/2 term translates the positions to center them on the X points,
        # where S is the total space allocated for a bar group at an X point. The extra W/2
        # is there because the leftmost bar is center-aligned to the X point, we want to
        # reference the total left position of that bar instead.
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
        # XXX we have to take into account the axis (0, 0) for the position matrix, maybe?
        position_matrix = np.apply_along_axis(transform_inv_helper, 1, tx_position_matrix)
        assert position_matrix.shape == tx_position_matrix.shape
        bar_start = np.apply_along_axis(transform_inv_helper, 1, tx_bar_start)
        assert bar_start.shape == tx_bar_start.shape
        bar_end = np.apply_along_axis(transform_inv_helper, 1, tx_bar_end)
        assert bar_end.shape == tx_bar_end.shape
        bar_width = bar_end - bar_start

        # Set bar width so that the size stays constant regardless of the X scale
        # for each x we have bar width differences, these are repeated for each stack.
        # We need to flatten the bar_width matrix in column-major order.
        # We do the same for the position matrix.
        # Care must be taken to preserve the X-data mapping. The dataframe may not be
        # sorted by X value.
        if view.stack_group:
            non_stack_idx = view.df.index.names.difference([view.stack_group])
            stack_groups = view.df.groupby(non_stack_idx)
        # Make sure we sort frame by X
        view.df = view.df.sort_values(group_by + [view.x], ascending=True)

        for group_slice, col, side in self._iter_group_slices(view, ngroups, naxes):
            # group_slice = slice(matrix_col * col_group_split, (matrix_col + 1) * col_group_split)
            col_name = view.yleft[col] if side == "l" else view.yright[col]
            bar_x_group = position_matrix[:, group_slice]
            bar_x_group_flat = bar_x_group.ravel(order="F")
            view.df[f"__bar_x_{side}{col}"] = np.tile(bar_x_group_flat, nstacks)
            bar_width_group = bar_width[:, group_slice]
            bar_width_flat = bar_width_group.ravel(order="F")
            view.df[f"__bar_width_{side}{col}"] = np.tile(bar_width_flat, nstacks)
            if view.stack_group:
                view.df[f"__bar_y_base_{side}{col}"] = stack_groups[col_name].apply(
                    lambda stack_slice: stack_slice.cumsum().shift(fill_value=0))
            else:
                view.df[f"__bar_y_base_{side}{col}"] = 0
        return view.df

    def _find_legend_entry(self, view, group_levels: list, group_key: tuple, col_name: str, axis: str):
        """
        Resolve a group, column, axis set to a color and label
        """
        Key = view.legend_info.build_key()
        # normalize group key to be a tuple
        if len(group_levels) == 1:
            group_key = (group_key, )

        key_levels = {}
        for lvl in view.legend_level:
            lvl_idx = group_levels.index(lvl)
            assert lvl_idx >= 0, "Legend level should always be part of the groupby levels"
            key_levels[lvl] = group_key[lvl_idx]
        if (len(view.yleft) > 1 and axis == "left") or (len(view.yright) > 1 and axis == "right"):
            # Expect the "column" legend level to exist
            key_levels["column"] = col_name
        if view.has_yleft and view.has_yright:
            # Expect the "axis" legend level to exist
            key_levels["axis"] = axis

        key = Key(**key_levels)
        return view.legend_info.find(key)

    def _draw_columns_text(self, view, cell):
        """
        Draw labels at the top (or bottom) of the bars.
        """
        if not view.bar_text:
            return

        non_stack_idx = view.df.index.names.difference([view.stack_group])
        for key, chunk in view.df.groupby(non_stack_idx):
            # Sum stacked values to find the max and draw the text at the Y
            l = [("l", idx, name) for idx, name in enumerate(view.yleft)]
            r = [("r", idx, name) for idx, name in enumerate(view.yright)]
            for side, col_idx, col_name in l + r:
                # Since we are grouping the complement of the stack group index, we only
                # have a single X coordinate for all values of the stack.
                x = chunk[f"__bar_x_{side}{col_idx}"].iloc[0]
                # Would be nice to know if this was a percentage
                # maybe the text pattern can be expressed in the legend info
                # as we keep the color already for each column "group"
                maxv = chunk[col_name].sum()
                if np.isnan(maxv) or np.isinf(maxv):
                    txt = str(maxv)
                    maxv = 0
                elif maxv - int(maxv) != 0:
                    txt = f"{maxv:.1f}"
                else:
                    txt = f"{int(maxv):d}"
                ax = cell.ax if side == "l" else cell.rax
                y_trans = ax.get_yaxis_transform()
                y_inv = y_trans.inverted()
                # Text padding depends on the scale (log/linear)
                if maxv < 0:
                    va = "top"
                    pad_direction = -1
                else:
                    va = "bottom"
                    pad_direction = 1
                if view.bar_text_pad > 0:
                    tx_pad = transform_y_helper(y_trans, [maxv]) + view.bar_text_pad * pad_direction
                    maxv = transform_y_helper(y_inv, tx_pad)[0]
                ax.text(x, maxv, txt, fontsize="x-small", rotation="vertical", va=va, ha="center")

    def render(self, view, cell):
        """
        For the input dataframe we split the column groups
        and stack groups. Each grouping have the associated
        x, yleft and yright columns, which are used to plot
        the data.
        """
        by = self._resolve_groups(cell, view)
        df = self._compute_bar_x(cell, view, cell.ax, by)
        groups = df.groupby(by)

        left_patches = []
        right_patches = []
        for key, chunk in groups:
            if view.has_yleft:
                for col_idx, col_name in enumerate(view.yleft):
                    color, label = self._find_legend_entry(view, by, key, col_name, "left")
                    values = view.get_col(col_name, chunk).squeeze()
                    x = view.get_col(f"__bar_x_l{col_idx}", chunk)
                    base = view.get_col(f"__bar_y_base_l{col_idx}", chunk)
                    width = view.get_col(f"__bar_width_l{col_idx}", chunk)
                    patches = cell.ax.bar(x, height=values, bottom=base, color=color, width=width)
                    cell.legend.set_item(label, patches)
            if view.has_yright:
                for col_idx, col_name in enumerate(view.yright):
                    color, label = self._find_legend_entry(view, by, key, col_name, "right")
                    values = view.get_col(col_name, chunk).squeeze()
                    x = view.get_col(f"__bar_x_r{col_idx}", chunk)
                    base = view.get_col(f"__bar_y_base_r{col_idx}", chunk)
                    width = view.get_col(f"__bar_width_r{col_idx}", chunk)
                    patches = cell.rax.bar(x, height=values, bottom=base, color=color, width=width)
                    cell.legend.set_item(label, patches)

        self._draw_columns_text(view, cell)
        if view.has_yleft and view.has_yright:
            # Need to realign the Y axes 0 coordinates
            align_y_at(cell.ax, 0, cell.rax, 0)


class HistRenderer(ViewRenderer):
    """
    Render an histogram plot.
    """
    def render(self, view, cell):
        """
        Note: This uses the bucket group as legend level
        This guarantees that the legend_level is always in the index of
        the groups after groupby()
        """
        xcol = view.get_x()

        # Note: xvec retains the bucket_group index (or more generally the legend_level)
        if view.bucket_group:
            groups = xcol.groupby(view.bucket_group)
            xvec = [chunk for _, chunk in groups]
            keys = [k for k, _ in groups]
            colors = view.legend_info.color(keys)
            labels = view.legend_info.label(keys)
            assert len(colors) == len(xvec), f"#colors({len(colors)}) does not match histogram #groups({len(xvec)})"
        else:
            assert len(xcol.shape) == 1 or xcol.shape[1] == 1
            xvec = xcol
            # TODO honor the legend_info here as well
            legend_key = view.get_col(view.legend_level).unique()
            assert len(legend_key) == 1
            colors = view.legend_info.color(legend_key)
            labels = view.legend_info.label(legend_key)

        n, bins, patches = cell.ax.hist(xvec, bins=view.buckets, rwidth=0.5, color=colors, align=view.bar_align)
        cell.legend.set_group(labels, patches)


class MplFigureManager(FigureManager):
    """Draw plots using matplotlib"""
    def __init__(self, config):
        super().__init__(config)
        # Initialized when allocating cells
        self.figures = []

    def allocate_cells(self, mosaic: Mosaic):
        if self.config.split_subplots:
            # Don't care about layout as we generate only one plot per figure
            for subplot in mosaic:
                figure = plt.figure(constrained_layout=True, figsize=(10, 7))
                ax = figure.subplots(1, 1)
                cell = MplCellData(title=subplot.get_cell_title(), figure=figure, ax=ax)
                subplot.cell = cell
                self.figures.append(figure)
        else:
            # Mosaic shape must be at leaset (N, 1)
            nrows, ncols = mosaic.shape()
            figure = plt.figure(constrained_layout=True, figsize=(10 * ncols, 7 * nrows))
            self.figures.append(figure)
            mosaic_axes = figure.subplot_mosaic(mosaic.layout)
            for name, subplot in mosaic.subplots.items():
                ax = mosaic_axes[name]
                subplot.cell = MplCellData(title=subplot.get_cell_title(), figure=figure, ax=ax)

    def _write(self, dest: Path, figure: Figure):
        for ext in self.config.plot_output_format:
            path = dest.with_suffix(f".{ext}")
            self.logger.debug("Emit %s plot %s", ext, path)
            figure.savefig(path)
        plt.close(figure)

    def draw(self, mosaic, title, dest):
        super().draw(mosaic, title, dest)
        if self.config.split_subplots:
            base = dest.parent / "split"
            base.mkdir(exist_ok=True)
            stem = dest.stem
            for idx, subplot in enumerate(mosaic):
                assert subplot.cell is not None, f"Missing cell for subplot {subplot}"
                subplot.cell.draw()
                cell_dest = base / dest.with_stem(f"{stem}-{idx}").name
                self._write(cell_dest, subplot.cell.figure)
        else:
            assert len(self.figures) == 1, "Unexpected number of figures"
            self.figures[0].suptitle(title)
            for subplot in mosaic:
                subplot.cell.draw()
            self._write(dest, self.figures[0])


class MplCellData(CellData):
    """
    A cell in the matplotlib plot. At render time, this will be associated
    to a unique set of axes.
    """
    def __init__(self, figure=None, ax=None, **kwargs):
        super().__init__(**kwargs)
        self._renderers = {
            "bar": BarRenderer,
            "hist": HistRenderer,
            "axline": SimpleLineRenderer,
        }
        self.legend = Legend()
        self.figure = figure
        self.ax = ax
        # Right axis is autogenerated based on the axes config
        self.rax = None

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

    def _pad_y_axis(self, ax, padding):
        ymin, ymax = ax.get_ylim()
        _, fig_pad = ax.transAxes.transform([0, padding]) - ax.transAxes.transform([0, 0])
        inv_tx = ax.transData.inverted()
        _, y_pad = inv_tx.transform([0, fig_pad]) - inv_tx.transform([0, 0])
        ax.set_ylim(ymin - y_pad, ymax + y_pad)

    def draw(self):
        assert self.figure, "Missing cell figure"
        assert self.ax, "Missing cell axes"

        # Auto enable yright axis if it is used by any view
        for view in self.views:
            if hasattr(view, "yright") and len(view.yright) > 0:
                self.yright_config.enable = True
                break
        if self.yright_config:
            self.rax = self.ax.twinx()

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
            self._config_x(self.x_config, self.ax)
        if self.yleft_config:
            self._config_y(self.yleft_config, self.ax)
        if self.yright_config:
            assert self.rax is not None, "Missing twin axis"
            self._config_y(self.yright_config, self.rax)

        # Render all the views in the cell
        for view in self.views:
            r = self.get_renderer(view)
            if not r:
                continue
            r.render(view, self)
        # Always render an horizontal line at origin
        self.ax.axhline(0, linestyle="--", linewidth=0.5, color="black")

        # Pad the viewport
        if self.yleft_config:
            self._pad_y_axis(self.ax, self.yleft_config.padding)
        if self.yright_config:
            self._pad_y_axis(self.rax, self.yright_config.padding)

        self.legend.build_legend(self)
        if self.legend.legend_position == "top":
            title_y = max(1.0, self.legend.bbox[1] + self.legend.bbox[3])
        self.ax.set_title(self.title, y=title_y, pad=6)
