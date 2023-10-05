import numpy as np
import pandas as pd
import polars as pl
from matplotlib.axes import Axes


class DynamicCoordAllocator:
    """
    A general purpose coordinate allocator on a matplotlib axis.

    This is an (hopefully) very generic coordinate allocator that takes into account
    multiple group key levels, stack key levels with configurable group alignment.

    Example:
    ... codeblock::

    """
    def __init__(self, ax: Axes, axis: int, group_keys: list, stack_keys: list, group_width: float, group_align: str,
                 group_order: str):
        """
        - ax: The axes to operate on
        - axis: 0 -> X axis, 1 -> Y axis
        - group_keys: A separate coordinate offset is assigned to each group.
        - stack_keys: A separate base coordinate is assigned to each stack in each group.
        - group_width: The fraction (0, 1) of the available space between two seed coordinates to use
        for drawing elements. E.g. if group_width = 0.6, 40% of the space is considered padding between
        groups.
        - group_align: Where to align the coordinate groups relative to the seed coordinates.
        Allowed values are "center" and "left". When align=center the seed coordinate is aligned to the
        center of the group. When align=left the seed coordinate is aligned to the left of the group
        """
        self.ax = ax
        self.axis = axis
        self.group_keys = group_keys
        self.stack_keys = stack_keys
        self.group_width = group_width
        self.group_align = group_align
        self.group_order = group_order

        assert self.axis == 0 or self.axis == 1
        assert self.group_width > 0 and self.group_width <= 1
        assert self.group_align in ["center", "left"]
        assert self.group_order in ["sequential", "interleaved"]

    def _fetch_col(self, df, name):
        df = df.reset_index().set_index(df.index, drop=False)
        return df[name]

    def _iter_group_slices(self, left_cols, right_cols, ngroups, naxes):
        """
        Generate the position and span width matrix column slices associated
        to each group of items to plot. In the simple case, each group is associated to an offset
        for each X point. If multiple columns are present there will be more than 1 matrix colum
        that will be flattened when assigning X position data.
        """
        if self.group_order == "sequential":
            col_set_size = ngroups
            col_index = 0
            for i in range(len(left_cols)):
                group_slice = slice(col_index * col_set_size, (col_index + 1) * col_set_size)
                yield (group_slice, i, "l")
                col_index += 1
            for i in range(len(right_cols)):
                group_slice = slice(col_index * col_set_size, (col_index + 1) * col_set_size)
                yield (group_slice, i, "r")
                col_index += 1
        elif self.group_order == "interleaved":
            n_ycols = max(len(left_cols), len(right_cols))
            col_index = 0
            for i in range(n_ycols):
                if i < len(left_cols):
                    group_slice = slice(col_index, ngroups * naxes + 1, naxes)
                    yield (group_slice, i, "l")
                    col_index += 1
                if i < len(right_cols):
                    group_slice = slice(col_index, ngroups * naxes + 1, naxes)
                    yield (group_slice, i, "r")
                    col_index += 1
        else:
            assert False, "Not reached"

    def compute_coords(self,
                       df: pd.DataFrame,
                       x: str,
                       left_cols: list,
                       right_cols: list,
                       prefix: str = "__coord_gen") -> pd.DataFrame:
        """
        Compute the group center x for the axis and then
        the x for each bar group.
        Returns the center x vector and the modified view dataframe.

        df: The dataframe to operate on
        x: the column to use for the coordinate seeds
        left_cols: columns to allocate to the left axis
        right_cols: columns to allocate to the right axis
        prefix: generated columns prefix

        Note: this assumes that the input dataframe is aligned on the
        group_keys and stack_keys levels.
        Operate on the assumption that waterver sorting was applied to the input
        data, it is reflected on the axis ordering.

        This will generate the following columns in the output dataframe ('<prefix>_<column_name>_<side><index>').
        - <side> can be 'l'/'r', it identifies whether the column is on the left or right column list.
        - <index> is the index of the corresponding column name in left_cols or right_cols.
        e.g. 'x_l0' contains the coordinates associated to df[left_cols[0]].
        - <column_name> can be:
          + x: the compute coordinate
          + width: the width of the space allocated to the hypotetical rectangle for the column, this is
          the width of each bar in a bar plot, but is useful for other things as well.
          + base: the base value for stacks. This is only relevant for stack grouping and is useful for
          stacked bar plots.
          + gstart: the start position of a group (repeated for each member of the group), will include padding.
          + gend: the end position of a group (repeated for each member of the group), will include padding.
        """
        assert df.index.is_unique, "Require unique index"
        if not isinstance(left_cols, list):
            raise TypeError(f"left_cols must be list, not {type(left_cols)}")
        if not isinstance(right_cols, list):
            raise TypeError(f"right_cols must be list, not {type(right_cols)}")
        # Only support right columns if left columns are present
        # This can be easily extended by unsupported for now
        assert (left_cols and right_cols) or left_cols

        # Setup parameters
        if left_cols and right_cols:
            naxes = len(left_cols) + len(right_cols)
            assert naxes >= 2
        else:
            naxes = len(left_cols)

        if self.group_keys:
            ngroups = len(df.groupby(self.group_keys))
        else:
            ngroups = 1

        if self.stack_keys:
            nstacks = len(df.groupby(self.stack_keys))
        else:
            nstacks = 1
        assert naxes > 0 and ngroups >= 1 and nstacks >= 1

        # Transformed coordinates will be in the Figure space
        # variables holding transformed coordinates are prefixed with f_ indicating
        # figure space.
        t = self.ax.transData
        t_inv = t.inverted()
        to_f_coords = ft.partial(transform_1d_helper, t, self.axis)
        to_d_coords = ft.partial(transform_1d_helper, t_inv, self.axis)

        # Find the space between each x tick, we scale the bars to fit into the interval
        # Assume that each x in xcol is the center of the bar group.
        # Since the x axis may be subject to a transformation, we need to take that into
        # account when computing widths so that the bars have uniform size.
        # 1. find the distance between each x value
        # 2. transform each width into axis coordinates
        # 3. split the intervals to fit the bars
        # 4. transform back the offsets
        # transform() requires an Nx2 array, we ignore the other axis here
        xcol = sorted(self._fetch_col(df, x).unique())
        assert len(xcol) > 0, "No seed coordinates for data points"
        f_xcol = to_f_coords(xcol)
        if len(f_xcol) > 1:
            f_space = np.diff(f_xcol)
            f_space = np.append(f_space, f_space[-1])
        else:
            f_space = np.array([1])
        assert f_space.shape == f_xcol.shape

        # Allocate artist width, this is relevant for non-linear scales
        f_width = (f_space * self.group_width) / (ngroups * naxes)
        # relative offsets in width units, starting from the related x position
        # if centering, this will need a translation
        rel_offsets = np.arange(0, (ngroups * naxes))
        # generate transformed offset matrix via column * row vector multiplication
        # Note that half of the groups are for the right axis if it exists
        f_width_vector = f_width.reshape(len(f_width), 1)
        rel_offsets_vector = rel_offsets.reshape(1, len(rel_offsets))
        # the row is indexed by x, the colum is indexed by group/column
        f_offset_matrix = f_width_vector * rel_offsets_vector
        assert f_offset_matrix.shape == (len(f_xcol), ngroups * naxes)

        # 1. Apply the offsets to the transformed x column.
        # 2. Shift the positions to center the groups at the given x column.
        # Assuming that we have ngroups * naxes = 4 and 4 points in xcol,
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
        f_position_matrix = f_offset_matrix + f_xcol[:, np.newaxis]
        if self.group_align == "center":
            align_offset = f_width / 2 - (f_space * self.group_width) / 2
            f_position_matrix += align_offset[:, np.newaxis]
            f_group_start = f_xcol - f_space / 2
            f_group_end = f_xcol + f_space / 2
        elif self.group_align == "left":
            raise NotImplementedError("TODO")
        # compute widths allocated to each group/column, these are taken from the position matrix
        # that gives the center of each span.
        f_span_start = f_position_matrix - f_width[:, np.newaxis] / 2
        f_span_end = f_position_matrix + f_width[:, np.newaxis] / 2
        assert f_span_start.shape == f_position_matrix.shape
        assert f_span_end.shape == f_position_matrix.shape

        # transform back the coordinates
        # XXX we have to take into account the axis (0, 0) for the position matrix, maybe?
        position_matrix = np.apply_along_axis(to_d_coords, 1, f_position_matrix)
        assert position_matrix.shape == f_position_matrix.shape
        span_start = np.apply_along_axis(to_d_coords, 1, f_span_start)
        assert span_start.shape == f_span_start.shape
        span_end = np.apply_along_axis(to_d_coords, 1, f_span_end)
        assert span_end.shape == f_span_end.shape
        span_width = span_end - span_start

        group_start = to_d_coords(f_group_start)
        group_end = to_d_coords(f_group_end)

        # Now broadcast the coordinates back into the dataframe.
        # - Positions are repeated for each stack cross section
        # - Span widths are repeated for each stack cross section
        # - Stack bases are assigned for each group cross section
        # Note: we need to flatten the matrixes in column-major order.
        if self.stack_keys:
            non_stack_idx = df.index.names.difference(self.stack_keys)
            stack_groups = df.groupby(non_stack_idx)
        # Care must be taken to preserve the X-data mapping. The dataframe may not be
        # sorted by X value yet.
        sort_by = self.group_keys + self.stack_keys + [x]
        df = df.sort_values(sort_by, ascending=True)

        for group_slice, col, side in self._iter_group_slices(left_cols, right_cols, ngroups, naxes):
            col_name = left_cols[col] if side == "l" else right_cols[col]
            x_group = position_matrix[:, group_slice]
            x_group_flat = x_group.ravel(order="F")
            df[f"{prefix}_x_{side}{col}"] = np.tile(x_group_flat, nstacks)
            span_width_group = span_width[:, group_slice]
            span_width_flat = span_width_group.ravel(order="F")
            df[f"{prefix}_width_{side}{col}"] = np.tile(span_width_flat, nstacks)
            if self.stack_keys:
                df[f"{prefix}_base_{side}{col}"] = stack_groups[col_name].apply(
                    lambda stack_slice: stack_slice.cumsum().shift(fill_value=0))
            else:
                df[f"{prefix}_base_{side}{col}"] = 0
        df[f"{prefix}_gstart"] = np.tile(group_start, ngroups * nstacks)
        df[f"{prefix}_gend"] = np.tile(group_end, ngroups * nstacks)
        return df


class PlotChain:
    """
    Base class that abstracts plotting of dataframes with a functional interface.
    """
    def __init__(df: pd.DataFrame | pl.DataFrame, ):
        """
        """
