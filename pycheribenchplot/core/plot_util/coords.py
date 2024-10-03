from dataclasses import dataclass

import numpy as np
import polars as pl

from ..config import Config, config_field


@dataclass
class CoordGenConfig(Config):
    shift_by: str | None = config_field(None, desc="Column used to compute the shift offset along the axis")
    stack_by: str | None = config_field(None,
                                        desc="Column used to compute the stack offset along the complementary axis")
    gap_ratio: float = config_field(
        0.9,
        desc=
        "The fraction (0, 1) of the available space between two seed coordinates to use for drawing elements. E.g. if gap_ratio = 0.6, 40% of the space is considered gap-space between groups."
    )
    pad_ratio: float = config_field(
        1.0,
        desc=
        "The fraction (0, 1) of the available space between two offset coordinates to use for drawing elements. E.g. if pad_ratio = 0.6, 40% of the space is considered pad-space between two elements at the same seed coordinate."
    )
    align: str = config_field(
        "center",
        desc=
        "Where to align the coordinate groups relative to the seed coordinates. Allowed values are 'center' and 'left'. When align=center the seed coordinate is aligned to the center of the group. When align=left the seed coordinate is aligned to the left of the group."
    )
    order: str = config_field("sequential", desc="TODO")

    def check_gap(self):
        pass

    def check_pad(self):
        pass

    def check_align(self):
        pass

    def check_order(self):
        pass


class CoordGenerator:
    def __init__(self, ax: "Axes", orient="x"):
        """
        Helper to create axis coordinates from a dataframe

        :param ax: The axes to operate on
        :param orient: Orientation of the plot. If orient='x', the generator will produce X-axis coordinats.
        If orient='y', the generator will produce Y-axis coordinates.
        """
        self.ax = ax
        self.orient = orient

        if orient != "x" and orient != "y":
            raise ValueError("Invalid orient value, must be either 'x' or 'y'")

    def compute_coordinates(self,
                            df: pl.DataFrame,
                            independent_var: str,
                            dependent_vars: list,
                            prefix: str = "__gen",
                            config: CoordGenConfig | None = None) -> pl.DataFrame:
        """
        Compute coordinates for a categorical axis.

        The seeds for coordinates are integers in [0..N], where N is the number of unique values
        of the `independent_var` column.
        Shift offsets are computed for each unique combination of `config.shift_by` column and `dependent_vars`.

        Note that this function maintains ordering of the values according to the input dataframe.

        The dataframe is modified so that there are new computed columns, prefixed by `prefix`, as follows:
        - coord: the computed independent variable coordinate
        - shift_<dependent_var>: the computed offset corresponding to the given dependent variable

        Note: this assumes that the input dataframe is aligned on the
        group_keys and stack_keys levels, and that it is sorted to the desired ordering.
        """
        if config is None:
            config = CoordGenConfig()
        # Axis corresponding to the selected orientation in numpy
        orient_axis = 0 if self.orient == "x" else 1

        seed_coord_groups = []
        # Note that we require the dataframe to be aligned on shift_by and stack_by columns.
        # This means that every shift_by and stack_by group must have the same number of elements.
        tmp_col = f"{prefix}_tmp"
        if config.shift_by:
            check = df.group_by(config.shift_by).len(name=tmp_col).select(pl.col(tmp_col) == pl.col(tmp_col).max())
            if not check[tmp_col].all():
                raise ValueError("The input dataframe is not aligned at the shift_by level")
            ngroups = len(check)
            seed_coord_groups.append(config.shift_by)
        else:
            ngroups = 1
        if config.stack_by:
            check = df.group_by(config.stack_by).len(name=tmp_col).select(pl.col(tmp_col) == pl.col(tmp_col).max())
            if not check[tmp_col].all():
                raise ValueError("The input dataframe is not aligned at the group_by level")
            nstacks = len(check)
            seed_coord_groups.append(config.stack_by)
        else:
            nstacks = 1
        nmetrics = len(dependent_vars)
        if nmetrics == 0:
            raise ValueError("At least one dependent_var column must be given")

        # Generate seed coordinates for the independent_var
        seed_col = f"{prefix}_coord"
        coord_step = 1.0
        seed_coord = pl.col(independent_var).cum_count()
        if seed_coord_groups:
            seed_coord = seed_coord.over(seed_coord_groups)
        workdf = df.with_columns(seed_coord.alias(seed_col).sub(1) * coord_step)

        # Determine the shift offset of every shift_by group and metric.
        assert len(dependent_vars) == 1, "XXX need to lift this limitation"
        offset_col = f"{prefix}_offset"
        if config.shift_by:
            base_offsets = workdf.select(pl.col(config.shift_by).unique(
                maintain_order=True)).with_columns(pl.col(config.shift_by).cum_count().alias(offset_col) - 1)
            workdf = workdf.join(base_offsets, on=config.shift_by)
        else:
            workdf = workdf.with_columns(pl.lit(0).alias(offset_col))

        # if config.order == "sequential":
        # ???
        # if config.order == "interleaved":
        # ???

        # Now we compute the actual offsets by computing the portion of space allocated
        # to each artist slot. Then we translate the offsets to center them on the seed coordinates.
        width = coord_step * config.gap_ratio / ngroups
        group_adjust = -(coord_step * config.gap_ratio) / 2
        align_adjust = width / 2 if config.align == "center" else 0
        workdf = workdf.with_columns(pl.col(offset_col) * width + group_adjust + align_adjust)

        # Set artist width available according to padding
        workdf = workdf.with_columns(pl.lit(width * config.pad_ratio).alias(f"{prefix}_width"))

        return workdf
