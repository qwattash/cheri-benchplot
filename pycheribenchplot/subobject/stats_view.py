from dataclasses import dataclass

import numpy as np
import polars as pl

from ..core.plot import PlotTarget, SlicePlotTask
from ..core.plot_util import DisplayGrid, DisplayGridConfig, grid_barplot
from ..core.task import dependency, output
from .scan_dwarf import AnnotateImpreciseSubobjectLayouts


@dataclass
class ImpreciseSizeDistPlotConfig(DisplayGridConfig):
    pass


class ImpreciseSizeDistPlot(SlicePlotTask):
    """
    Produce a plot that shows the distribution of subobject imprecision sizes.

    This reports the number of extra padding in the base and top of sub-object
    capabilities.
    """
    public = True
    task_namespace = "subobject"
    task_name = "imprecise-size-distribution"
    task_config_class = ImpreciseSizeDistPlotConfig

    rc_params = {"text.usetex": True}

    @dependency
    def layouts(self):
        return AnnotateImpreciseSubobjectLayouts(self.session, self.slice_info, self.analysis_config)

    @output
    def size_dist(self):
        return PlotTarget(self)

    def run_plot(self):
        """
        Generate the distribution plot.

        Config defaults:
         - tile the grid rows by scenario.
         - tile the grid columns by target.
        """
        df = self.layouts.imprecise_layouts.get()
        # XXX maybe add knob to narrow bounds as much as possible on bitfields
        # top = offset + ((bit_offset + bit_size) / 8).ceil()
        req_top = pl.col("byte_offset") + pl.col("byte_size")
        df = df.with_columns((pl.col("byte_offset") - pl.col("base")).alias("base_padding"),
                             (pl.col("top") - req_top).alias("top_padding"))
        assert (df["base_padding"] >= 0).all(), "Negative base padding!"
        assert (df["top_padding"] >= 0).all(), "Negative top padding!"
        # Compute padding size range for bins
        bin_max = max(df["base_padding"].max(), df["top_padding"].max())
        bin_pow = np.arange(0, np.log2(bin_max) + 1, dtype=int)
        bin_edges = [2**n for n in bin_pow]

        # Now produce the histograms for base and top padding and combine them into long form,
        # keyed on the `side` column.
        base_pad_hist = self.histogram(df, "base_padding", prefix="hist_",
                                       bins=bin_edges).with_columns(pl.lit("base").alias("side"))
        top_pad_hist = self.histogram(df, "top_padding", prefix="hist_",
                                      bins=bin_edges).with_columns(pl.lit("top").alias("side"))
        hist = pl.concat([base_pad_hist, top_pad_hist], how="vertical").cast({"hist_bin": pl.UInt64})

        grid_config = self.config.set_fixed(hue="side")
        # Build a default mapping for the hist_bin labels
        grid_config = grid_config.set_display_defaults(
            param_values={"hist_bin": {
                2**n: f"$2^{{{n}}}$"
                for n in bin_pow
            }})

        with DisplayGrid(self.size_dist, hist, grid_config) as grid:
            grid.map(grid_barplot, x="hist_bin", y="hist_count")
            grid.add_legend()


# class ImpreciseMembersPlot(PlotTask, StructLayoutAnalysisMixin):
#     """
#     Produce a plot that shows the amount of bytes by which a sub-object is imprecise.

#     This reports the number of extra padding in the base and top of the sub-object
#     capability.
#     """
#     public = True
#     task_namespace = "subobject"
#     task_name = "imprecise-subobject-plot"

#     rc_params = {
#         "axes.labelsize": "medium",
#         "font.size": 9,
#         "xtick.labelsize": 9
#     }

#     @output
#     def imprecise_fields_size_hist(self):
#         """Target for the plot showing the padding of imprecise fields"""
#         return PlotTarget(self, "size-hist")

#     @output
#     def imprecise_fields_padding(self):
#         """
#         Target for the plot showing the distribution of the representability
#         padding sizes.
#         """
#         return PlotTarget(self, "pad")

#     @output
#     def imprecise_fields_padding_by_size(self):
#         """
#         Target for the plot showing the distribution of the representability
#         padding sizes separated into base and top rounding.
#         """
#         return PlotTarget(self, "pad-by-size")

#     @output
#     def imprecise_fields_stats(self):
#         """
#         Target for a csv file that summarizes the imprecise fields and where they are from.
#         """
#         return Target(self, "stats", ext="csv")

#     def load_one_dataset(self, gen_task: ExtractImpreciseSubobject) -> pl.DataFrame:
#         # The requested top for the member, this rounds up the size to the next byte
#         # boundary for bit fields.
#         req_top = (MemberBounds.offset + StructMember.size + func.min(func.coalesce(StructMember.bit_size, 0), 1))
#         # yapf: disable
#         imprecise_members = (
#             select(
#                 StructType.file, StructType.line,
#                 StructType.name, MemberBounds.name.label("member_name"),
#                 MemberBounds.offset, StructMember.size,
#                 req_top.label("req_top"), MemberBounds.base, MemberBounds.top
#             )
#             .join(MemberBounds.member_entry)
#             .join(MemberBounds.owner_entry)
#             .where(
#                 (MemberBounds.base < MemberBounds.offset) |
#                 (MemberBounds.top > MemberBounds.offset + StructMember.size +
#                  func.coalesce(func.min(StructMember.bit_size, 0), 1) + 1)
#             )
#         )
#         # yapf: enable
#         df = pl.read_database(imprecise_members, gen_task.struct_layout_db.sql_engine)
#         return df

#     def load_imprecise_subobjects(self) -> pl.DataFrame:
#         loaded = self.load_layouts()
#         # Now that we have everything, compute the helper columns for padding
#         loaded = loaded.with_columns(
#             padding_base=pl.col("offset") - pl.col("base"),
#             padding_top=pl.col("top") - pl.col("req_top")
#         )
#         loaded = loaded.with_columns(
#             padding_total=pl.col("padding_base") + pl.col("padding_top")
#         )
#         # Sanity checks
#         assert (loaded["padding_base"] >= 0).all()
#         assert (loaded["padding_top"] >= 0).all()
#         assert ((loaded["padding_base"] != 0) | (loaded["padding_top"] != 0)).all()
#         return loaded

#     def run_plot(self):
#         df = self.load_imprecise_subobjects()
#         sns.set_theme()

#         df = df.with_columns(pl.col("dataset_gid").map_elements(self.g_uuid_to_label).alias("target"))

#         stat_cols = ["target", "member_name", "size", "padding_base", "padding_top", "padding_total"]
#         df.select(stat_cols).sort("member_name", "size").write_csv(self.imprecise_fields_stats.single_path())

#         with new_figure(self.imprecise_fields_size_hist.paths()) as fig:
#             ax = fig.subplots()
#             sns.histplot(df, x="size", hue="target", log_scale=2, ax=ax)

#             ax.set_xlabel("Requested sub-object size (bytes)")
#             ax.set_ylabel("# of imprecise fields")

#         with new_figure(self.imprecise_fields_padding.paths()) as fig:
#             ax = fig.subplots()
#             sns.histplot(df, x="padding_total", hue="target", multiple="dodge",
#                          log_scale=2, ax=ax)

#             ax.set_xlabel("Capability imprecision (bytes)")
#             ax.set_ylabel("# of imprecise fields")

#         show_df = df.melt(id_vars=df.columns,
#                           value_vars=["padding_base", "padding_top"],
#                           variable_name="side",
#                           value_name="value")
#         # yapf: disable
#         show_df = show_df.with_columns(
#             pl.col("side").alias("legend"),
#             pl.col("member_name").alias("label"),
#             (pl.col("req_top") - pl.col("offset")).alias("req_size"))
#         # yapf: enable

#         with new_facet(self.imprecise_fields_padding_by_size.paths(), show_df,
#                        col="target", col_wrap=2, sharey=False) as facet:
#             facet.map_dataframe(sns.histplot, "value", hue="legend",
#                                 element="step", log_scale=2)
#             facet.add_legend()
#             facet.set_axis_labels(x_var="Capability imprecision (bytes)",
#                                   y_var="# of imprecise fields")
