import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import seaborn.objects as so
from sqlalchemy import func, select

from ..core.artefact import Target
from ..core.plot import PlotTarget, PlotTask, new_figure, new_facet
from ..core.task import output
from .imprecise import ExtractImpreciseSubobject
from .layouts_view import StructLayoutAnalysisMixin
from .model import MemberBounds, StructMember, StructMemberFlags, StructType


class ImpreciseMembersPlot(PlotTask, StructLayoutAnalysisMixin):
    """
    Produce a plot that shows the amount of bytes by which a sub-object is imprecise.

    This reports the number of extra padding in the base and top of the sub-object
    capability.
    """
    public = True
    task_namespace = "subobject"
    task_name = "imprecise-subobject-plot"

    rc_params = {
        "axes.labelsize": "medium",
        "font.size": 9,
        "xtick.labelsize": 9
    }

    @output
    def imprecise_fields_size_hist(self):
        """Target for the plot showing the padding of imprecise fields"""
        return PlotTarget(self, "size-hist")

    @output
    def imprecise_fields_padding(self):
        """
        Target for the plot showing the distribution of the representability
        padding sizes.
        """
        return PlotTarget(self, "pad")

    @output
    def imprecise_fields_padding_by_size(self):
        """
        Target for the plot showing the distribution of the representability
        padding sizes separated into base and top rounding.
        """
        return PlotTarget(self, "pad-by-size")

    @output
    def imprecise_fields_stats(self):
        """
        Target for a csv file that summarizes the imprecise fields and where they are from.
        """
        return Target(self, "stats", ext="csv")

    def load_one_dataset(self, gen_task: ExtractImpreciseSubobject) -> pl.DataFrame:
        # The requested top for the member, this rounds up the size to the next byte
        # boundary for bit fields.
        req_top = (MemberBounds.offset + StructMember.size + func.min(func.coalesce(StructMember.bit_size, 0), 1))
        # yapf: disable
        imprecise_members = (
            select(
                StructType.file, StructType.line,
                StructType.name, MemberBounds.name.label("member_name"),
                MemberBounds.offset, StructMember.size,
                req_top.label("req_top"), MemberBounds.base, MemberBounds.top
            )
            .join(MemberBounds.member_entry)
            .join(MemberBounds.owner_entry)
            .where(
                (MemberBounds.base < MemberBounds.offset) |
                (MemberBounds.top > MemberBounds.offset + StructMember.size +
                 func.coalesce(func.min(StructMember.bit_size, 0), 1) + 1)
            )
        )
        # yapf: enable
        df = pl.read_database(imprecise_members, gen_task.struct_layout_db.sql_engine)
        return df

    def load_imprecise_subobjects(self) -> pl.DataFrame:
        loaded = self.load_layouts()
        # Now that we have everything, compute the helper columns for padding
        loaded = loaded.with_columns(
            padding_base=pl.col("offset") - pl.col("base"),
            padding_top=pl.col("top") - pl.col("req_top")
        )
        loaded = loaded.with_columns(
            padding_total=pl.col("padding_base") + pl.col("padding_top")
        )
        # Sanity checks
        assert (loaded["padding_base"] >= 0).all()
        assert (loaded["padding_top"] >= 0).all()
        assert ((loaded["padding_base"] != 0) | (loaded["padding_top"] != 0)).all()
        return loaded

    def run_plot(self):
        df = self.load_imprecise_subobjects()
        sns.set_theme()

        df = df.with_columns(pl.col("dataset_gid").map_elements(self.g_uuid_to_label).alias("target"))

        stat_cols = ["target", "member_name", "size", "padding_base", "padding_top", "padding_total"]
        df.select(stat_cols).sort("member_name", "size").write_csv(self.imprecise_fields_stats.single_path())

        with new_figure(self.imprecise_fields_size_hist.paths()) as fig:
            ax = fig.subplots()
            sns.histplot(df, x="size", hue="target", log_scale=2, ax=ax)

            ax.set_xlabel("Requested sub-object size (bytes)")
            ax.set_ylabel("# of imprecise fields")

        with new_figure(self.imprecise_fields_padding.paths()) as fig:
            ax = fig.subplots()
            sns.histplot(df, x="padding_total", hue="target", multiple="dodge",
                         log_scale=2, ax=ax)

            ax.set_xlabel("Capability imprecision (bytes)")
            ax.set_ylabel("# of imprecise fields")

        show_df = df.melt(id_vars=df.columns,
                          value_vars=["padding_base", "padding_top"],
                          variable_name="side",
                          value_name="value")
        # yapf: disable
        show_df = show_df.with_columns(
            pl.col("side").alias("legend"),
            pl.col("member_name").alias("label"),
            (pl.col("req_top") - pl.col("offset")).alias("req_size"))
        # yapf: enable

        with new_facet(self.imprecise_fields_padding_by_size.paths(), show_df,
                       col="target", col_wrap=2, sharey=False) as facet:
            facet.map_dataframe(sns.histplot, "value", hue="legend",
                                element="step", log_scale=2)
            facet.add_legend()
            facet.set_axis_labels(x_var="Capability imprecision (bytes)",
                                  y_var="# of imprecise fields")


class LLVMSubobjectSizeDistribution(PlotTask, StructLayoutAnalysisMixin):
    """
    Generate plots showing the distribution of subobject bounds sizes.

    This was originally designed to work with compiler diagnostics
    we should be able to adapt it to use both diagnostics and static data.
    """
    public = True
    task_namespace = "subobject"
    task_name = "subobject-size-distribution-plot"

    @output
    def size_distribution(self):
        return PlotTarget(self, "size-distribution-kern")

    def load_one_dataset(self, gen_task: ExtractImpreciseSubobject) -> pl.DataFrame:
        # The requested top for the member, this rounds up the size to the next byte
        # boundary for bit fields.
        req_top = (MemberBounds.offset + StructMember.size + func.min(func.coalesce(StructMember.bit_size, 0), 1))
        # yapf: disable
        all_members = (
            select(
                StructType.file, StructType.line,
                StructType.name, MemberBounds.name.label("member_name"),
                MemberBounds.offset, StructMember.size,
                req_top.label("req_top"), MemberBounds.base, MemberBounds.top
            )
            .join(MemberBounds.member_entry)
            .join(MemberBounds.owner_entry)
            .where(
                (MemberBounds.base < MemberBounds.offset) |
                (MemberBounds.top > MemberBounds.offset + StructMember.size +
                 func.coalesce(func.min(StructMember.bit_size, 0), 1))
            )
        )
        # yapf: enable
        df = pl.read_database(imprecise_members, gen_task.struct_layout_db.sql_engine)
        return df

    def _plot_size_distribution(self, df: pl.DataFrame, target: PlotTarget):
        """
        Helper to plot the distribution of subobject bounds sizes
        """
        # Determine buckets we are going to use
        min_size = max(df["size"].min(), 1)
        max_size = max(df["size"].max(), 1)
        log_buckets = range(int(np.log2(min_size)), int(np.log2(max_size)) + 1)
        buckets = [2**i for i in log_buckets]

        with new_figure(target.paths()) as fig:
            ax = fig.subplots()
            sns.histplot(df, x="size", stat="count", bins=buckets, ax=ax)
            ax.set_yscale("log", base=10)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("size (bytes)")
            ax.set_ylabel("# of csetbounds")

    def run_plot(self):
        containers = self.data.all_layouts.get()
        ## XXX Port this
        # Note that it would be nice if we could also read stats
        # from the compiler-emitted setbounds and see how these distributions
        # compare.

        # # Filter only setbounds that are marked kind=subobject
        # data_df = df.loc[df["kind"] == SetboundsKind.SUBOBJECT.value]

        # sns.set_theme()

        # show_df = data_df.loc[data_df["src_module"] == "kernel"]
        # self._plot_size_distribution(show_df, self.size_distribution_kernel)

        # show_df = data_df.loc[data_df["src_module"] != "kernel"]
        # self._plot_size_distribution(show_df, self.size_distribution_modules)

        # self._plot_size_distribution(data_df, self.size_distribution_all)
