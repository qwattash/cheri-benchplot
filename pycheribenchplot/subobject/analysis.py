from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import seaborn.objects as so
from sqlalchemy import func, select

from ..core.analysis import AnalysisTask
from ..core.artefact import SQLTarget, ValueTarget
from ..core.plot import PlotTarget, PlotTask, new_figure
from ..core.task import dependency, output
from .imprecise import ExtractImpreciseSubobject
from .model import MemberBounds, StructMember, StructType


class StructLayoutAnalysisMixin:
    """
    Prepare the SQL database data sources for each datagen context.

    This is a base task to conveniently fetch aggregated data.
    The dataset is annotated by the platform g_uuid, so it is possible
    to observe the difference between the behaviour for Morello and
    RISC-V variants.
    """
    def _get_datagen_tasks(self) -> dict["uuid.UUID", SQLTarget]:
        targets = {}
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(ExtractImpreciseSubobject)
            targets[b.uuid] = task.struct_layout_db
        return targets


class ImpreciseMembersPlot(PlotTask, StructLayoutAnalysisMixin):
    """
    Produce a plot that shows the amount of bytes by which a sub-object is imprecise.

    This reports the number of extra padding in the base and top of the sub-object
    capability.
    """
    public = True
    task_namespace = "subobject"
    task_name = "imprecise-subobject-plot-v2"

    @output
    def imprecise_fields_plot(self):
        """Target for the plot showing the padding of imprecise fields"""
        return PlotTarget(self, "imprecision")

    @output
    def imprecise_fields_size_plot(self):
        """Target for the plot showing XXX"""
        return PlotTarget(self, "size")

    @output
    def imprecise_fields_size_hist(self):
        """Target for the plot showing the distribution of the representability padding sizes"""
        return PlotTarget(self, "size-hist")

    def load_one_dataset(self, gen_task: ExtractImpreciseSubobject) -> pl.DataFrame:
        # The requested top for the member, this rounds up the size to the next byte
        # boundary for bit fields.
        req_top = (MemberBounds.offset + StructMember.size + func.min(func.coalesce(StructMember.bit_size, 0), 1))
        imprecise_members = (select(
            StructType.file, StructType.line,
            StructType.name, MemberBounds.name.label("member_name"), MemberBounds.offset, StructMember.size,
            req_top.label("req_top"), MemberBounds.base, MemberBounds.top).join(MemberBounds.member_entry).join(
                MemberBounds.owner_entry).where((MemberBounds.base < MemberBounds.offset)
                                                | (MemberBounds.top > MemberBounds.offset + StructMember.size +
                                                   func.coalesce(func.min(StructMember.bit_size, 0), 1))))
        df = pl.read_database(imprecise_members, gen_task.struct_layout_db.sql_engine)
        return df

    def load_imprecise_subobjects(self) -> pl.DataFrame:
        loaded = None
        for gen_task in self._get_datagen_tasks():
            df = self.load_one_dataset(gen_task)
            df = gen_task.add_dataset_metadata(df)
            if loaded is not None:
                loaded.vstack(df, in_place=True)
            else:
                loaded = df
        assert loaded is not None, "No input data"
        # Now that we have everything, compute the helper columns for padding
        loaded = loaded.with_columns(padding_base=pl.col("offset") - pl.col("base"),
                                     padding_top=pl.col("top") - pl.col("req_top"))
        # Sanity checks
        assert (loaded["padding_base"] >= 0).all()
        assert (loaded["padding_top"] >= 0).all()
        return loaded

    def run_plot(self):
        df = self.load_imprecise_subobjects()
        sns.set_theme()

        show_df = df.melt(id_vars=df.columns,
                          value_vars=["padding_base", "padding_top"],
                          variable_name="side",
                          value_name="value")
        show_df = show_df.with_columns(
            (pl.col("side") + " " + pl.col("dataset_gid").map_elements(self.g_uuid_to_label)).alias("legend"),
            pl.col("member_name").alias("label"))

        with new_figure(self.imprecise_fields_plot.paths(), figsize=(10, 50)) as fig:
            ax = fig.subplots()
            (so.Plot(show_df, y="label", x="value", color="legend").on(ax).add(so.Bar(),
                                                                               so.Dodge(by=["dataset_gid", "side"]),
                                                                               dataset_gid=show_df["dataset_gid"],
                                                                               side=show_df["side"],
                                                                               orient="y").plot())

            # self.adjust_legend_on_top(fig, ax)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Capability imprecision (bytes)")
            ax.set_ylabel("Sub-object field")
            plt.setp(ax.get_yticklabels(), ha="right", va="center", fontsize="xx-small")
