from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import seaborn.objects as so
from sqlalchemy import func, select
from sqlalchemy.orm import aliased

from ..core.analysis import AnalysisTask
from ..core.artefact import HTMLTemplateTarget, SQLTarget, ValueTarget
from ..core.plot import PlotTarget, PlotTask, new_figure
from ..core.task import dependency, output
from .imprecise import ExtractImpreciseSubobject
from .model import MemberBounds, StructMember, StructMemberFlags, StructType


class StructLayoutAnalysisMixin:
    """
    Prepare the SQL database data sources for each datagen context.

    This is a base task to conveniently fetch aggregated data.
    The dataset is annotated by the platform g_uuid, so it is possible
    to observe the difference between the behaviour for Morello and
    RISC-V variants.
    """
    def _get_datagen_tasks(self) -> dict["uuid.UUID", ExtractImpreciseSubobject]:
        targets = {}
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(ExtractImpreciseSubobject)
            targets[b.uuid] = task
        return targets

    def load_layouts(self) -> pl.DataFrame:
        """
        Load structure layouts containing imprecise sub-objects.
        """
        loaded = None
        for uuid, gen_task in self._get_datagen_tasks().items():
            df = self.load_one_dataset(gen_task)
            df = gen_task.add_dataset_metadata(df)
            if loaded is not None:
                loaded.vstack(df, in_place=True)
            else:
                loaded = df
        assert loaded is not None, "No input data"
        return loaded

    def load_one_dataset(self, gen_task: ExtractImpreciseSubobject) -> pl.DataFrame:
        raise NotImplementedError("Must override")


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
        loaded = self.load_layouts()
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
            pl.col("member_name").alias("label"), (pl.col("req_top") - pl.col("offset")).alias("req_size"))

        with new_figure(self.imprecise_fields_plot.paths(), figsize=(10, 50)) as fig:
            ax = fig.subplots()
            (so.Plot(show_df, y="label", x="value", color="legend").on(ax).add(so.Bar(),
                                                                               so.Dodge(by=["dataset_gid", "side"]),
                                                                               dataset_gid=show_df["dataset_gid"],
                                                                               side=show_df["side"],
                                                                               orient="y").plot())

            self.adjust_legend_on_top(fig, ax)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Capability imprecision (bytes)")
            ax.set_ylabel("Sub-object field")
            plt.setp(ax.get_yticklabels(), ha="right", va="center", fontsize="xx-small")

        with new_figure(self.imprecise_fields_size_hist.paths()) as fig:
            ax = fig.subplots()
            sns.histplot(show_df, x="req_size", hue="legend", log_scale=2, ax=ax, element="step")
            ax.set_xlabel("Sub-object requested size (bytes)")
            ax.set_ylabel("# of imprecise sub-objects")


class ImpreciseSubobjectLayouts(AnalysisTask, StructLayoutAnalysisMixin):
    """
    Render an HTML to interactively browse the structure layouts with annotated
    imprecise sub-objects.

    This marks structures and members with imprecise array capabilities and
    capablities that alias other pointer fields.
    """
    public = True
    task_namespace = "subobject"
    task_name = "imprecise-subobject-layouts-v2"

    @output
    def html_doc(self):
        return HTMLTemplateTarget(self, "imprecise-subobject-layout.html.jinja")

    def load_one_dataset(self, gen_task: ExtractImpreciseSubobject) -> pl.DataFrame:
        """
        Load structure layouts containing imprecise sub-objects.

        The dataframe contains the flattened layout of data structures that
        contain at least one imprecise sub-object member.
        """
        # The requested top for the member, this rounds up the size to the next byte
        # boundary for bit fields.
        # yapf: disable
        aliased_bounds_by = aliased(MemberBounds)
        aliased_by = (
            select(
                MemberBounds.id,
                func.aggregate_strings(aliased_bounds_by.id, ",").label("aliased_by")
            )
            .join(MemberBounds.aliased_by.of_type(aliased_bounds_by))
            .group_by(MemberBounds.id)
        ).subquery()

        # Is a member capability aliasing other pointer members?
        aliased_bounds_with = aliased(MemberBounds)
        any_ptr_flag = StructMemberFlags.IsPtr | StructMemberFlags.IsFnPtr
        aliasing_ptrs = (
            select(
                MemberBounds.id,
                func.min(func.count(aliased_bounds_with.id), 1).label("is_aliasing_ptrs"),
            )
            .join(MemberBounds.aliasing_with.of_type(aliased_bounds_with))
            .join(aliased_bounds_with.member_entry)
            .where(StructMember.flags.op("&")(any_ptr_flag.value) != 0)
            .group_by(MemberBounds.id)
        ).subquery()

        imprecise_layouts = (
            select(
                # Base columns
                StructType.id, StructType.file, StructType.line, StructType.name,
                StructType.size.label("total_size"),
                StructMember.bit_offset, StructMember.name.label("member_name"),
                StructMember.type_name, StructMember.size, StructMember.bit_size,
                MemberBounds.name.label("flat_name"), MemberBounds.offset,
                MemberBounds.is_imprecise,
                MemberBounds.mindex,
                # Synthetic columns from subqueries
                MemberBounds.id.label("flat_member_id"),
                aliased_by.c.aliased_by,
                aliasing_ptrs.c.is_aliasing_ptrs,
                func.min(StructMember.flags.op("&")(StructMemberFlags.IsArray.value), 1).label("is_array")
            )
            .join(MemberBounds.owner_entry)
            .join(MemberBounds.member_entry)
            .join(aliased_by, MemberBounds.id == aliased_by.c.id, isouter=True)
            .join(aliasing_ptrs, MemberBounds.id == aliasing_ptrs.c.id, isouter=True)
            .where(StructType.has_imprecise)
        )
        # yapf: enable
        schema_types = {
            "id": pl.UInt64,
            "file": pl.Utf8,
            "line": pl.UInt64,
            "name": pl.Utf8,
            "total_size": pl.UInt64,
            "bit_offset": pl.Int8,
            "member_name": pl.Utf8,
            "type_name": pl.Utf8,
            "size": pl.UInt64,
            "bit_size": pl.Int8,
            "flat_name": pl.Utf8,
            "offset": pl.UInt64,
            "is_imprecise": pl.Boolean,
            "mindex": pl.UInt64,
            "flat_member_id": pl.UInt64,
            "aliased_by": pl.Utf8,
            "is_aliasing_ptrs": pl.Boolean,
            "is_array": pl.Boolean
        }
        self.logger.info("Loading struct layout data for %s", gen_task)
        df = pl.read_database(imprecise_layouts, gen_task.struct_layout_db.sql_engine, schema_overrides=schema_types)
        return df

    def run(self):
        """
        Render the HTML template.

        The dataframe is grouped recursively to produce nested datasets that are
        convenient to use for the template. Synthetic columns generated explicitly for
        the template rendering are prefixed by 'tmpl_'.
        """
        layouts = self.load_layouts()
        assert layouts.select("dataset_id", "flat_member_id").is_duplicated().any() == False

        def gen_struct_members(df: pl.DataFrame):
            # Helper columns for the template
            # yapf: disable
            expr_bit_suffix = (
                pl.when(pl.col("bit_offset").is_not_null())
                .then(":+" + pl.col("bit_offset").cast(str))
                .otherwise(pl.lit(""))
            )
            expr_bit_size_suffix = (
                pl.when(pl.col("bit_size").is_not_null())
                .then(":+" + pl.col("bit_size").cast(str))
                .otherwise(pl.lit(""))
            )
            df = df.with_columns(
                (pl.col("offset").map_elements(lambda o: f"{o:#x}") + expr_bit_suffix).alias("tmpl_offset"),
                # (pl.col("offset").cast(str) + expr_bit_suffix).alias("tmpl_offset"),
                (pl.col("size").cast(str) + expr_bit_size_suffix).alias("tmpl_size"),
                pl.col("bit_size").fill_null(0),
                (pl.col("flat_name").str.split("::").list.len()).alias("tmpl_depth")
            )
            # yapf: enable
            return df.sort(pl.col("mindex"), descending=False)

        def gen_struct_layouts(group_name: str, df: pl.DataFrame):
            # Get layouts ordered by name
            layouts = df.select("name", "id").unique().sort("name")["id"]
            # For each layout, emit information and group cross section
            for struct_id in layouts:
                xs = df.filter(pl.col("id") == struct_id)
                # Per-layout information
                assert xs["file"].n_unique() == 1, \
                    f"Invalid layout with non-unique 'file' for {group_name}:{name}"
                assert xs["line"].n_unique() == 1, \
                    f"Invalid layout with non-unique 'line' for {group_name}:{name}"
                assert xs["total_size"].n_unique() == 1, \
                    f"Invalid layout with non-unique 'total_size' for {group_name}:{name}"
                name = xs["name"][0]
                info = {
                    "tmpl_name": name,
                    "tmpl_layout_id": group_name + "-" + name,
                    "tmpl_location": xs["file"][0] + ":" + str(xs["line"][0]),
                    "tmpl_layout_size": "{:#x}".format(xs["total_size"][0]),
                    "tmpl_has_imprecise_alias_pointer": xs["is_aliasing_ptrs"].any(),
                    "tmpl_has_imprecise_array": (xs["is_imprecise"] & xs["is_array"]).any()
                }
                yield (info, xs, gen_struct_members(xs))

        def gen_layout_groups():
            for (group_name, ), group in layouts.group_by(["dataset_gid"]):
                label = self.g_uuid_to_label(group_name)
                self.logger.debug("Prepare structures for %s", label)
                df = group.with_columns(pl.lit(label).alias("tmpl_group_name"))
                yield (df, gen_struct_layouts(group_name, df))

        self.html_doc.render(layout_groups=gen_layout_groups())
