import polars as pl
import polars.selectors as cs
from sqlalchemy import func, select

from ..core.analysis import AnalysisTask
from ..core.artefact import HTMLTemplateTarget
from ..core.task import dependency, output
from .model import LayoutMember, TypeLayout
from .scan_dwarf import LoadStructLayouts


class TargetGroupHelper:
    def __init__(self, group_id, desc, struct_list):
        self.id = group_id
        self.desc = desc
        self.struct_list = struct_list


class StructDescHelper:
    def __init__(self, struct_id, struct_data):
        self.id = struct_id
        self.df = struct_data

        # Generate helper columns
        self.df = self.df.with_columns(pl.col("member_name").str.strip_prefix(pl.col("name") + "::"))

        self.df = self.df.with_columns(
            (pl.col("member_name").str.split("::").list.len() - 1).alias("_record_helper_depth"),
            (~pl.col("array_items").is_null()).alias("_record_helper_is_array"))

    @property
    def name(self):
        struct_name = self.df["name"].unique()
        assert len(struct_name) == 1, "Non-unique struct name"
        name = struct_name[0]
        if self.df["is_union"].all():
            name = "union " + name
        else:
            name = "struct " + name
        return name

    @property
    def location(self):
        struct_file = self.df["file"].unique()
        struct_line = self.df["line"].unique()
        assert len(struct_file) == 1, "Non-unique struct file"
        assert len(struct_line) == 1, "Non-unique struct line"
        return f"{struct_file[0]}:{struct_line[0]}"

    @property
    def size(self):
        struct_size = self.df["total_size"].unique()
        assert len(struct_size) == 1, "Non-unique struct size"
        return struct_size[0]

    @property
    def has_imprecise_array(self):
        return not self.df.filter(pl.col("is_imprecise") & ~pl.col("array_items").is_null()).is_empty()

    @property
    def has_imprecise_ptr_access(self):
        return self.df["_alias_pointer_member"].any()

    @property
    def members(self):
        return self.df.sort(by=["member_id", "byte_offset", "bit_offset"])


class ImpreciseSubobjectLayouts(AnalysisTask):
    """
    Render an HTML to interactively browse the structure layouts with annotated
    imprecise sub-objects.

    This marks structures and members with imprecise array capabilities and
    capablities that alias other pointer fields.
    """
    public = True
    task_namespace = "subobject"
    task_name = "imprecise-layouts"

    @dependency
    def layouts(self):
        # yapf: disable
        has_imprecise = (
            select(TypeLayout.id)
            .join(LayoutMember.owner_entry)
            .group_by(TypeLayout.id)
            .having(func.max(LayoutMember.is_imprecise) == 1)
        )
        q = (
            select(
                TypeLayout.id,
                TypeLayout.file,
                TypeLayout.line,
                TypeLayout.name,
                TypeLayout.is_union,
                TypeLayout.size.label("total_size"),
                LayoutMember.id.label("member_id"),
                LayoutMember.name.label("member_name"),
                LayoutMember.type_name.label("member_type"),
                LayoutMember.byte_size,
                LayoutMember.bit_size,
                LayoutMember.byte_offset,
                LayoutMember.bit_offset,
                LayoutMember.array_items,
                LayoutMember.base,
                LayoutMember.top,
                LayoutMember.is_pointer,
                LayoutMember.is_function,
                LayoutMember.is_union.label("is_member_union"),
                LayoutMember.is_imprecise,
            )
            .join(LayoutMember.owner_entry)
            .where(TypeLayout.id.in_(has_imprecise))
        )
        # yapf: enable
        return LoadStructLayouts(self.session, self.analysis_config, query=q)

    def outputs(self):
        yield from super().outputs()
        scenarios = self.layouts.struct_layouts.get()["scenario"].unique()
        for scenario in scenarios:
            yield (scenario, HTMLTemplateTarget(self, "html/imprecise-subobject-layout.html.jinja", scenario))

    def run(self):
        """
        Render the HTML template.

        The dataframe is grouped recursively to produce nested datasets that are
        convenient to use for the template. Synthetic columns generated explicitly for
        the template rendering are prefixed by 'tmpl_'.
        """
        df = self.layouts.struct_layouts.get()

        # Columns that uniquely identify a structure layout within a dataset parameterization
        STRUCT_LAYOUT_ID_COLS = ["dataset_id", "name", "file", "line", "total_size"]
        # Columns that uniquely identify a flattened layout member within
        # a dataset parameterization
        STRUCT_MEMBER_ID_COLS = STRUCT_LAYOUT_ID_COLS + ["member_name"]
        IMPRECISE_MEMBER_ID_COLS = STRUCT_LAYOUT_ID_COLS + ["member_name__r"]

        # Compute overlap group IDs for each imprecise member.
        # This is used to color overlapping elements depending onthe capability
        # base and top of imprecise members

        imprecise = df.filter(is_imprecise=True).with_row_index("_alias_color")
        tmp_df = df.join(imprecise, on=STRUCT_MEMBER_ID_COLS, how="left", suffix="__r").select(~cs.ends_with("__r"))

        # For each structure layout, join each member with the imprecise members
        # within that layout.
        tmp_df = tmp_df.join(imprecise, suffix="__r", on=STRUCT_LAYOUT_ID_COLS, how="left")

        # We then mark the members that fall within the base/top of imprecise members.
        top = pl.col("byte_offset") + pl.col("byte_size") + ((pl.col("bit_offset") + pl.col("bit_size")) / 8).ceil()
        have_overlap = ((pl.col("byte_offset") <= pl.col("top__r")) & (pl.col("base__r") <= top) &
                        # Do not consider overlapping members that are known children of the imprecise member
                        ~pl.col("member_name").str.starts_with(pl.col("member_name__r") + "::") &
                        # Do not consider overlapping members that are known parents of the imprecise member
                        ~pl.col("member_name__r").str.starts_with(pl.col("member_name") + "::") &
                        # Can't overlap with yourself
                        (pl.col("member_name__r") != pl.col("member_name")))
        tmp_df = tmp_df.with_columns(
            pl.when(have_overlap).then(pl.col("_alias_color__r")).otherwise(pl.lit(None)).alias("_aliased_by"))

        # Collect all imprecise members that alias at least a pointer/function pointer
        imprecise = tmp_df.filter(~pl.col("_aliased_by").is_null()).group_by(IMPRECISE_MEMBER_ID_COLS).agg(
            pl.col("is_pointer").any().alias("_alias_pointer_member"))

        # Finally, coalesce _aliased_by color lists, keeping all left columns unchanged
        tmp_df = tmp_df.group_by(STRUCT_MEMBER_ID_COLS).agg((~cs.ends_with("__r") & cs.exclude("_aliased_by")).first(),
                                                            pl.col("_aliased_by"))

        assert len(tmp_df) == len(df)
        assert set(tmp_df.columns) == set(df.columns + ["_aliased_by", "_alias_color"])
        df = tmp_df.join(imprecise,
                         left_on=STRUCT_MEMBER_ID_COLS,
                         right_on=IMPRECISE_MEMBER_ID_COLS,
                         suffix="__r",
                         how="left").select(~cs.ends_with("__r"))

        def make_struct_list(df_slice) -> list[StructDescHelper]:
            ## XXX sort by field index as well?
            struct_groups = df_slice.sort(by=["name", "file", "line"]).group_by("id", maintain_order=True)
            desc_list = []
            for (struct_id, ), struct_members in struct_groups:
                desc_list.append(StructDescHelper(struct_id, struct_members))
            return desc_list

        for (scenario, ), df_slice in df.group_by("scenario"):
            dslice = df_slice.with_columns((pl.lit("target=") + pl.col("target") + pl.lit(" variant=") +
                                            pl.col("variant") + " runtime=" + pl.col("runtime")).alias("_block"))
            blocks = dslice.sort(by="_block").group_by("_block", maintain_order=True)
            layout_data = []
            for index, (group_keys, data) in enumerate(blocks):
                layout_data.append(TargetGroupHelper(index, group_keys[0], make_struct_list(data)))

            self.output_map[scenario].render(layout_data=layout_data)
