import polars as pl

from ..core.analysis import AnalysisTask
from ..core.artefact import HTMLTemplateTarget
from ..core.task import dependency, output
from .scan_dwarf import AnnotateImpreciseSubobjectLayouts


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
        return AnnotateImpreciseSubobjectLayouts(self.session, self.analysis_config)

    def outputs(self):
        yield from super().outputs()
        scenarios = self.layouts.imprecise_layouts.get()["scenario"].unique()
        for scenario in scenarios:
            yield (scenario, HTMLTemplateTarget(self, "html/imprecise-subobject-layout.html.jinja", scenario))

    def run(self):
        """
        Render the HTML template.

        Produce a plot for each scenario, assuming that the scenario keys a different set dwarf sources.
        """
        df = self.layouts.imprecise_layouts.get()

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
