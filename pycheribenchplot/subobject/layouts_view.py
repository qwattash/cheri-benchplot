from dataclasses import dataclass

import polars as pl

from ..core.analysis import SliceAnalysisTask
from ..core.artefact import HTMLTemplateTarget
from ..core.config import Config, config_field
from ..core.task import dependency, output
from .scan_dwarf import AnnotateImpreciseSubobjectLayouts


class BlockGroupHelper:
    def __init__(self, group_id, df):
        self.id = group_id
        self._df = df

    @property
    def desc(self):
        """
        Human readable description of the free parameter axes for this block group
        """
        assert self._df.n_unique("_block") == 1, "Invalid block group"
        return self._df["_block"].first()

    @property
    def struct_list(self):
        """
        Generate a list of data structure descriptors for the block.
        The data strucutes will have a stable order across blocks.
        """
        ## XXX sort by field index as well?
        struct_groups = self._df.sort(by=["name", "file", "line"]).group_by("id", maintain_order=True)
        for (struct_id, ), struct_members in struct_groups:
            yield StructDescHelper(struct_id, struct_members)


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


@dataclass
class ImpreciseSubobjectConfig(Config):
    """
    Configure imprecise subobject report rendering.
    """
    group_name: list[str] | None = config_field(
        None, desc="List of parameter axes to include in the subobject group description")


class ImpreciseSubobjectLayouts(SliceAnalysisTask):
    """
    Render an HTML to interactively browse the structure layouts with annotated
    imprecise sub-objects.

    This marks structures and members with imprecise array capabilities and
    capablities that alias other pointer fields.
    """
    public = True
    task_namespace = "subobject"
    task_name = "imprecise-layouts"
    task_config_class = ImpreciseSubobjectConfig

    # Allow only one-dimensional data in the slice parameterisation.
    max_degrees_of_freedom = 1

    @dependency
    def layouts(self):
        return AnnotateImpreciseSubobjectLayouts(self.session, self.slice_info, self.analysis_config)

    @output
    def html_output(self):
        return HTMLTemplateTarget(self, "imprecise-subobject-layout.html.jinja")

    def run(self):
        """
        Render the HTML template.

        Produce a plot for each scenario, assuming that the scenario keys a different set dwarf sources.
        """
        df = self.layouts.imprecise_layouts.get()

        if self.config.group_name is None:
            self.config.group_name = self.slice_info.free_axes
        block_desc = [pl.lit(f"{n}=") + pl.col(n) for n in self.config.group_name]
        df = df.with_columns(pl.concat_str(block_desc, separator=" ").alias("_block"))

        if self.slice_info.free_axes:
            blocks = df.group_by(self.slice_info.free_axes)
        else:
            blocks = [(None, df)]

        layout_data = []
        for index, (_, block_df) in enumerate(blocks):
            layout_data.append(BlockGroupHelper(index, block_df))
        self.html_output.render(layout_data=layout_data)
