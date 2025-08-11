import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import polars as pl
import polars.selectors as cs
from sqlalchemy import func, select

from ..core.analysis import AnalysisTask
from ..core.artefact import DataFrameLoadTask, SQLTarget, ValueTarget
from ..core.config import Config, ConfigPath, config_field
from ..core.task import ExecutionTask, dependency, output
from ..core.util import SubprocessHelper, resolve_system_command
from .model import LayoutMember, TypeLayout


@dataclass
class PathMatchSpec(Config):
    """
    Path match specifier used to filter ELF objects that should be
    scanned for DWARF information.
    """
    path: ConfigPath = config_field(Config.REQUIRED, desc="Base path interpreted as the command `find {path}`")

    matcher: str | None = config_field(None, desc="Path match regex interpreted as a grep extended regular expression")


@dataclass
class ExtractImpreciseConfig(Config):
    """
    Configure the imprecise sub-object extractor.
    """
    dwarf_scraper: ConfigPath | None = config_field(
        None, desc="Optional path to the dwarf_scraper tool. Otherwise, assume that the tool is in $PATH")

    verbose_scraper: bool = config_field(False, desc="Enable verbose output from dwarf_scraper")

    dwarf_data_sources: list[PathMatchSpec] = config_field(
        Config.REQUIRED,
        desc="List of paths that are used to extract DWARF information. "
        "Note that multiple paths will be considered as belonging to the same 'object', "
        "if multiple objects are being compared, we should use parameterization. "
        "Relative paths will be interpreted as relative paths into a cheribuild rootfs.")

    src_prefix: str | None = config_field(None, desc="Optional path prefix to strip from source file paths")


class ExtractImpreciseSubobject(ExecutionTask):
    """
    Extract CheriBSD DWARF type information for aggregate data types.
    This will generate a database using the dwarf_scraper tool.
    The dwarf_scraper tool must be in $PATH or passed as a task configuration
    parameter.
    """
    public = True
    task_namespace = "subobject"
    task_name = "extract-imprecise"
    task_config_class = ExtractImpreciseConfig

    @output
    def struct_layout_db(self):
        return SQLTarget(self, "layout-db")

    def run(self):
        super().run()
        self.script.set_template("dwarf-scraper.sh.jinja")
        self.script.extend_context({
            "dws_config": self.config,
            # XXX this should be relative to the benchmark run dir
            "dws_database": self.struct_layout_db.single_path()
        })


class TypeLayoutLoader(DataFrameLoadTask):
    """
    Load flattened layouts from the database into a polars dataframe.
    """
    task_name = "type-layout-loader"

    def __init__(self, target, query):
        super().__init__(target)
        self._query = query

    def _load_one(self, path):
        df = pl.read_database(self._query, self.target.sql_engine)
        return df


class LoadStructLayouts(AnalysisTask):
    """
    Load imprecise subobject data into an unified dataframe
    for futher aggregation.
    """
    task_namespace = "subobject"
    task_name = "load-struct-layouts"

    def __init__(self, session, analysis_config, query=None, **kwargs):
        super().__init__(session, analysis_config, **kwargs)
        if query is None:
            query = select(TypeLayout, LayoutMember).join(LayoutMember.owner_entry)
        self._query = query

    @output
    def struct_layouts(self):
        return ValueTarget(self, "all-layouts")

    @dependency
    def dataset_layouts(self):
        for desc in self.session.all_benchmarks():
            task = desc.find_exec_task(ExtractImpreciseSubobject)
            yield TypeLayoutLoader(task.struct_layout_db, self._query)

    def run(self):
        super().run()
        data = []
        for loader in self.dataset_layouts:
            data.append(loader.df.get())
        df = pl.concat(data, how="vertical", rechunk=True)
        self.struct_layouts.assign(df)


class AnnotateImpreciseSubobjectLayouts(AnalysisTask):
    """
    Load imprecise subobject data and annotate imprecise subobject layouts.

    This slightly renames the loaded data and introduces the following columns:
    - _alias_color: A unique ID for each imprecise member.
    - _aliased_by: A list of _alias_color IDs referencing an imprecise member.
      The current member can be accessed via the imprecise capability for
      the referenced member.
    - _alias_pointer_member: Non-null for imprecise members only.
      Marks imprecise members for which the capability aliases another
      pointer member.
    """
    task_namespace = "subobject"
    task_name = "annotate-imprecise-subobjects"

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

    @output
    def imprecise_layouts(self):
        return ValueTarget(self, "imprecise-layouts")

    def run(self):
        """
        Construct the transformed dataframe.
        """
        df = self.layouts.struct_layouts.get()
        if len(df) == 0:
            # XXX Did not find any imprecise members, return an empty frame with the right columns
            self.imprecise_layouts.assign(df)
            return

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
        have_overlap = ((pl.col("byte_offset") < pl.col("top__r")) & (pl.col("base__r") < top) &
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
                                                            pl.col("_aliased_by").drop_nulls())

        assert len(tmp_df) == len(df)
        assert set(tmp_df.columns) == set(df.columns + ["_aliased_by", "_alias_color"])
        df = tmp_df.join(imprecise,
                         left_on=STRUCT_MEMBER_ID_COLS,
                         right_on=IMPRECISE_MEMBER_ID_COLS,
                         suffix="__r",
                         how="left").select(~cs.ends_with("__r"))
        self.imprecise_layouts.assign(df)
