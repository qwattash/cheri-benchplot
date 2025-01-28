import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import polars as pl
from sqlalchemy import select

from ..core.analysis import AnalysisTask
from ..core.artefact import DataFrameLoadTask, SQLTarget, ValueTarget
from ..core.config import Config, ConfigPath
from ..core.task import ExecutionTask, dependency, output
from ..core.tvrs import TVRSExecConfig, TVRSExecTask
from ..core.util import SubprocessHelper, resolve_system_command
from .model import LayoutMember, TypeLayout


@dataclass
class PathMatchSpec(Config):
    #: Base path interpreted as the command `find {path}`
    path: ConfigPath
    #: Path match regex interpreted as a grep regular expression
    matcher: str | None = None


@dataclass
class ExtractImpreciseConfig(TVRSExecConfig):
    """
    Configure the imprecise sub-object extractor.
    """
    #: Optional path to the dwarf scraper tool
    dwarf_scraper: ConfigPath | None = None

    #: Enable verbose output from dwarf_scraper
    verbose_scraper: bool = False


@dataclass
class ExtractImpreciseScenario(Config):
    #: List of paths that are used to extract DWARF information.
    #: Note that multiple paths will be considered as belonging to the same "object",
    #: if multiple objects are being compared, we should use parameterization.
    #: Note that relative paths will be interpreted as relative paths into a
    #: cheribuild rootfs.
    dwarf_data_sources: List[PathMatchSpec]
    #: Optional path prefix to strip from source file paths
    src_prefix: str | None = None


class ExtractImpreciseSubobject(TVRSExecTask):
    """
    Extract CheriBSD DWARF type information for aggregate data types.
    This will generate a database using the dwarf_scraper tool.
    The dwarf_scraper tool must be in $PATH or passed as a task configuration
    parameter.
    """
    public = True
    task_namespace = "subobject"
    task_name = "extract-imprecise"
    script_template = "dwarf-scraper.sh.jinja"
    task_config_class = ExtractImpreciseConfig
    scenario_config_class = ExtractImpreciseScenario

    @output
    def struct_layout_db(self):
        return SQLTarget(self, "layout-db")

    def run(self):
        super().run()
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
