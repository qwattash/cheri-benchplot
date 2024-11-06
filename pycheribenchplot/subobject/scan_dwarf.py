import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from ..core.artefact import SQLTarget
from ..core.config import Config, ConfigPath
from ..core.task import ExecutionTask, output
from ..core.tvrs import TVRSExecConfig, TVRSExecTask
from ..core.util import SubprocessHelper, resolve_system_command


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
    strip_src_prefix: str | None = None


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
