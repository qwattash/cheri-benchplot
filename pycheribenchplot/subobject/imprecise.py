limport re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

from ..core.artefact import SQLTarget
from ..core.config import Config, ConfigPath
from ..core.task import DataGenTask, output
from ..core.util import SubprocessHelper, resolve_system_command


@dataclass
class PathMatchSpec(Config):
    path: ConfigPath
    matcher: str | None = None


@dataclass
class ExtractImpreciseSubobjectConfig(Config):
    """
    Configure the imprecise sub-object extractor.
    """

    #: List of paths that are used to extract DWARF information.
    #: Note that multiple paths will be considered as belonging to the same "object",
    #: if multiple objects are being compared, we should use parameterization.
    #: Note that relative paths will be interpreted as relative paths into a
    #: cheribuild rootfs.
    dwarf_data_sources: List[PathMatchSpec]

    #: Optional path to the dwarf scraper tool
    dwarf_scraper: ConfigPath|None = None

    #: Optional path prefix to strip from source file paths
    src_path_prefix: ConfigPath|None = None


class ExtractImpreciseSubobject(DataGenTask):
    """
    Extract CheriBSD DWARF type information for aggregate data types.
    This will generate a database using the dwarf_scraper tool.
    The dwarf_scraper tool must be in $PATH or passed as a task configuration
    parameter.

    Example configuration:

    .. code-block:: json

        {
            "instance_config": {
                "instances": [{
                    "kernel": "KERNEL-CONFIG-WITH-SUBOBJ-STATS",
                    "cheri_target": "riscv64-purecap",
                    "kernelabi": "purecap"
                }, {
                    "kernel": "KERNEL-CONFIG-WITH-SUBOBJ-STATS",
                    "cheri_target": "morello-purecap",
                    "kernelabi": "purecap"
                }]
            },
            "benchmark_config": [{
                "name": "example",
                "generators": [{
                    "handler": "kernel-static.cheribsd-subobject-stats",
                    "task_options": {
                        "dwarf_data_sources": [
                            {"path": "/path/to/my/file.elf"}
                        ],
                        "dwarf_scraper": "path/to/dwarf_scraper"
                    }
                }]
            }]
        }

    """
    public = True
    task_namespace = "subobject"
    task_name = "extract-imprecise-v2"
    task_config_class = ExtractImpreciseSubobjectConfig

    def __init__(self, benchmark, script, task_config):
        super().__init__(benchmark, script, task_config)

        if self.config.dwarf_scraper:
            self._dwarf_scraper = self.config.dwarf_scraper
            if not self._dwarf_scraper.exists():
                self.logger.error("dwarf_scraper tool does not exist at %s",
                                  self.config.dwarf_scraper)
                raise RuntimeError("Can not find dwarf_scraper")
        else:
            self._dwarf_scraper = resolve_system_command("dwarf_scraper")

    @output
    def struct_layout_db(self):
        return SQLTarget(self, "layout-db")

    def run(self):
        # Ensure that the database directory exists
        self.struct_layout_db.single_path().parent.mkdir(exist_ok=True)
        args = ["--stdin", "--database", self.struct_layout_db.single_path(),
                "--scrapers", "struct-layout"]
        if self.config.src_path_prefix:
            args += ["--prefix", self.config.src_path_prefix]
        scraper = SubprocessHelper(self._dwarf_scraper, args, logger=self.logger)
        scraper.start()
        for src_spec in self.config.dwarf_data_sources:
            for item in self._resolve_paths(src_spec):
                self.logger.debug("Inspect subobject bounds from %s", item)
                if not item.is_file():
                    self.logger.error("File %s is not a regular file, skipping", item)
                    continue
                scraper.stdin.write(str(item).encode())
        scraper.stdin.close()
        scraper.wait()

    def _resolve_paths(self, target: PathMatchSpec) -> Iterator[Path]:
        """
        Resolve a path matcher to a list of paths.
        """
        if target.matcher is None:
            yield from [target.path]
        else:
            matcher = re.compile(target.matcher)
            def match_path(p: Path):
                r = p.relative_to(target.path)
                return matcher.match(str(r))
            yield from filter(match_path, target.path.rglob("*"))
