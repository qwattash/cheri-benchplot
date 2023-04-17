import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pycheribenchplot.core.analysis import BenchmarkDataLoadTask
from pycheribenchplot.core.artefact import LocalFileTarget
from pycheribenchplot.core.config import Config
from pycheribenchplot.core.task import DataGenTask, output
from pycheribenchplot.core.util import SubprocessHelper, resolve_system_command


@dataclass
class CompilationDBConfig(Config):
    ephemeral_build_root: bool = True


class CheriBSDCompilationDB(DataGenTask):
    """
    This task extracts the set of files that are actually used during a build.
    These are considered interesting in terms of changes, especially regarding drivers.
    We only care about files in the cheribsd source tree during the compilation process.
    Currently build both the riscv and morello targets and make an union of the files
    we touch. Both the FPGA/Morello HW and QEMU targets are built. We only build the
    kernels, in a temporary directory to ensure reproducibility.

    We use the strace command here to detect every file ever touched by the compilation
    process. This allows to pick up both C sources, headers and anything else.
    """
    public = True
    task_namespace = "kernel-history"
    task_name = "cheribuild-cdb-trace"
    task_config_class = CompilationDBConfig

    def __init__(self, session, script, task_config=None):
        super().__init__(session, script, task_config=task_config)
        #: Path to cheribuild script
        self._cheribuild = self.session.user_config.cheribuild_path / "cheribuild.py"
        #: strace is required for this
        self._strace = resolve_system_command("strace", self.logger)

    def _extract_files(self, path: Path) -> set[str]:
        open_re = re.compile(r"openat\(.*, ?\"([a-zA-Z0-9_/.-]+\.[hcmS])\"")
        file_set = set()
        with open(path, "r") as strace_fd:
            for line in strace_fd:
                m = open_re.search(line)
                if not m:
                    continue
                p = Path(m.group(1))
                try:
                    rp = p.relative_to(self.session.user_config.cheribsd_path)
                    file_set.add(str(rp))
                except ValueError:
                    # not relative to cheribsd_path
                    continue
        return file_set

    def _do_cheribuild(self, build_root: Path) -> Path:
        instance_config = self.benchmark.config.instance
        target = f"cheribsd-{instance_config.cheri_target.value}"
        kernel = instance_config.kernel

        strace_file = build_root / "strace-output.txt"
        strace_opts = ["-o", strace_file, "-qqq", "--signal=!SIGCHLD", "-f", "--trace=open,openat", "-z"]
        cbuild_opts = [
            "--build-root", build_root, "--clean", "--skip-update", "--skip-buildworld", "--skip-install", target,
            f"--{target}/kernel-config", kernel
        ]
        cbuild_cmd = [self._cheribuild] + cbuild_opts
        build_cmd = SubprocessHelper(self._strace, strace_opts + cbuild_cmd)
        build_cmd.run()

        return strace_file

    def _do_run(self, build_root):
        compilation_db = self._do_cheribuild(build_root)
        file_set = self._extract_files(compilation_db)
        df = pd.DataFrame({"files": list(file_set)})
        df.to_json(self.compilation_db.path)

    def run(self):
        """
        Try to build cheribsd and record the files that are built/used.
        We only care about files in the cheribsd source tree during the compilation process.
        Currently build both the riscv and morello targets and make an union of the files
        we touch. Both the FPGA/Morello HW and QEMU targets are built. We only build the
        kernels, in a temporary directory to ensure reproducibility.
        """

        if self.config.ephemeral_build_root:
            with TemporaryDirectory() as build_dir:
                self._do_run(Path(build_dir))
        else:
            self._do_run(self.session.user_config.build_path)

    @output
    def compilation_db(self):
        return LocalFileTarget(self, ext="json")
