import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ..core.artefact import LocalFileTarget
from ..core.config import Config
from ..core.task import DataGenTask, output
from ..core.util import SubprocessHelper, resolve_system_command


@dataclass
class CheriBSDBuildConfig(Config):
    """
    Task options for the cheribsd-subobject-bounds-stats generator
    """
    clean_build: bool = True
    ephemeral_build_root: bool = False


class CheriBSDBuild(DataGenTask):
    """
    Run a cheribsd build to extract information from the compiler or the build system
    """
    task_config_class = CheriBSDBuildConfig

    def __init__(self, benchmark, script, task_config=None):
        super().__init__(benchmark, script, task_config=task_config)
        #: Path to cheribuild script
        self._cheribuild = self.session.user_config.cheribuild_path / "cheribuild.py"

    def _kernel_build_path(self, build_root: Path) -> Path:
        """
        Retrieve the kernel build directory from the build root directory
        """
        instance_config = self.benchmark.config.instance
        build_base = build_root / f"cheribsd-{instance_config.cheri_target.value}-build"
        path_match = list(build_base.glob(f"**/sys/{instance_config.kernel}"))
        if len(path_match) == 0:
            self.logger.error("No kernel build directory for %s in %s", instance_config.kernel, build_root)
            raise FileNotFoundError("Missing kernel build directory")
        assert len(path_match) == 1
        return path_match[0]

    def _make_subprocess(self, build_root: Path, cbuild_opts: list) -> SubprocessHelper:
        build_cmd = SubprocessHelper(self._cheribuild, cbuild_opts)

    def _do_build(self, build_root: Path):
        instance_config = self.benchmark.config.instance
        cheri_target = instance_config.cheri_target

        kconfig = (self.session.user_config.cheribsd_path / cheri_target.freebsd_kconf_dir() / instance_config.kernel)
        target = f"cheribsd-{cheri_target.value}"
        cbuild_opts = [
            "--build-root", build_root, "--skip-update", "--skip-buildworld", "--skip-install", target,
            f"--{target}/kernel-config", instance_config.kernel
        ]
        if self.config.clean_build:
            cbuild_opts += ["--clean"]
        if instance_config.platform.is_fpga():
            mfs_image = self.session.user_config.sdk_path / f"cheribsd-mfs-root-{cheri_target}.img"
            cbuild_opts += [f"--{target}/mfs-root-image", mfs_image]

        build_cmd = self._make_subprocess(build_root, cbuild_opts)
        build_cmd.run()

        self._extract(build_root)

    def _extract(self, build_root: Path):
        raise NotImplementedError("Must override")

    def run(self):
        if self.config.ephemeral_build_root:
            with TemporaryDirectory() as build_dir:
                tmp_path = Path(build_dir)
                self._do_build(tmp_path)
        else:
            self._do_build(self.session.user_config.build_path)


class CheriBSDCompilationDB(CheriBSDBuild):
    public = True
    task_namespace = "kernel-build"
    task_name = "cheribuild-cdb-trace"

    def __init__(self, benchmark, script, task_config=None):
        super().__init__(benchmark, script, task_config=task_config)
        #: strace is required for this
        self._strace = resolve_system_command("strace", self.logger)

    def _strace_tmp_output(self, build_root: Path) -> Path:
        return build_root / "strace-output.txt"

    def _make_subprocess(self, build_root, cbuild_opts):
        strace_opts = [
            "-o",
            self._strace_tmp_output(build_root), "-qqq", "--signal=!SIGCHLD", "-f", "--trace=open,openat", "-z"
        ]
        cbuild_cmd = [self._cheribuild] + cbuild_opts
        return SubprocessHelper(self._strace, strace_opts + cbuild_cmd)

    def _extract(self, build_root):
        open_re = re.compile(r"openat\(.*, ?\"([a-zA-Z0-9_/.-]+\.[hcmS])\"")
        file_set = set()
        with open(self._strace_tmp_output(build_root), "r") as strace_fd:
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
        df = pd.DataFrame({"files": list(file_set)})
        df.to_json(self.compilation_db.path)

    @output
    def compilation_db(self):
        return LocalFileTarget(self, ext="json")
