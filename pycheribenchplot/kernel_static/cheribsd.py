from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from ..core.config import Config
from ..core.task import DataGenTask
from ..core.util import SubprocessHelper


@dataclass
class CheriBSDBuildConfig(Config):
    """
    Task options for the cheribsd-subobject-bounds-stats generator
    """
    clean_build: bool = False
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

    def _do_build(self, build_root: Path):
        instance_config = self.benchmark.config.instance

        kconfig = (self.session.user_config.cheribsd_path / instance_config.cheri_target.freebsd_kconf_dir() /
                   instance_config.kernel)
        target = f"cheribsd-{instance_config.cheri_target.value}"
        cbuild_opts = [
            "--build-root", build_root, "--skip-update", "--skip-buildworld", "--skip-install", target,
            f"--{target}/kernel-config", instance_config.kernel
        ]
        if self.config.clean_build:
            cbuild_opts += "--clean"

        build_cmd = SubprocessHelper(self._cheribuild, cbuild_opts)
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
