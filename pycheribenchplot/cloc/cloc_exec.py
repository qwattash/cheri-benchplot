from dataclasses import MISSING, dataclass
from pathlib import Path
from typing import Any

import polars as pl
import polars.selectors as cs
from git import Repo
from git.exc import BadName, InvalidGitRepositoryError

from ..core.artefact import PLDataFrameLoadTask, Target
from ..core.config import Config, ConfigPath, config_field
from ..core.error import ConfigurationError
from ..core.task import ExecutionTask, output


@dataclass
class ClocRepoConfig(Config):
    repo_path: ConfigPath = config_field(MISSING, desc="Target repository path")
    baseline_ref: str = config_field(MISSING, desc="Baseline git ref")
    name: str | None = config_field(None, desc="Human-readable name of the cloc data, defaults to the repo path")
    head_ref: str = config_field("HEAD", desc="Repository head ref")
    baseline_path: ConfigPath | None = config_field(
        None, desc="Baseline repository path if the baseline is from another subtree")
    accept_ext: list[str] = config_field(list, desc="Extensions to accept e.g. 'c'")
    accept_filter: str | None = config_field(None, desc="Filter the directory to diff within the repository")
    reject_filter: str | None = config_field(None, desc="Filter out specified directory within the repository")
    reject_filter: list[str] = config_field(list, desc="")
    cloc_args: list[str] = config_field(list, desc="Extra arguments for cloc")

    def _check_ref(self, repo_path, ref):
        try:
            repo = Repo(repo_path)
            return repo.commit(ref).hexsha
        except BadName:
            raise ValidationError("Invalid configuration: commit REF %s is not valid", ref)

    def resolve_refs(self, user_config):
        if not self.repo_path.is_absolute():
            self.repo_path = user_config.src_path / self.repo_path
        if not self.baseline_path.is_absolute():
            self.baseline_path = user_config.src_path / self.baseline_path
        # Verify that the git refs are valid and resolve them to git SHA.
        self.head_ref = self._check_ref(self.repo_path, self.head_ref)
        self.baseline_ref = self._check_ref(self.baseline_path, self.baseline_ref)

    def __post_init__(self):
        if self.baseline_path is None:
            self.baseline_path = self.repo_path
        if self.name is None:
            self.name = str(self.repo_path)


class LoadClocFile(PLDataFrameLoadTask):
    """
    Loader for cloc output data
    """
    task_namespace = "cloc"
    task_name = "load-cloc"

    def _unnest_column(self, df, col):
        unwrapped = df.select(col).unnest(col).unpivot(variable_name="file", value_name=col)
        count = unwrapped.with_columns(pl.col(col).struct.field("code").alias(col))
        return count

    def _load_one_json(self, path: Path) -> pl.DataFrame:
        df = pl.read_json(path)

        if "added" in df.columns:
            return self._load_diff_file(df)
        else:
            return self._load_count_file(df)

    def _load_diff_file(self, df):
        df = df.select("added", "modified", "removed", "same")
        added = self._unnest_column(df, "added")
        modified = self._unnest_column(df, "modified")
        removed = self._unnest_column(df, "removed")
        same = self._unnest_column(df, "same")

        assert added.shape == modified.shape
        assert added.shape == removed.shape
        assert added.shape == same.shape

        # now join everything together on the file column
        df = added.join(modified, on="file").join(removed, on="file").join(same, on="file")
        return df

    def _load_count_file(self, df):
        df = df.select(cs.exclude("header", "SUM"))
        df = df.unpivot(variable_name="file", value_name="count")
        df = df.with_columns(pl.col("count").struct.field("code").alias("count"))
        return df


class ClocExecTask(ExecutionTask):
    """
    Extract LoC diff information from one or more git repositories.

    The extraction parameters are determined by the scenario.
    """
    task_namespace = "cloc"
    task_name = "extract"
    public = True
    task_config_class = ClocRepoConfig

    @output
    def cloc_output(self):
        return Target(self, "out", ext="json", loader=LoadClocFile)

    @output
    def cloc_baseline(self):
        return Target(self, "baseline", ext="json", loader=LoadClocFile)

    def repo_config(self):
        return self.config

    def run(self):
        super().run()

        if self.benchmark.config.iterations > 1:
            self.logger.warning("Cloc task is running for multiple iterations")

        cli_args = []

        self.script.exec_iteration("cloc", template="cloc.hook.jinja")
        self.script.extend_context({
            "cloc_config": self.repo_config(),
            "cloc_output": self.cloc_output.single_path(),
            "cloc_baseline": self.cloc_baseline.single_path(),
            "cloc_args": cli_args
        })


@dataclass
class ClocRepoSpec(Config):
    matches: dict[str,
                  Any] = config_field(Config.REQUIRED,
                                      desc="Use this repo spec when all parameterisation keys match the given value.")
    repo_config: ClocRepoConfig = config_field(Config.REQUIRED, desc="Cloc configuration.")

    def check_matches(self, params: dict[str, Any]) -> bool:
        """
        Check whether this RepoSpec matches the given parameterisation
        """
        for key, val in self.matches.items():
            if key not in params or params[key] != val:
                return False
        return True


@dataclass
class ClocMultiRepoConfig(Config):
    """
    Configuration to run the Cloc tool over multiple repositories.

    Each repository defines a ClocRepoConfig, which is enabled using
    parameterisation matchers.
    """
    repos: list[ClocRepoSpec] = config_field(
        list,
        desc="Per-repository configurations. Each benchmark run selects one configuration "
        "based on the matcher, note that the match must be unique.")

    def resolve_refs(self, user_config):
        for repo_spec in self.repos:
            repo_spec.repo_config.resolve_refs(user_config)


class ClocMultiRepoExecTask(ClocExecTask):
    """
    Select the cloc configuration based on the parameterisation.
    """
    task_namespace = "cloc"
    task_name = "multi-extract"
    public = True
    task_config_class = ClocMultiRepoConfig

    def repo_config(self):
        selected = None
        for repo_spec in self.config.repos:
            if not repo_spec.check_matches(self.benchmark.parameters):
                continue
            if selected is not None:
                self.logger.error("Can not select unique cloc RepoConfig, multiple matches for parameterisation %s",
                                  self.benchmark.parameters)
                raise ConfigurationError("Multiple matching repo configs")
            selected = repo_spec.repo_config
        return selected


@dataclass
class CheriBSDClocConfig(ClocMultiRepoConfig):
    """
    Configuration with explicit repos related to CheriBSD LoC extraction.
    """
    cheribsd: ClocRepoSpec = config_field(Config.REQUIRED, desc="CheriBSD main source tree cloc matcher and config.")
    drm: ClocRepoSpec | None = config_field(None, desc="CheriBSD DRM subtree cloc matcher and config.")
    zfs: ClocRepoSpec | None = config_field(None, desc="CheriBSD ZFS subtree cloc matcher and config.")

    def resolve_refs(self, user_config):
        self.cheribsd.repo_config.resolve_refs(user_config)
        if self.drm:
            self.drm.repo_config.resolve_refs(user_config)
        if self.zfs:
            self.zfs.repo_config.resolve_refs(user_config)
        super().resolve_refs(user_config)


class CheriBSDClocExecTask(ClocMultiRepoExecTask):
    """
    Helper task to extract cheribsd LoC diff information.

    This is a convenience task that handles defaults for CheriBSD specific repos:
    - cheribsd: Configures cloc to extract LoC diff from all kernel subsystems.
    - drm: Configures cloc to extract LoC diff from the DRM kernel subsystem.
    - zfs: Configures cloc to extract LoC diff from the ZFS kernel subsystem.

    Cheri kernel sources include a number of subrepos that must be cloned
    separately in order to be used as baseline:
     - drm: https://github.com/evadot/drm-subtree
     - zfs: https://github.com/CTSRD-CHERI/zfs.git
    """
    task_namespace = "cloc"
    task_name = "extract-cheribsd"
    public = True
    task_config_class = CheriBSDClocConfig

    def __init__(self, context, script, task_config):
        super().__init__(context, script, task_config)
        self.config.resolve_refs(self.session.user_config)

    def repo_config(self):
        default_ext = ["c", "h", "S", "s", "m"]

        if self.config.cheribsd.check_matches(self.benchmark.parameters):
            bsd_config = self.config.cheribsd.repo_config
            self.logger.info("Using CheriBSD baseline REF %s", bsd_config.baseline_ref)
            if not bsd_config.accept_ext:
                self.logger.info("Using default CheriBSD file ext filter: '%s'", default_ext)
                bsd_config.accept_ext = default_ext
            if not bsd_config.accept_filter:
                default_accept = "sys/"
                self.logger.info("Using default CheriBSD accept filter: '%s'", default_accept)
                bsd_config.accept_filter = default_accept
            return bsd_config

        if self.config.drm and self.config.drm.check_matches(self.benchmark.parameters):
            drm_config = self.config.drm.repo_config
            self.logger.info("Using DRM baseline REF %s", drm_config.baseline_ref)
            if drm_config.repo_path != self.config.cheribsd.repo_config.repo_path:
                # Complain if the target DRM repo is not the cheribsd repo
                self.logger.warning("Unexpected DRM repo path != CheriBSD repo path")
            if not drm_config.accept_ext:
                self.logger.info("Using default DRM file ext filter: '%s'", default_ext)
                drm_config.accept_ext = default_ext
            if not drm_config.accept_filter:
                default_accept = "sys/dev/drm/"
                self.logger.info("Using default DRM accept filter: '%s'", default_accept)
                drm_config.accept_filter = default_accept
            return drm_config

        if self.config.zfs and self.config.zfs.check_matches(self.benchmark.parameters):
            zfs_config = self.config.zfs.repo_config
            self.logger.info("Using ZFS baseline REF %s", zfs_config.baseline_ref)
            if not zfs_config.accept_ext:
                self.logger.info("Using default ZFS file ext filter: '%s'", default_ext)
                zfs_config.accept_ext = default_ext
            return zfs_config

        return super().repo_config()
