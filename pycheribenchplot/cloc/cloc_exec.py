from dataclasses import MISSING, dataclass
from pathlib import Path

import polars as pl
import polars.selectors as cs
from git import Repo
from git.exc import BadName, InvalidGitRepositoryError

from ..core.artefact import PLDataFrameLoadTask, Target
from ..core.config import Config, ConfigPath, config_field
from ..core.task import output
from ..core.tvrs import TVRSExecConfig, TVRSExecTask


@dataclass
class ClocScenario(Config):
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


class ClocExecTask(TVRSExecTask):
    """
    Extract LoC diff information from one or more git repositories.

    The extraction parameters are determined by the scenario.
    """
    task_namespace = "cloc"
    task_name = "extract"
    public = True
    task_config_class = TVRSExecConfig
    scenario_config_class = ClocScenario

    @output
    def cloc_output(self):
        return Target(self, "out", ext="json", loader=LoadClocFile)

    @output
    def cloc_baseline(self):
        return Target(self, "baseline", ext="json", loader=LoadClocFile)

    def run(self):
        super().run()

        if self.benchmark.config.iterations > 1:
            self.logger.warning("Cloc task is running for multiple iterations")

        cli_args = []

        self.script.exec_iteration("cloc", template="cloc.hook.jinja")
        self.script.extend_context({
            "cloc_output": self.cloc_output.single_path(),
            "cloc_baseline": self.cloc_baseline.single_path(),
            "cloc_args": cli_args
        })


@dataclass
class CheriBSDClocConfig(TVRSExecConfig):
    """
    Configuration with pre-defined scenario configurations for CheriBSD LoC extraction.
    """
    cheribsd: ClocScenario = None
    drm: ClocScenario | None = None
    zfs: ClocScenario | None = None

    def resolve_refs(self, user_config):
        self.cheribsd.resolve_refs(user_config)
        if self.drm:
            self.drm.resolve_refs(user_config)
        if self.zfs:
            self.zfs.resolve_refs(user_config)


class CheriBSDClocExecTask(ClocExecTask):
    """
    Helper task to extract cheribsd LoC diff information.

    This is a convenience task that pre-fills the configuration scenarios
    with the following:
    - cheribsd: Configures cloc to extract LoC diff from all kernel subsystems.
    - drm: Configures cloc to extract LoC diff from the DRM kernel subsystem.
    - zfs: Configures cloc to extract LoC diff from the ZFS kernel subsystem.

    If the pre-filled scenarios are not named in the configuration file, this will
    behave as ClocExecTask.

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

    def scenario(self):
        scenario_key = self.benchmark.parameters["scenario"]
        default_ext = ["c", "h", "S", "s", "m"]
        match scenario_key:
            case "cheribsd":
                self.logger.info("Using CheriBSD baseline REF %s", self.config.cheribsd.baseline_ref)
                if not self.config.cheribsd.accept_ext:
                    self.logger.info("Using default CheriBSD file ext filter: '%s'", default_ext)
                    self.config.cheribsd.accept_ext = default_ext
                if not self.config.cheribsd.accept_filter:
                    default_accept = "sys/"
                    self.logger.info("Using default CheriBSD accept filter: '%s'", default_accept)
                    self.config.cheribsd.accept_filter = default_accept
                return self.config.cheribsd
            case "drm":
                if not self.config.drm:
                    return super().scenario()
                self.logger.info("Using DRM baseline REF %s", self.config.drm.baseline_ref)
                if self.config.drm.repo_path != self.config.cheribsd.repo_path:
                    # Complain if the target DRM repo is not the cheribsd repo
                    self.logger.warning("Unexpected DRM repo path != CheriBSD repo path")
                if self.config.drm and not self.config.drm.accept_ext:
                    self.logger.info("Using default DRM file ext filter: '%s'", default_ext)
                    self.config.drm.accept_ext = default_ext
                if not self.config.drm.accept_filter:
                    default_accept = "sys/dev/drm/"
                    self.logger.info("Using default DRM accept filter: '%s'", default_accept)
                    self.config.drm.accept_filter = default_accept
                return self.config.drm
            case "zfs":
                if not self.config.zfs:
                    return super().scenario()
                self.logger.info("Using ZFS baseline REF %s", self.config.zfs.baseline_ref)
                if self.config.zfs and not self.config.zfs.accept_ext:
                    self.logger.info("Using default ZFS file ext filter: '%s'", default_ext)
                    self.config.zfs.accept_ext = default_ext
                return self.config.zfs
        return super().scenario()
