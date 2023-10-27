from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from git import Repo
from git.exc import BadName, InvalidGitRepositoryError

from ..core.artefact import LocalFileTarget
from ..core.config import Config
from ..core.error import ConfigurationError
from ..core.pandas_util import generalized_xs, map_index
from ..core.plot import PlotTarget, PlotTask, new_figure
from ..core.task import dependency, output
from .cloc_generic import ExtractLoCBase, ExtractRepoLoCConfig
from .model import LoCCountModel, LoCDiffModel


def abspath_or(path: Path | str, default_base: Path) -> Path:
    """
    Utility function to get either an absolute path or a default value
    """
    path = Path(path)
    if path.is_absolute():
        return path
    else:
        return default_base / path


@dataclass
class ExtractCheriBSDLoCConfig(ExtractRepoLoCConfig):
    drm: Optional[ExtractRepoLoCConfig] = None
    zfs: Optional[ExtractRepoLoCConfig] = None


class ExtractLoCCheriBSD(ExtractLoCBase):
    """
    Task that generates LoC information for the CheriBSD kernel changes.

    This is a convenience task to avoid to have to manually specify complex
    path filtering rules in the configuration file.

    Cheri kernel sources include a number of subrepos that must be cloned
    separately in order to be used as baseline:
     - drm: https://github.com/evadot/drm-subtree
     - zfs: https://github.com/CTSRD-CHERI/zfs.git
    """
    public = True
    task_name = "cheribsd-kernel"
    task_config_class = ExtractCheriBSDLoCConfig

    def __init__(self, session, task_config=None):
        super().__init__(session, task_config=task_config)

        # Determine defaults for the main kernel repo
        if self.config.repo_path is None:
            repo_fullpath = self.session.user_config.cheribsd_path
            self.config.repo_path = repo_fullpath.relative_to(self.session.user_config.src_path)
        else:
            repo_fullpath = self.session.user_config.src_path / self.config.repo_path
        self._repo = Repo(repo_fullpath)

        if not self.config.name:
            self.config.name = self.config.repo_path.name

        # Load ref to last freebsd merged commit
        with open(repo_fullpath / ".last_merge", "r") as last_merge:
            default_baseline = last_merge.read().strip()

        # Cleanup the freebsd refs, this can not be done in config because
        # we can't access the user_config there...
        self.config.baseline_ref = self._resolve_git_ref(self._repo, self.config.baseline_ref, default_baseline)
        self.config.head_ref = self._resolve_git_ref(self._repo, self.config.head_ref, self._repo.head.commit.hexsha)

        if self.config.baseline_ref is None:
            self.config.baseline_ref = last_merge

        # Initialize the subrepo configurations if there are no overrides
        if self.config.drm is None:
            self.config.drm = ExtractRepoLoCConfig(
                name="drm-subtree",
                # Note that we use the CheriBSD repo but we have a separate baseline
                repo_path=self.config.repo_path,
                baseline_ref="16c676289c008fbc28995c9fcc81309c0f560b75",
                head_ref=str(self.config.head_ref),
                accept_ext=r"\.[chSsm]$",
                accept_filter=[r"^sys/dev/drm/"])
        else:
            # Verify that the configuration is what we expect
            if self.config.drm.name != "drm-subtree":
                self.logger.error("Invalid DRM repo_name='%s', expected to be 'drm-subtree'", self.config.drm.name)
                raise ConfigurationError("Invalid DRM repo_name, expected to be 'drm-subtree'")

        if self.config.zfs is None:
            self.config.zfs = ExtractRepoLoCConfig(
                name="zfs",
                repo_path=Path("zfs"),
                # Note the refs are in the zfs repo
                baseline_ref="freebsd",
                head_ref="cheri-purecap",
                accept_ext=r"\.[chSsm]$")

        # Ensure that drm and zfs are there
        drm_repo = abspath_or(self.config.drm.repo_path, self.session.user_config.src_path)
        zfs_repo = abspath_or(self.config.zfs.repo_path, self.session.user_config.src_path)
        try:
            self._drm_repo = Repo(drm_repo)
            self.logger.debug("DRM subrepo baseline at %s", drm_repo)
        except InvalidGitRepositoryError:
            self.logger.error("Invalid DRM subrepo clone at %s", drm_repo)
            raise
        try:
            self._zfs_repo = Repo(zfs_repo)
            self.logger.debug("ZFS subrepo baseline at %s", zfs_repo)
        except InvalidGitRepositoryError:
            self.logger.error("Invalid ZFS subrepo clone at %s", zfs_repo)
            raise

        # Add standard kernel file filters
        self.config.accept_ext = r"\.[chSsmy]$"
        self.config.accept_filter += [r"^sys/"]
        self.config.reject_filter += [
            r"tests?", r"^sys/contrib/openzfs", r"^sys/contrib/subrepo-openzfs", r"^sys/dev/drm"
        ]

        self.logger.debug("Constructed LoC CheriBSD configuration: %s", asdict(self.config))

    def _resolve_git_ref(self, repo, ref, default):
        if ref is None:
            try:
                resolved = repo.commit(default).hexsha
            except BadName:
                self.logger.exception("Invalid default commit ref %s", default)
                raise
        else:
            try:
                resolved = repo.commit(ref).hexsha
            except BadName:
                self.logger.exception("Invalid cheribsd ref %s", ref)
                raise
        return resolved

    def run(self):
        # Extract base kernel LoC data
        diff_df, baseline_df = self._extract_loc(self.config, ["--match-d=sys/"])
        # Extract the drm LoC data
        diff_drm_df, baseline_drm_df = self._extract_loc(self.config.drm, ["--match-d=sys/"])
        # Extract zfs LoC data
        diff_zfs_df, baseline_zfs_df = self._extract_loc(self.config.zfs)

        # Since zfs is extracted outside of the cheribsd tree, we need to prepend the
        # subrepo path otherwise the paths will not make sense when matching to the compilation DB
        diff_zfs_df = map_index(diff_zfs_df, "file", lambda v: "sys/contrib/subrepo-openzfs/" + v)
        baseline_zfs_df = map_index(baseline_zfs_df, "file", lambda v: "sys/contrib/subrepo-openzfs/" + v)

        all_diff = pd.concat([diff_df, diff_drm_df, diff_zfs_df])
        all_diff.to_csv(self.cloc_diff.path)
        all_baseline = pd.concat([baseline_df, baseline_drm_df, baseline_zfs_df])
        all_baseline.to_csv(self.cloc_baseline.path)

    @output
    def cloc_diff(self):
        return LocalFileTarget(self, prefix="cloc-diff", ext="csv", model=LoCDiffModel)

    @output
    def cloc_baseline(self):
        return LocalFileTarget(self, prefix="cloc-baseline", ext="csv", model=LoCCountModel)
