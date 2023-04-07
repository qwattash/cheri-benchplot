import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import pandas as pd
import seaborn as sns
from git import Repo
from git.exc import BadName

from pycheribenchplot.core.analysis import AnalysisTask
from pycheribenchplot.core.artefact import (AnalysisFileTarget, DataFrameTarget, LocalFileTarget)
from pycheribenchplot.core.config import Config
from pycheribenchplot.core.pandas_util import generalized_xs
from pycheribenchplot.core.plot import PlotTarget, PlotTask, new_figure
from pycheribenchplot.core.plot_util.mosaic import mosaic_treemap
from pycheribenchplot.core.task import (DataGenTask, SessionDataGenTask, dependency, output)
from pycheribenchplot.core.util import SubprocessHelper, resolve_system_command

from .model import LoCCountModel, LoCDiffModel


@dataclass
class CheriBSDKernelLineChangesConfig(Config):
    #: The freebsd baseline tag or commit SHA to use, if none is given
    #: this defaults to the content of .last_merge
    freebsd_baseline: Optional[str] = None
    #: The freebsd head holding the CHERI changes to compare
    freebsd_head: Optional[str] = None
    #: Filter the diff based on the files found in the compilation DB
    restrict_to_compilation_db: bool = False


class CheriBSDKernelLineChanges(SessionDataGenTask):
    """
    This task extracts cheribsd kernel changes in terms of lines of codes.

    Use git diff with respect to a given freebsd-main-xxxx tag as the baseline
    to extract the diff.
    """
    public = True
    task_namespace = "kernel-history"
    task_name = "cheribsd-diff"
    task_config_class = CheriBSDKernelLineChangesConfig

    def __init__(self, session, task_config=None):
        super().__init__(session, task_config=task_config)
        self._cloc = resolve_system_command("cloc", self.logger)

        with open(self.session.user_config.cheribsd_path / ".last_merge", "r") as last_merge:
            default_baseline = last_merge.read().strip()

        # Cleanup the freebsd refs, this can not be done in config because
        # we can't access the user_config there...
        self._repo = Repo(self.session.user_config.cheribsd_path)
        self.config.freebsd_head = self._resolve_cheribsd_ref(self.config.freebsd_head, self._repo.head.commit.hexsha)
        self.config.freebsd_baseline = self._resolve_cheribsd_ref(self.config.freebsd_baseline, default_baseline)

    def _resolve_cheribsd_ref(self, ref, default):
        if ref is None:
            try:
                resolved = self._repo.commit(default).hexsha
            except BadName:
                self.logger.exception("Invalid default commit ref %s", default)
                raise
        else:
            try:
                resolved = self._repo.commit(ref)
            except BadName:
                self.logger.exception("Invalid cheribsd ref %s", ref)
                raise
        return resolved

    def _filter_files(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Only retain files in sys/* that are not tests.
        Also remove subrepo-openzfs diff which causes noise.
        """
        paths = df.index.get_level_values("file")
        in_sys = paths.str.startswith("sys/")
        is_test = paths.str.contains(r"tests?")
        is_valid_ext = paths.str.contains(r"\.[chSsm]$")
        # The openzfs and drm patches add a whole subtree, which pollutes the diff
        # with non-cheri changes.
        is_subrepo_zfs = paths.str.contains(r"subrepo-openzfs")
        is_drm = paths.str.startswith("sys/dev/drm")
        return df.loc[in_sys & is_valid_ext & ~is_drm & ~is_test & ~is_subrepo_zfs].copy()

    def run(self):
        concurrency = self.session.config.concurrent_workers
        with TemporaryDirectory() as out_dir:
            outfile = Path(out_dir) / "cloc"
            cloc_args = [
                "--skip-uniqueness",
                # f"--processes={concurrency}",
                f"--report-file={outfile}",
                "--exclude-content=DO NOT EDIT",
                "--file-encoding=UTF-8",
                "--fullpath",
                "--match-d=sys",
                "--by-file",
                "--json",
                "--git",
                "--count-and-diff",
                self.config.freebsd_baseline,
                self.config.freebsd_head
            ]
            cloc_cmd = SubprocessHelper(self._cloc, cloc_args)
            cloc_cmd.run(cwd=self.session.user_config.cheribsd_path)

            # cloc generates multiple files, move them to the real outputs
            shutil.copy(outfile.with_suffix("." + self.config.freebsd_baseline), self.raw_cloc_baseline.path)
            shutil.copy(outfile.with_suffix("." + self.config.freebsd_head), self.raw_cloc_head.path)
            shutil.copy(outfile.with_suffix(f".diff.{self.config.freebsd_baseline}.{self.config.freebsd_head}"),
                        self.raw_cloc_diff.path)

        # Prepare the diff LoC count dataframe
        with open(self.raw_cloc_diff.path, "r") as raw_cloc:
            raw_data = json.load(raw_cloc)
        df_set = []
        for key in ["added", "same", "modified", "removed"]:
            chunk = raw_data[key]
            df = pd.DataFrame(chunk).transpose()
            df["how"] = key
            # Drop the nFiles column as it is not meaningful
            df = df.drop("nFiles", axis=1)
            df.index.name = "file"
            df.set_index("how", append=True, inplace=True)
            df_set.append(df)
        df = pd.concat(df_set)
        df = self._filter_files(df)
        df.to_json(self.cloc.path, orient="table")

        # Prepare the baseline LoC count dataframe
        with open(self.raw_cloc_baseline.path, "r") as raw_cloc:
            raw_data = json.load(raw_cloc)
        raw_data.pop("header")
        raw_data.pop("SUM")
        df = pd.DataFrame(raw_data).transpose()
        df.index.name = "file"
        df = self._filter_files(df)
        df.to_json(self.cloc_baseline.path, orient="table")

    @output
    def raw_cloc_baseline(self):
        return LocalFileTarget.from_task(self, prefix="raw-cloc-baseline", ext="json")

    @output
    def raw_cloc_head(self):
        return LocalFileTarget.from_task(self, prefix="raw-cloc-head", ext="json")

    @output
    def raw_cloc_diff(self):
        return LocalFileTarget.from_task(self, prefix="raw-cloc-diff", ext="json")

    @output
    def cloc(self):
        return LocalFileTarget.from_task(self, prefix="cloc", ext="json")

    @output
    def cloc_baseline(self):
        return LocalFileTarget.from_task(self, prefix="cloc-baseline", ext="json")


class LoadLoCData(AnalysisTask):
    """
    Load line of code changes by file.
    This is the place to configure filtering based on the compilation DB or other
    criteria, so that all dependent tasks operate on the filtered data normally.
    """
    task_namespace = "kernel-history"
    task_name = "load-loc-data"

    def run(self):
        # Should have a way to properly fetch the configured datagen task here
        cloc_task = CheriBSDKernelLineChanges(self.session, task_config=CheriBSDKernelLineChangesConfig())
        df = pd.read_json(cloc_task.cloc.path, orient="table")
        self.diff_df.assign(df)

        df = pd.read_json(cloc_task.cloc_baseline.path, orient="table")
        self.baseline_df.assign(df)

    @output
    def diff_df(self):
        return DataFrameTarget(LoCDiffModel.to_schema(self.session))

    @output
    def baseline_df(self):
        return DataFrameTarget(LoCCountModel.to_schema(self.session))


class CheriBSDLineChangesSummary(AnalysisTask):
    """
    Produce a latex-compatible table summary for the SLoC changes.

    The following columns are present:
    - Type: header, source, assembly, other
    - Total SLoC
    - Changed SLoC
    - % Changed SLoC
    - Total Files
    - Changed Files
    - % Changed Files
    """
    public = True
    task_namespace = "kernel-history"
    task_name = "cheribsd-loc-diff-summary"

    @dependency
    def loc_diff(self):
        return LoadLoCData(self.session, self.analysis_config)

    def run(self):
        df = self.loc_diff.diff_df.get()
        baseline_df = self.loc_diff.baseline_df.get()

        # drop the rows with how == "same"
        data_df = generalized_xs(df, how="same", complement=True)
        # don't care about the split in how, all of them are changes
        data_df = data_df.groupby("file").sum()
        # drop rows that do not have code changes
        data_df = data_df.loc[data_df["code"] != 0]

        # now classify files by extension
        def to_file_type(path):
            # Don't expect C++ in the kernel
            if path.endswith(".h"):
                return "header"
            if path.endswith(".c"):
                return "source"
            if path.endswith(".S") or path.endswith(".s"):
                return "assembly"
            return "other"

        data_df["Type"] = data_df.index.get_level_values("file").map(to_file_type)
        baseline_df["Type"] = baseline_df.index.get_level_values("file").map(to_file_type)
        # Group by type and sum the "code" column
        data_count = data_df.groupby("Type")["code"].sum()
        baseline_count = baseline_df.groupby("Type")["code"].sum()
        data_nfiles = data_df.groupby("Type").size()
        baseline_nfiles = baseline_df.groupby("Type").size()

        # add a row with the grand total
        data_count.loc["total"] = data_count.sum()
        baseline_count.loc["total"] = baseline_count.sum()
        data_nfiles.loc["total"] = data_nfiles.sum()
        baseline_nfiles.loc["total"] = baseline_nfiles.sum()

        # Finally, build the stats frame
        stats = {
            "Total SLoC": baseline_count,
            "Changed SLoC": data_count,
            "% Changed SLoC": 100 * data_count / baseline_count,
            "Total files": baseline_nfiles,
            "Changed files": data_nfiles,
            "% Changed files": 100 * data_nfiles / baseline_nfiles
        }

        stats_df = pd.DataFrame(stats).round(1)
        stats_df.to_csv(self.table.path)
        data_df.reset_index().set_index(["Type", "file"]).sort_index().to_csv(self.changed_files.path)

    @output
    def table(self):
        return AnalysisFileTarget.from_task(self, ext="csv")

    @output
    def changed_files(self):
        return AnalysisFileTarget.from_task(self, prefix="all-files", ext="csv")


class CheriBSDLineChangesByPath(PlotTask):
    """
    Produce a mosaic plot showing the changes nesting by path
    """
    public = True
    task_namespace = "kernel-history"
    task_name = "cheribsd-cloc-by-path"

    @dependency
    def loc_diff(self):
        return LoadLoCData(self.session, self.analysis_config)

    def run(self):
        df = self.loc_diff.diff_df.get()
        # Don't care how the lines changed
        data_df = generalized_xs(df, how="same", complement=True)
        data_df = data_df.groupby("file").sum()
        # Drop the rows where code changes are 0, as these probably mark
        # historical changes or just comments
        data_df = data_df.loc[data_df["code"] != 0]
        sns.set_palette("crest")

        with new_figure(self.plot.path) as fig:
            ax = fig.subplots()
            filename_components = data_df.index.get_level_values("file").str.split("/")
            # Remove the leading "sys", as it is everywhere, and the filename component
            data_df["components"] = filename_components.map(lambda c: c[1:-1])
            mosaic_treemap(data_df, fig, ax, bbox=(0, 0, 100, 100), values="code", groups="components", maxlevel=3)

    @output
    def plot(self):
        return PlotTarget.from_task(self)


class CheriBSDLineChangesByFile(PlotTask):
    """
    Produce an histogram sorted by the file with highest diff.
    """
    pass


class CheriBSDLineChangesByType(PlotTask):
    pass
