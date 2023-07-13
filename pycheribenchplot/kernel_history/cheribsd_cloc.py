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

from ..core.analysis import AnalysisTask
from ..core.artefact import (AnalysisFileTarget, DataFrameTarget, LocalFileTarget)
from ..core.config import Config
from ..core.model import check_data_model
from ..core.pandas_util import generalized_xs
from ..core.plot import PlotTarget, PlotTask, new_figure
from ..core.plot_util.mosaic import mosaic_treemap
from ..core.task import DataGenTask, SessionDataGenTask, dependency, output
from ..core.util import SubprocessHelper, resolve_system_command
from .cheribsd_build import CheriBSDCompilationDB
from .model import (AllCompilationDBModel, CompilationDBModel, LoCCountModel, LoCDiffModel)


def to_file_type(path: str):
    # Don't expect C++ in the kernel
    if path.endswith(".h"):
        return "header"
    if path.endswith(".c"):
        return "source"
    if path.endswith(".S") or path.endswith(".s"):
        return "assembly"
    return "other"


@dataclass
class CheriBSDKernelLineChangesConfig(Config):
    #: The freebsd baseline tag or commit SHA to use, if none is given
    #: this defaults to the content of .last_merge
    freebsd_baseline: Optional[str] = None
    #: The freebsd head holding the CHERI changes to compare
    freebsd_head: Optional[str] = None
    #: DRM subtree reference repo path (relative to cheri repositories path),
    #: should be a clone of https://github.com/evadot/drm-subtree
    drm_subtree_repo: str = "drm-subtree"
    #: ZFS subtree reference repo path (relative to cheri repositories path),
    #: should be a clone of https://github.com/CTSRD-CHERI/zfs.git
    zfs_subtree_repo: str = "zfs"


@dataclass
class ClocSubtreeParams:
    #: subtree path
    subtree: str
    #: raw baseline LoC file
    raw_cloc_baseline_path: Path
    #: raw LoC file for the changed version to compare
    raw_cloc_head_path: Path
    #: raw diff LoC file
    raw_cloc_diff_path: Path
    #: baseline ref
    baseline_ref: str
    #: head ref
    head_ref: str


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
                resolved = self._repo.commit(ref).hexsha
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
        is_valid_ext = paths.str.contains(r"\.[chSsmy]$")
        # The openzfs and drm patches add a whole subtree, which pollutes the diff
        # with non-cheri changes.
        is_subrepo_zfs = paths.str.contains(r"subrepo-openzfs")
        is_drm = paths.str.startswith("sys/dev/drm")
        return df.loc[in_sys & is_valid_ext & ~is_drm & ~is_test & ~is_subrepo_zfs].copy()

    def _get_common_cloc_args(self):
        concurrency = self.session.config.concurrent_workers
        args = [
            "--skip-uniqueness",
            # f"--processes={concurrency}",
            "--exclude-content=DO NOT EDIT",
            "--file-encoding=UTF-8",
            "--fullpath",
            "--by-file",
            "--json",
            "--git",
            "--count-and-diff"
        ]
        return args

    def _compute_diff_loc(self, raw_diff_path: Path) -> pd.DataFrame:
        """
        Normalize the diff per file into a dataframe
        """
        with open(raw_diff_path, "r") as raw_cloc:
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
        return df

    def _compute_loc(self, raw_path: Path) -> pd.DataFrame:
        """
        Normalize baseline count per file into a dataframe
        """
        with open(raw_path, "r") as raw_cloc:
            raw_data = json.load(raw_cloc)
        raw_data.pop("header")
        raw_data.pop("SUM")
        df = pd.DataFrame(raw_data).transpose()
        df.index.name = "file"
        return df

    def cloc_subtree(self,
                     params: ClocSubtreeParams,
                     extra_args: list[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract cloc stats for a subtree
        """
        with TemporaryDirectory() as out_dir:
            outfile = Path(out_dir) / "cloc"
            cloc_args = self._get_common_cloc_args()
            if extra_args:
                cloc_args += extra_args
            cloc_args += [f"--report-file={outfile}", params.baseline_ref, params.head_ref]

            cloc_cmd = SubprocessHelper(self._cloc, cloc_args)
            cloc_cmd.run(cwd=self.session.user_config.cheribsd_path)

            # cloc generates multiple files, move them to the real outputs
            baseline_suffix = str(params.baseline_ref).replace("/", "_")
            head_suffix = str(params.head_ref).replace("/", "_")
            shutil.copy(outfile.with_suffix("." + baseline_suffix), params.raw_cloc_baseline_path)
            shutil.copy(outfile.with_suffix("." + head_suffix), params.raw_cloc_head_path)
            shutil.copy(outfile.with_suffix(f".diff.{baseline_suffix}.{head_suffix}"), params.raw_cloc_diff_path)

        # Prepare the diff LoC count dataframe
        diff_df = self._compute_diff_loc(params.raw_cloc_diff_path)

        # Fixup the paths to always be relative to the main cheribsd repo
        def patch_path(pathstr: str):
            if not params.subtree:
                return pathstr
            path = Path(pathstr)
            if path.is_absolute() and Path(params.baseline_ref).exists():
                path = path.relative_to(params.baseline_ref)
            return params.subtree + "/" + str(path)

        tmp_df = diff_df.reset_index()
        tmp_df["file"] = tmp_df["file"].map(patch_path)
        diff_df = tmp_df.set_index(["file", "how"])

        # Prepare the baseline LoC count dataframe
        baseline_df = self._compute_loc(params.raw_cloc_baseline_path)
        tmp_df = baseline_df.reset_index()
        tmp_df["file"] = tmp_df["file"].map(patch_path)
        baseline_df = tmp_df.set_index("file")

        return (diff_df, baseline_df)

    def cloc_kernel(self):
        """
        Run cloc over the main kernel sources
        """
        params = ClocSubtreeParams(subtree=None,
                                   raw_cloc_baseline_path=self.raw_cloc_baseline.path,
                                   raw_cloc_head_path=self.raw_cloc_head.path,
                                   raw_cloc_diff_path=self.raw_cloc_diff.path,
                                   baseline_ref=self.config.freebsd_baseline,
                                   head_ref=self.config.freebsd_head)
        diff_df, baseline_df = self.cloc_subtree(params, extra_args=["--match-d=sys"])

        # Need to filter out directories we are surely not interested in
        # The rest will be returned and optionally can be further filtered
        # during analysis.
        return (self._filter_files(diff_df), self._filter_files(baseline_df))

    def cloc_drm(self):
        """
        Run cloc for the DRM subtree
        """
        params = ClocSubtreeParams(subtree="sys/dev/drm",
                                   raw_cloc_baseline_path=self.raw_cloc_drm_baseline.path,
                                   raw_cloc_head_path=self.raw_cloc_drm_head.path,
                                   raw_cloc_diff_path=self.raw_cloc_drm_diff.path,
                                   baseline_ref=self.session.user_config.src_path / self.config.drm_subtree_repo,
                                   head_ref=self.config.freebsd_head + ":sys/dev/drm")
        return self.cloc_subtree(params)

    def cloc_zfs(self):
        """
        Run cloc for the ZFS subtree
        """
        params = ClocSubtreeParams(subtree="sys/contrib/subrepo-openzfs",
                                   raw_cloc_baseline_path=self.raw_cloc_zfs_baseline.path,
                                   raw_cloc_head_path=self.raw_cloc_zfs_head.path,
                                   raw_cloc_diff_path=self.raw_cloc_zfs_diff.path,
                                   baseline_ref=self.session.user_config.src_path / self.config.zfs_subtree_repo,
                                   head_ref=self.config.freebsd_head + ":sys/contrib/subrepo-openzfs")
        return self.cloc_subtree(params)

    def run(self):
        df_pairs = []
        df_pairs.append(self.cloc_kernel())
        df_pairs.append(self.cloc_drm())
        df_pairs.append(self.cloc_zfs())

        df = pd.concat([pair[0] for pair in df_pairs])
        # Dump, reset the index because otherwise we lose the name of the index levels
        df.reset_index().to_json(self.cloc.path)

        df = pd.concat([pair[1] for pair in df_pairs])
        # Dump, see above
        df.reset_index().to_json(self.cloc_baseline.path)

    @output
    def raw_cloc_baseline(self):
        return LocalFileTarget(self, prefix="raw-cloc-baseline", ext="json")

    @output
    def raw_cloc_head(self):
        return LocalFileTarget(self, prefix="raw-cloc-head", ext="json")

    @output
    def raw_cloc_diff(self):
        return LocalFileTarget(self, prefix="raw-cloc-diff", ext="json")

    @output
    def raw_cloc_drm_baseline(self):
        return LocalFileTarget(self, prefix="raw-cloc-drm-baseline", ext="json")

    @output
    def raw_cloc_drm_head(self):
        return LocalFileTarget(self, prefix="raw-cloc-drm-head", ext="json")

    @output
    def raw_cloc_drm_diff(self):
        return LocalFileTarget(self, prefix="raw-cloc-drm-diff", ext="json")

    @output
    def raw_cloc_zfs_baseline(self):
        return LocalFileTarget(self, prefix="raw-cloc-zfs-baseline", ext="json")

    @output
    def raw_cloc_zfs_head(self):
        return LocalFileTarget(self, prefix="raw-cloc-zfs-head", ext="json")

    @output
    def raw_cloc_zfs_diff(self):
        return LocalFileTarget(self, prefix="raw-cloc-zfs-diff", ext="json")

    @output
    def cloc(self):
        return LocalFileTarget(self, prefix="cloc", ext="json", model=LoCDiffModel)

    @output
    def cloc_baseline(self):
        return LocalFileTarget(self, prefix="cloc-baseline", ext="json", model=LoCCountModel)


@dataclass
class LoCDataConfig(Config):
    #: Filter the diff based on the files found in the compilation DB
    restrict_to_compilation_db: bool = True


class LoadLoCData(AnalysisTask):
    """
    Load line of code changes by file.
    This is the place to configure filtering based on the compilation DB or other
    criteria, so that all dependent tasks operate on the filtered data normally.
    """
    task_namespace = "kernel-history"
    task_name = "load-loc-data"
    task_config_class = LoCDataConfig

    @dependency
    def compilation_db(self):
        if not self.config.restrict_to_compilation_db:
            return []
        load_tasks = []
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(CheriBSDCompilationDB)
            load_tasks.append(task.compilation_db.get_loader())
        return load_tasks

    @dependency
    def cloc_count(self):
        task = self.session.find_exec_task(CheriBSDKernelLineChanges)
        return task.cloc.get_loader()

    @dependency
    def cloc_baseline(self):
        task = self.session.find_exec_task(CheriBSDKernelLineChanges)
        return task.cloc_baseline.get_loader()

    @check_data_model
    def _load_compilation_db(self) -> AllCompilationDBModel:
        """
        Fetch all compilation DBs and merge them
        """
        cdb_set = []
        for loader in self.compilation_db:
            cdb_set.append(loader.df.get())
        return pd.concat(cdb_set).groupby("files").first().reset_index()

    def run(self):
        diff_df = self.cloc_count.df.get()
        baseline_df = self.cloc_baseline.df.get()
        if self.config.restrict_to_compilation_db:
            cdb = self._load_compilation_db()
            # Filter by compilation DB files
            diff_df = diff_df.loc[diff_df.index.isin(cdb["files"], level="file")]
            baseline_df = baseline_df.loc[baseline_df.index.isin(cdb["files"], level="file")]

            # Store a sorted version of the compilation DB for debugging
            cdb["type"] = cdb["files"].map(to_file_type)
            out_cdb = cdb.sort_values(by=["type", "files"]).set_index("type", append=True)
            out_cdb.to_csv(self.debug_all_files.path)

        self.diff_df.assign(diff_df)
        self.baseline_df.assign(baseline_df)

    @output
    def diff_df(self):
        return DataFrameTarget(self, LoCDiffModel)

    @output
    def baseline_df(self):
        return DataFrameTarget(self, LoCCountModel)

    @output
    def debug_all_files(self):
        return AnalysisFileTarget(self, prefix="debug", ext="csv")


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
    task_config_class = LoCDataConfig

    @dependency
    def loc_diff(self):
        return LoadLoCData(self.session, self.analysis_config, task_config=self.config)

    def run(self):
        df = self.loc_diff.diff_df.get()
        baseline_df = self.loc_diff.baseline_df.get()

        # drop the rows with how == "same"
        data_df = generalized_xs(df, how="same", complement=True)
        # don't care about the split in how, all of them are changes
        data_df = data_df.groupby("file").sum()
        # drop rows that do not have code changes
        data_df = data_df.loc[data_df["code"] != 0]

        # Determine file types
        data_df["Type"] = data_df.index.get_level_values("file").map(to_file_type)
        baseline_df["Type"] = baseline_df.index.get_level_values("file").map(to_file_type)
        # Determine dev vs nodev
        nodev = ~data_df.index.get_level_values("file").str.startswith("sys/dev")
        nodev_df = data_df.loc[nodev]
        baseline_nodev = ~baseline_df.index.get_level_values("file").str.startswith("sys/dev")
        baseline_nodev_df = baseline_df.loc[baseline_nodev]
        # Group by type and sum the "code" column
        data_count = data_df.groupby("Type")["code"].sum()
        data_nodev_count = nodev_df.groupby("Type")["code"].sum()

        baseline_count = baseline_df.groupby("Type")["code"].sum()
        baseline_nodev_count = baseline_nodev_df.groupby("Type")["code"].sum()

        data_nfiles = data_df.groupby("Type").size()
        data_nodev_nfiles = nodev_df.groupby("Type").size()

        baseline_nfiles = baseline_df.groupby("Type").size()
        baseline_nodev_nfiles = baseline_nodev_df.groupby("Type").size()

        # add a row with the grand total
        data_count.loc["total"] = data_count.sum()
        data_nodev_count.loc["total"] = data_nodev_count.sum()
        baseline_count.loc["total"] = baseline_count.sum()
        baseline_nodev_count.loc["total"] = baseline_nodev_count.sum()
        data_nfiles.loc["total"] = data_nfiles.sum()
        data_nodev_nfiles.loc["total"] = data_nodev_nfiles.sum()
        baseline_nfiles.loc["total"] = baseline_nfiles.sum()
        baseline_nodev_nfiles.loc["total"] = baseline_nodev_nfiles.sum()

        # Finally, build the stats frame
        stats = {
            "Total SLoC": baseline_count,
            "Changed SLoC": data_count,
            "% Changed SLoC": 100 * data_count / baseline_count,
            "Total SLoC w/o drivers": baseline_nodev_count,
            "Changed SLoC w/o drivers": data_nodev_count,
            "Total files": baseline_nfiles,
            "Changed files": data_nfiles,
            "% Changed files": 100 * data_nfiles / baseline_nfiles,
            "Total files w/o drivers": baseline_nodev_nfiles,
            "Changed files w/o drivers": data_nodev_nfiles
        }

        stats_df = pd.DataFrame(stats).round(1)
        stats_df.to_csv(self.table.path)
        data_df.reset_index().set_index(["Type", "file"]).sort_index().to_csv(self.changed_files.path)

    @output
    def table(self):
        return AnalysisFileTarget(self, ext="csv")

    @output
    def changed_files(self):
        return AnalysisFileTarget(self, prefix="all-files", ext="csv")


class CheriBSDLineChangesByPath(PlotTask):
    """
    Produce a mosaic plot showing the changes nesting by path
    """
    public = True
    task_namespace = "kernel-history"
    task_name = "cheribsd-cloc-by-path"
    task_config_class = LoCDataConfig

    @dependency
    def loc_diff(self):
        return LoadLoCData(self.session, self.analysis_config, task_config=self.config)

    def run_plot(self):
        df = self.loc_diff.diff_df.get()
        # Don't care how the lines changed
        data_df = generalized_xs(df, how="same", complement=True)
        data_df = data_df.groupby("file").sum()
        # Drop the rows where code changes are 0, as these probably mark
        # historical changes or just comments
        data_df = data_df.loc[data_df["code"] != 0]

        # XXX this does not agree well with multithreading
        sns.set_palette("crest")

        with new_figure(self.plot.path) as fig:
            ax = fig.subplots()
            filename_components = data_df.index.get_level_values("file").str.split("/")
            # Remove the leading "sys", as it is everywhere, and the filename component
            data_df["components"] = filename_components.map(lambda c: c[1:-1])
            mosaic_treemap(data_df, fig, ax, bbox=(0, 0, 100, 100), values="code", groups="components", maxlevel=2)

    @output
    def plot(self):
        return PlotTarget(self)


class CheriBSDLineChangesByFile(PlotTask):
    """
    Produce an histogram sorted by the file with highest diff.

    The histogram is stacked by the "how" index, showing whether the kind of diff.
    There are two outputs, one using absolute LoC changes, the other using the
    percentage of LoC changed with respect to the total file LoC.
    """
    public = True
    task_namespace = "kernel-history"
    task_name = "cheribsd-cloc-by-file"
    task_config_class = LoCDataConfig

    @dependency
    def loc_diff(self):
        return LoadLoCData(self.session, self.analysis_config, task_config=self.config)

    def _prepare_bars(self, df, sort_by, values_col):
        """
        Helper to prepare the dataframe for plotting

        Need to prepare the frame by pivoting how into columns and retaining only
        the relevant code column. This is the format expected by df.plot()
        Filter top N files, stacked by the 'how' index
        """
        NTOP = 50 * len(df.index.unique("how"))
        show_df = df.sort_values(by=[sort_by, "file", "how"], ascending=False).iloc[:NTOP].reset_index()
        show_df["file"] = show_df["file"].str.removeprefix("sys/")
        return show_df.pivot(index=[sort_by, "file"], columns="how", values=values_col)

    def _do_plot(self, df: pd.DataFrame, output: PlotTarget, xlabel: str):
        """
        Helper to plot bars for LoC changes.
        This expects a dataframe produced by _prepare_bars().
        """
        # Use tab10 colors, added => green, modified => orange, removed => red
        cmap = sns.color_palette(as_cmap=True)
        # ordered as green, orange, red
        colors = [cmap[2], cmap[1], cmap[3]]

        with new_figure(output.path) as fig:
            ax = fig.subplots()
            df.reset_index().plot(x="file",
                                  y=["added", "modified", "removed"],
                                  stacked=True,
                                  kind="barh",
                                  ax=ax,
                                  color=colors)
            ax.tick_params(axis="y", labelsize="xx-small")
            ax.set_xlabel(xlabel)

    def run_plot(self):
        df = self.loc_diff.diff_df.get()
        base_df = self.loc_diff.baseline_df.get()
        data_df = generalized_xs(df, how="same", complement=True)

        # Ensure we have the proper theme
        sns.set_theme()

        # Find out the total and percent_total LoC changed by file
        total_loc = data_df.groupby("file").sum()
        total_loc = total_loc.join(base_df, on="file", rsuffix="_baseline")
        total_loc["percent"] = 100 * total_loc["code"] / total_loc["code_baseline"]

        # Backfill the total for each value of the "how" index so we can do the sorting
        # Note: some files are added completely, we ignore them here because the
        # they are not particularly interesting for the relative diff by file.
        # These would always account for 100% of the changes, polluting the histogram
        # with not very useful information.
        aligned, _ = total_loc.align(data_df, level="file")
        data_df["total"] = aligned["code"]
        data_df["total_percent"] = aligned["percent"]
        data_df["baseline"] = aligned["code_baseline"]
        data_df.dropna(inplace=True, subset=["baseline"])

        # Compute relative added/modified/removed counts
        data_df["code_percent"] = 100 * data_df["code"] / data_df["baseline"]

        # Plot absolute LoC changed
        show_df = self._prepare_bars(data_df, "total", "code")
        self._do_plot(show_df, self.absolute_loc_plot, "# of lines")

        # Plot relative LoC, ranked by absolute value
        show_df = self._prepare_bars(data_df, "total", "code_percent")
        self._do_plot(show_df, self.percent_loc_sorted_by_absolute_loc_plot, "% of file lines")

        # Plot relative LoC changed
        show_df = self._prepare_bars(data_df, "total_percent", "code_percent")
        self._do_plot(show_df, self.percent_loc_plot, "% of file lines")

        # Plot absolute LoC, ranked by percent change
        show_df = self._prepare_bars(data_df, "total_percent", "code")
        self._do_plot(show_df, self.absolute_loc_sorted_by_percent_loc_plot, "# of lines")

        ftype = data_df.index.get_level_values("file").map(to_file_type)
        # Plot absolute LoC in assembly files, ranked by absolute change
        show_df = self._prepare_bars(data_df.loc[ftype == "assembly"], "total", "code")
        self._do_plot(show_df, self.absolute_loc_assembly, "# of lines")

        # Plot absolute LoC in header files, ranked by absolute change
        show_df = self._prepare_bars(data_df.loc[ftype == "header"], "total", "code")
        self._do_plot(show_df, self.absolute_loc_headers, "# of lines")

        # Plot absolute LoC in source files, ranked by absolute change
        show_df = self._prepare_bars(data_df.loc[ftype == "source"], "total", "code")
        self._do_plot(show_df, self.absolute_loc_sources, "# of lines")

    @output
    def absolute_loc_plot(self):
        return PlotTarget(self, prefix="abs")

    @output
    def percent_loc_sorted_by_absolute_loc_plot(self):
        return PlotTarget(self, prefix="rel-sorted-by-abs")

    @output
    def percent_loc_plot(self):
        return PlotTarget(self, prefix="rel")

    @output
    def absolute_loc_sorted_by_percent_loc_plot(self):
        return PlotTarget(self, prefix="abs-sorted-by-rel")

    @output
    def absolute_loc_assembly(self):
        return PlotTarget(self, prefix="abs-asm")

    @output
    def absolute_loc_headers(self):
        return PlotTarget(self, prefix="abs-hdr")

    @output
    def absolute_loc_sources(self):
        return PlotTarget(self, prefix="abs-src")


class CheriBSDLineChangesByComponent(PlotTask):
    """
    Produce an histogram sorted by the component with highest diff.

    Components are matched by path. This kind of plot is useful as it shows the
    distribution of kernel changes with intermediate granularity.
    File granularity is too confusing, while directory hierarchy may miss the
    relationship between different kernel components.
    This uses the same reporting histogram as cloc-by-file.
    There are two outputs, one using absolute diff numbers, the other using
    the changed LoC percentage with respect with total component lines.
    """
    public = True
    task_namespace = "kernel-history"
    task_name = "cheribsd-cloc-by-component"
    task_config_class = LoCDataConfig

    components = {
        "platform": r"sys/(riscv|arm)",
        "vm": r"sys/vm/(?!uma)",
        "net": r"sys/net",
        "alloc": r"(sys/(vm/uma|kern/.*vmem)|sys/sys/vmem)",
        "dev": r"sys/dev(?!/drm)",
        "kern": r"(sys/kern/(?!.*vmem|vfs)|sys/sys/(?!vmem))",
        "compat": r"sys/compat(?!/freebsd64)",
        "compat64": r"sys/compat/freebsd64",
        "vfs": r"sys/kern/vfs",
        "fs": r"sys/fs",
        "cheri": r"sys/cheri",
        "drm": r"sys/dev/drm",
        "zfs": r"sys/contrib/subrepo-openzfs"
    }

    # Extra filters that are useful to ensure the sanity of the results
    # but should not go in the main plot
    extra_components = {
        "all_platforms": r"sys/(riscv|arm|amd|x86|i386|power)",
    }

    @dependency
    def loc_diff(self):
        return LoadLoCData(self.session, self.analysis_config, task_config=self.config)

    def _filter_component(self, name, filter_, data_df, base_df) -> pd.DataFrame:
        return changed_sloc

    def _do_plot(self, component_map: dict, target: PlotTarget):
        """
        Produce the plot given a set of component filters and a target where to emit the plot
        """
        df = self.loc_diff.diff_df.get()
        df = generalized_xs(df, how="same", complement=True)
        base_df = self.loc_diff.baseline_df.get()

        # Ensure we have the proper theme
        sns.set_theme()
        # Use tab10 colors, added => green, modified => orange, removed => red
        cmap = sns.color_palette(as_cmap=True)
        # ordered as green, orange, red
        colors = [cmap[2], cmap[1], cmap[3]]

        # Produce the dataframe containing counts for each component
        index = pd.MultiIndex.from_product([self.components.keys(), ["added", "modified", "removed"]],
                                           names=["component", "how"])
        data_set = []
        extra_set = []
        all_matches = None
        all_base_matches = None
        for name, filter_ in component_map.items():
            matches = df.index.get_level_values("file").str.match(filter_)
            base_matches = base_df.index.get_level_values("file").str.match(filter_)
            self.logger.debug("Component %s ('%s') matched %d diff entries and %d baseline entries", name, filter_,
                              matches.sum(), base_matches.sum())
            if all_matches is None:
                all_matches = matches
                all_base_matches = base_matches
            else:
                all_matches = (all_matches | matches)
                all_base_matches = (all_base_matches | base_matches)
            changed_sloc = df.loc[matches].groupby("how").sum()
            changed_sloc["component"] = name
            changed_sloc["baseline"] = base_df.loc[base_matches]["code"].sum()
            data_set.append(changed_sloc)
        # Catch-all set
        other = df.loc[~all_matches].groupby("how").sum()
        other["component"] = "other"
        other["baseline"] = base_df.loc[~all_base_matches]["code"].sum()
        data_set.append(other)
        # Finally merge everything
        data_df = pd.concat(data_set)
        data_df["percent"] = 100 * data_df["code"] / data_df["baseline"]

        with new_figure(target.paths()) as fig:
            ax_l, ax_r = fig.subplots(1, 2, sharey=True)
            # Absolute SLoC on the left
            show_df = data_df.reset_index().pivot(index="component", columns="how", values="code")
            show_df.reset_index().plot(x="component",
                                       y=["added", "modified", "removed"],
                                       stacked=True,
                                       kind="barh",
                                       ax=ax_l,
                                       color=colors,
                                       legend=False)
            ax_l.tick_params(axis="y", labelsize="x-small")
            ax_l.set_xlabel("# of lines")

            # Percent SLoC on the right
            show_df = data_df.reset_index().pivot(index="component", columns="how", values="percent")
            show_df.reset_index().plot(x="component",
                                       y=["added", "modified", "removed"],
                                       stacked=True,
                                       kind="barh",
                                       ax=ax_r,
                                       color=colors,
                                       legend=False)
            ax_r.set_xlabel("% of lines")

            # The legend is shared at the top center
            handles, labels = ax_l.get_legend_handles_labels()
            fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))

    def run_plot(self):
        self._do_plot(self.components, self.plot)
        debug_components = dict(self.components)
        debug_components.update(self.extra_components)
        self._do_plot(debug_components, self.debug_plot)

    @output
    def plot(self):
        return PlotTarget(self)

    @output
    def debug_plot(self):
        return PlotTarget(self, prefix="debug")
