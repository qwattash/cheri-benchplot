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

from pycheribenchplot.core.analysis import AnalysisTask, BenchmarkDataLoadTask
from pycheribenchplot.core.artefact import (AnalysisFileTarget, DataFrameTarget, LocalFileTarget)
from pycheribenchplot.core.config import Config
from pycheribenchplot.core.pandas_util import generalized_xs
from pycheribenchplot.core.plot import PlotTarget, PlotTask, new_figure
from pycheribenchplot.core.plot_util.mosaic import mosaic_treemap
from pycheribenchplot.core.task import (DataGenTask, SessionDataGenTask, dependency, output)
from pycheribenchplot.core.util import SubprocessHelper, resolve_system_command

from .cdb import CheriBSDCompilationDB
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


@dataclass
class LoCDataConfig(Config):
    #: Filter the diff based on the files found in the compilation DB
    restrict_to_compilation_db: bool = False


class CompilationDBLoad(BenchmarkDataLoadTask):
    task_namespace = "kernel-history"
    task_name = "cheribsd-loc-cdb-load"
    exec_task = CheriBSDCompilationDB
    target_key = "compilation-db"
    model = CompilationDBModel


class LoadLoCData(AnalysisTask):
    """
    Load line of code changes by file.
    This is the place to configure filtering based on the compilation DB or other
    criteria, so that all dependent tasks operate on the filtered data normally.
    """
    task_namespace = "kernel-history"
    task_name = "load-loc-data"
    task_config_class = LoCDataConfig

    def dependencies(self):
        yield from super().dependencies()
        if self.config.restrict_to_compilation_db:
            self._cdb_load = []
            for ctx in self.session.benchmark_matrix.to_numpy().ravel():
                task = CompilationDBLoad(ctx, self.analysis_config)
                self._cdb_load.append(task)
                yield task

    def _load_compilation_db(self):
        """
        Fetch all compilation DBs and merge them
        """
        cdb_set = []
        for loader in self._cdb_load:
            cdb_set.append(loader.output_map["df"].get())
        df = pd.concat(cdb_set).groupby("files").first().reset_index()
        return AllCompilationDBModel.to_schema(self.session).validate(df)

    def run(self):
        # Should have a way to properly fetch the configured datagen task here
        cloc_task = CheriBSDKernelLineChanges(self.session, task_config=CheriBSDKernelLineChangesConfig())

        diff_df = pd.read_json(cloc_task.cloc.path, orient="table")
        baseline_df = pd.read_json(cloc_task.cloc_baseline.path, orient="table")
        if self.config.restrict_to_compilation_db:
            cdb = self._load_compilation_db()
            # Filter by compilation DB files
            diff_df = diff_df.loc[diff_df.index.isin(cdb["files"], level="file")]
            baseline_df = baseline_df.loc[baseline_df.index.isin(cdb["files"], level="file")]

        self.diff_df.assign(diff_df)
        self.baseline_df.assign(baseline_df)

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
        return PlotTarget.from_task(self)


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
        return PlotTarget.from_task(self, prefix="abs")

    @output
    def percent_loc_sorted_by_absolute_loc_plot(self):
        return PlotTarget.from_task(self, prefix="rel-sorted-by-abs")

    @output
    def percent_loc_plot(self):
        return PlotTarget.from_task(self, prefix="rel")

    @output
    def absolute_loc_sorted_by_percent_loc_plot(self):
        return PlotTarget.from_task(self, prefix="abs-sorted-by-rel")

    @output
    def absolute_loc_assembly(self):
        return PlotTarget.from_task(self, prefix="abs-asm")

    @output
    def absolute_loc_headers(self):
        return PlotTarget.from_task(self, prefix="abs-hdr")

    @output
    def absolute_loc_sources(self):
        return PlotTarget.from_task(self, prefix="abs-src")


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
        "platform": r"sys/(riscv|arm|amd|x86|i386|power)",
        "vm": r"sys/vm/(?!uma)",
        "net": r"sys/net",
        "alloc": r"(sys/(vm/uma|kern/.*vmem)|sys/sys/vmem)",
        "dev": r"sys/dev",
        "kern": r"(sys/kern/(?!.*vmem|vfs)|sys/sys/(?!vmem))",
        "compat": r"sys/compat",
        "vfs": r"sys/kern/vfs",
        "fs": r"sys/fs",
        "cheri": r"sys/cheri"
    }

    @dependency
    def loc_diff(self):
        return LoadLoCData(self.session, self.analysis_config, task_config=self.config)

    def run_plot(self):
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
        all_matches = None
        all_base_matches = None
        for name, filter_ in self.components.items():
            matches = df.index.get_level_values("file").str.match(filter_)
            base_matches = base_df.index.get_level_values("file").str.match(filter_)
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

        with new_figure(self.plot.path) as fig:
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

    @output
    def plot(self):
        return PlotTarget.from_task(self)
