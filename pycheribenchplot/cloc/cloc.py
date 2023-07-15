from dataclasses import dataclass, field

import pandas as pd
import seaborn as sns

from ..compile_db import CompilationDB, CompilationDBModel
from ..core.analysis import AnalysisTask
from ..core.artefact import AnalysisFileTarget, DataFrameTarget
from ..core.config import Config
from ..core.model import check_data_model
from ..core.pandas_util import generalized_xs
from ..core.plot import PlotTarget, PlotTask, new_figure
from ..core.task import dependency, output
from .cheribsd import ExtractLoCCheriBSD
from .cloc_generic import ExtractLoCMultiRepo
from .model import LoCCountModel, LoCDiffModel


@dataclass
class LoCAnalysisConfig(Config):
    pass


class LoadLoCGenericData(AnalysisTask):
    """
    Load line of code changes by file.
    This is the place to configure filtering based on the compilation DB or other
    criteria, so that all dependent tasks operate on the filtered data normally.

    Note that this the central loader for all LoC data. This allows the consumers
    downstream to ignore the source of the data and just plot/process it.
    The only requirement is that the data input must conform to the
    LoCDiffModel and LoCCountModel.
    """
    task_namespace = "cloc"
    task_name = "load-loc-generic-data"
    task_config_class = LoCAnalysisConfig

    @dependency(optional=True)
    def compilation_db(self):
        """
        Load all compilation databases.

        Note that there is a cdb for each instance, as the instance
        already parameterizes both the kernel ABI and the user world ABI.
        """
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(CompilationDB)
            yield from (t.get_loader() for t in task.cdb)

    @dependency(optional=True)
    def cloc_diff(self):
        task = self.session.find_exec_task(ExtractLoCMultiRepo)
        return [tgt.get_loader() for tgt in task.cloc_diff]

    @dependency(optional=True)
    def cloc_baseline(self):
        task = self.session.find_exec_task(ExtractLoCMultiRepo)
        return [tgt.get_loader() for tgt in task.cloc_baseline]

    @dependency(optional=True)
    def cloc_cheribsd_diff(self):
        task = self.session.find_exec_task(ExtractLoCCheriBSD)
        return task.cloc_diff.get_loader()

    @dependency(optional=True)
    def cloc_cheribsd_baseline(self):
        task = self.session.find_exec_task(ExtractLoCCheriBSD)
        return task.cloc_baseline.get_loader()

    @check_data_model
    def _load_compilation_db(self) -> CompilationDBModel:
        """
        Fetch all compilation DBs and merge them
        """
        cdb_set = []
        for loader in self.compilation_db:
            cdb_set.append(loader.df.get())
        return pd.concat(cdb_set).groupby("file").first().reset_index()

    def run(self):
        # Merge the data from inputs
        all_diff = []
        all_baseline = []
        if self.cloc_diff:
            all_diff += [loader.df.get() for loader in self.cloc_diff]
            all_baseline = [loader.df.get() for loader in self.cloc_baseline]
        if self.cloc_cheribsd_diff:
            all_diff += [self.cloc_cheribsd_diff.df.get()]
            all_baseline += [self.cloc_cheribsd_baseline.df.get()]

        if not all_diff or not all_baseline:
            self.logger.error("No data to consume")
            raise RuntimeError("Empty input")
        if len(all_diff) != len(all_baseline):
            self.logger.error("Must have the same number of Diff and Baseline LoC dataframes. " +
                              "Something is very wrong with dependencies.")
            raise RuntimeError("Dependency mismatch")

        diff_df = pd.concat(all_diff)
        baseline_df = pd.concat(all_baseline)

        if self.compilation_db:
            cdb = self._load_compilation_db()
            # Filter by compilation DB files
            filtered_diff_df = diff_df.loc[diff_df.index.isin(cdb["file"], level="file")]
            filtered_baseline_df = baseline_df.loc[baseline_df.index.isin(cdb["file"], level="file")]
            self.cdb_diff_df.assign(filtered_diff_df)
            self.cdb_baseline_df.assign(filtered_baseline_df)
        self.diff_df.assign(diff_df)
        self.baseline_df.assign(baseline_df)

    @output
    def diff_df(self):
        return DataFrameTarget(self, LoCDiffModel, output_id="all-diff")

    @output
    def baseline_df(self):
        return DataFrameTarget(self, LoCCountModel, output_id="all-baseline")

    @output
    def cdb_diff_df(self):
        return DataFrameTarget(self, LoCDiffModel, output_id="cdb-diff")

    @output
    def cdb_baseline_df(self):
        return DataFrameTarget(self, LoCCountModel, output_id="cdb-baseline")


class ReportLoCGeneric(PlotTask):
    """
    Produce a plot of CLoC changed for each repository
    """
    public = True
    task_namespace = "cloc"
    task_name = "plot-deltas"
    task_config_class = LoCAnalysisConfig

    @dependency
    def loc_data(self):
        return LoadLoCGenericData(self.session, self.analysis_config, task_config=self.config)

    def run_plot(self):
        df = self.loc_data.diff_df.get()
        df = generalized_xs(df, how="same", complement=True)
        base_df = self.loc_data.baseline_df.get()

        # Ensure we have the proper theme
        sns.set_theme()
        # Use tab10 colors, added => green, modified => orange, removed => red
        cmap = sns.color_palette(as_cmap=True)
        # ordered as green, orange, red
        colors = [cmap[2], cmap[1], cmap[3]]

        # Aggregate counts by repo
        base_counts_df = base_df["code"].groupby(["repo"]).sum()
        agg_df = df.groupby(["repo", "how"]).sum().join(base_counts_df, on="repo", rsuffix="_baseline")
        agg_df["code_pct"] = 100 * agg_df["code"] / agg_df["code_baseline"]

        agg_df.to_csv(self.raw_data.path)

        with new_figure(self.plot.paths()) as fig:
            ax_l, ax_r = fig.subplots(1, 2, sharey=True)
            # Absolute SLoC on the left
            show_df = agg_df.reset_index().pivot(index="repo", columns="how", values="code")
            show_df.reset_index().plot(x="repo",
                                       y=["added", "modified", "removed"],
                                       stacked=True,
                                       kind="barh",
                                       ax=ax_l,
                                       color=colors,
                                       legend=False)
            ax_l.tick_params(axis="y", labelsize="x-small")
            ax_l.set_xlabel("# of lines")

            # Percent SLoC on the right
            show_df = agg_df.reset_index().pivot(index="repo", columns="how", values="code_pct")
            show_df.reset_index().plot(x="repo",
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
        return PlotTarget(self)

    @output
    def raw_data(self):
        return AnalysisFileTarget(self, prefix="raw", ext="csv")
