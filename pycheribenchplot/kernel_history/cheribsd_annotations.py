import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns

from pycheribenchplot.core.analysis import AnalysisTask
from pycheribenchplot.core.artefact import DataFrameTarget, LocalFileTarget
from pycheribenchplot.core.config import Config
from pycheribenchplot.core.plot import PlotTarget, PlotTask, new_figure
from pycheribenchplot.core.task import DataGenTask, dependency, output
from pycheribenchplot.core.util import resolve_system_command
from pycheribenchplot.kernel_history.cdb import (CheriBSDCompilationDB, CompilationDBConfig)
from pycheribenchplot.kernel_history.model import (AllCompilationDBModel, AllFileChangesModel, CompilationDBModel,
                                                   RawFileChangesModel)


@dataclass
class KernelFileChangesConfig(Config):
    compilationdb: Optional[CompilationDBConfig] = field(default_factory=CompilationDBConfig)


class CheriBSDKernelFileChanges(DataGenTask):
    """
    This task extracts cheribsd kernel changes tags using the tooling in cheribsd.
    For this to work, the cheribsd source tree must be synchronized to the kernel version we are running.

    Note that this is done for each benchmark in the benchmark matrix, because we want to extract
    separate information about each kernel configuration.
    The data can always be aggregated afterwards because we retain the source file of the annotations.
    """
    public = True
    task_namespace = "kernel-history"
    task_name = "cheribsd-changes"
    task_config_class = KernelFileChangesConfig

    def __init__(self, benchmark, script, task_config=None):
        super().__init__(benchmark, script, task_config=task_config)
        awkscript = Path("tools") / "tools" / "cheri-changes" / "extract-cheri-changes.awk"
        #: The awk script that generates the cheri-changes json output
        self._cheribsd_changes_awkscript = self.session.user_config.cheribsd_path / awkscript
        self._awk = resolve_system_command("awk")

    def _decode_changes(self, raw_data: str) -> list:
        decoder = json.JSONDecoder()
        chunks = []
        while raw_data:
            data, idx = decoder.raw_decode(raw_data)
            chunks.extend(data)
            raw_data = raw_data[idx:].strip()
        return chunks

    @dependency
    def cheribuild_trace(self):
        return CheriBSDCompilationDB(self.benchmark, self.script, task_config=self.config.compilationdb)

    def run(self):
        self.logger.debug("Extracting kernel changes")
        if not self._cheribsd_changes_awkscript.exists():
            self.logger.error("Can not extract cheribsd kernel changes, missing awk script %s",
                              self._cheribsd_changes_awkscript)
            raise RuntimeError("Can not extract kernel changes")

        cdb_df = pd.read_json(self.cheribuild_trace.compilation_db.path)
        kernel_files = [Path(entry) for entry in cdb_df["files"]]

        # For each of the files, we extract the changes annotations, if any
        # We chunk the list in groups of 20 files to lower the number of calls to awk
        chunks = np.array_split(np.array(kernel_files), len(kernel_files) / 20)
        json_chunks = []
        awk_cmd = [self._awk, "-f", self._cheribsd_changes_awkscript]
        for chunk in chunks:
            cmd = awk_cmd + [self.session.user_config.cheribsd_path / p for p in chunk]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                self.logger.error("Failed to extract CheriBSD changes from %s: %s", chunk, result.stderr)
                raise RuntimeError("Failed to run awk")
            json_chunks.extend(self._decode_changes(result.stdout.decode("utf-8")))

        # Build a dataframe that we can verify
        df = pd.DataFrame.from_records(json_chunks)

        df["changes"] = df.get("changes", None).map(lambda v: v if isinstance(v, list) else [])
        df["changes_purecap"] = df.get("changes_purecap", None).map(lambda v: v if isinstance(v, list) else [])
        df["hybrid_specific"] = df.get("hybrid_specific", False).fillna(False)

        df.to_json(self.changes.path, index=False)

    @output
    def changes(self):
        return LocalFileTarget(self, ext="json", model=RawFileChangesModel)

    @output
    def compilation_db(self):
        return self.cheribuild_trace.compilation_db


class CheriBSDAllFileChanges(AnalysisTask):
    task_namespace = "kernel-history"
    task_name = "cheribsd-files-union"

    @dependency
    def load_files(self):
        for b in self.session.all_benchmarks():
            exec_task = b.load_exec_task(CheriBSDKernelFileChanges)
            yield exec_task.changes.get_loader()

    @dependency
    def load_cdb(self):
        for b in self.session.all_benchmarks():
            exec_task = b.load_exec_task(CheriBSDKernelFileChanges)
            yield exec_task.compilation_db.get_loader()

    def run(self):
        # Files changed
        df_set = []
        # Compilation db files
        cdb_set = []

        for loader in self.load_files:
            df_set.append(loader.df.get())
        for loader in self.load_cdb:
            cdb_set.append(loader.df.get())

        merged = pd.concat(df_set)
        # Ensure that the changes columns are hashable
        merged["changes"] = merged["changes"].map(sorted).map(tuple)
        merged["changes_purecap"] = merged["changes_purecap"].map(sorted).map(tuple)
        df = merged.groupby(level="file").first()
        # Verify that the groups are consistent, otherwise we have invalid input data
        check = (merged.groupby(level="file").nunique() <= 1).all().all()
        assert check, "Invalid file changes data, conflicting changes tags found"

        cdb_df = pd.concat(cdb_set).groupby("files").first().reset_index()

        self.df.assign(df)
        self.compilation_db.assign(cdb_df)

    @output
    def df(self):
        return DataFrameTarget(self, AllFileChangesModel)

    @output
    def compilation_db(self):
        return DataFrameTarget(self, AllCompilationDBModel)


class CheriBSDChangesByType(PlotTask):
    """
    Generate a bar plot with the number files containing changes of a given type.
    """
    public = True
    task_namespace = "kernel-history"
    task_name = "cheribsd-changes-plot"

    @dependency
    def merged_annotations(self):
        return CheriBSDAllFileChanges(self.session, self.analysis_config)

    def run_plot(self):
        df = self.merged_annotations.df.get()
        cdb_df = self.merged_annotations.compilation_db.get()

        df["all_changes"] = (df["changes"] + df["changes_purecap"]).map(set)
        changed_files = len(df)
        total_files = len(cdb_df)

        count_label = "Number of Files"

        data_df = df.explode("all_changes")
        data_df = data_df.groupby("all_changes").size().to_frame(name=count_label).reset_index()
        data_df["% of changed files"] = data_df[count_label] * 100 / changed_files
        data_df = data_df.rename({"all_changes": "Type of change"}, axis=1)
        data_df["% of all files"] = data_df[count_label] * 100 / total_files

        sns.set_theme()

        with new_figure(self.changes.paths()) as fig:
            ax = fig.subplots()
            sns.barplot(ax=ax, data=data_df, x="Type of change", y=count_label, color="steelblue")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")

        with new_figure(self.rel_changes.paths()) as fig:
            ax = fig.subplots()
            sns.barplot(ax=ax, data=data_df, x="Type of change", y="% of changed files", color="steelblue")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")

        with new_figure(self.rel_total_changes.paths()) as fig:
            ax = fig.subplots()
            sns.barplot(ax=ax, data=data_df, x="Type of change", y="% of all files", color="steelblue")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")

    @output
    def changes(self):
        return PlotTarget(self, prefix="changes")

    @output
    def rel_changes(self):
        return PlotTarget(self, prefix="rel-changes")

    @output
    def rel_total_changes(self):
        return PlotTarget(self, prefix="rel-total-changes")
