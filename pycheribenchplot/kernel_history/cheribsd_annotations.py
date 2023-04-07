import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pycheribenchplot.core.analysis import AnalysisTask, BenchmarkDataLoadTask
from pycheribenchplot.core.artefact import DataFrameTarget, LocalFileTarget
from pycheribenchplot.core.config import Config
from pycheribenchplot.core.plot import PlotTask, new_figure
from pycheribenchplot.core.task import DataGenTask
from pycheribenchplot.core.util import resolve_system_command
from pycheribenchplot.kernel_history.cdb import CompilationDBConfig
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

    def dependencies(self):
        self._compilationdb = CheriBSDCompilationDB(self.benchmark, self.script, task_config=self.config.compilationdb)
        yield self._compilationdb

    def run(self):
        self.logger.debug("Extracting kernel changes")
        if not self._cheribsd_changes_awkscript.exists():
            self.logger.error("Can not extract cheribsd kernel changes, missing awk script %s",
                              self._cheribsd_changes_awkscript)
            raise RuntimeError("Can not extract kernel changes")

        cdb_df = pd.read_json(self._compilationdb.output_map["compilation-db"].path)
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
        df.set_index(["file"], inplace=True)

        df.to_json(self.output_map["changes"].path, orient="table", index=True)

    def outputs(self):
        yield "changes", LocalFileTarget.from_task(self, ext="json")
        # re-export dependency output
        yield "compilation-db", self._compilationdb.output_map["compilation-db"]


class CheriBSDFilesLoad(BenchmarkDataLoadTask):
    task_namespace = "kernel-history"
    task_name = "cheribsd-files-load"
    exec_task = CheriBSDKernelFileChanges
    target_key = "changes"
    model = RawFileChangesModel

    def _load_one_json(self, path, **kwargs):
        kwargs["orient"] = "table"
        return super()._load_one_json(path, **kwargs)


class CompilationDBLoad(BenchmarkDataLoadTask):
    task_namespace = "kernel-history"
    task_name = "cheribsd-cdb-load"
    exec_task = CheriBSDKernelFileChanges
    target_key = "compilation-db"
    model = CompilationDBModel


class CheriBSDAllFileChanges(AnalysisTask):
    task_namespace = "kernel-history"
    task_name = "cheribsd-files-union"

    def dependencies(self):
        self._changes_load = []
        self._cdb_load = []
        for ctx in self.session.benchmark_matrix.to_numpy().ravel():
            task = CheriBSDFilesLoad(ctx, self.analysis_config)
            self._changes_load.append(task)
            yield task
            task = CompilationDBLoad(ctx, self.analysis_config)
            self._cdb_load.append(task)
            yield task

    def run(self):
        # Files changed
        df_set = []
        # Compilation db files
        cdb_set = []

        for loader in self._changes_load:
            df_set.append(loader.output_map["df"].get())
        for loader in self._cdb_load:
            cdb_set.append(loader.output_map["df"].get())

        merged = pd.concat(df_set)
        # Ensure that the changes columns are hashable
        merged["changes"] = merged["changes"].map(sorted).map(tuple)
        merged["changes_purecap"] = merged["changes_purecap"].map(sorted).map(tuple)
        df = merged.groupby(level="file").first()
        # Verify that the groups are consistent, otherwise we have invalid input data
        check = (merged.groupby(level="file").nunique() <= 1).all().all()
        assert check, "Invalid file changes data, conflicting changes tags found"

        cdb_df = pd.concat(cdb_set).groupby("files").first().reset_index()

        self.output_map["df"].assign(df)
        self.output_map["compilation-db"].assign(cdb_df)

    def outputs(self):
        yield "df", DataFrameTarget(AllFileChangesModel.to_schema(self.session))
        yield "compilation-db", DataFrameTarget(AllCompilationDBModel.to_schema(self.session))


class CheriBSDChangesByType(PlotTask):
    """
    Generate a bar plot with the number files containing changes of a given type.
    """
    public = True
    task_namespace = "kernel-history"
    task_name = "cheribsd-changes-plot"

    def dependencies(self):
        self._files = CheriBSDAllFileChanges(self.session, self.analysis_config)
        yield self._files

    def run(self):
        df = self._files.output_map["df"].get()
        cdb_df = self._files.output_map["compilation-db"].get()

        df["all_changes"] = (df["changes"] + df["changes_purecap"]).map(set)
        changed_files = len(df)
        total_files = len(cdb_df)

        count_label = "Number of Files"

        data_df = df.explode("all_changes")
        data_df = data_df.groupby("all_changes").size().to_frame(name=count_label).reset_index()
        data_df["% of changed files"] = data_df[count_label] * 100 / changed_files
        data_df = data_df.rename({"all_changes": "Type of change"}, axis=1)
        data_df["% of all files"] = data_df[count_label] * 100 / total_files

        with new_figure(self.output_map["changes"].path) as fig:
            ax = fig.subplots()
            sns.barplot(ax=ax, data=data_df, x="Type of change", y=count_label, color="steelblue")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")

        with new_figure(self.output_map["rel-changes"].path) as fig:
            ax = fig.subplots()
            sns.barplot(ax=ax, data=data_df, x="Type of change", y="% of changed files", color="steelblue")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")

        with new_figure(self.output_map["rel-total-changes"].path) as fig:
            ax = fig.subplots()
            sns.barplot(ax=ax, data=data_df, x="Type of change", y="% of all files", color="steelblue")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")

    def outputs(self):
        yield "changes", self._plot_output("changes")
        yield "rel-changes", self._plot_output("rel-changes")
        yield "rel-total-changes", self._plot_output("rel-total-changes")
