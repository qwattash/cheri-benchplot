import re
from dataclasses import dataclass

import polars as pl

from ..core.analysis import AnalysisTask
from ..core.artefact import DataFrameLoadTask, RemoteBenchmarkIterationTarget
from ..core.config import Config, ConfigPath, config_field
from ..core.task import ExecutionTask, output


class C18nKtraceLoader(DataFrameLoadTask):
    task_namespace = "c18n"
    task_name = "ktrace-load"

    def _load_chunks(self, src_file, chunk_size):
        transition_regex = re.compile(
            r"(?P<pid>[0-9]+) .*RTLD: c18n: (?P<caller>[\w./<>+-\[\]]+) -> (?P<callee>[\w./<>+-\[\]]+) "
            r"at \[(?P<symbol_number>[0-9]+)\] (?P<symbol>[\w<>-]+)")

        chunk = []
        for line in src_file:
            line = line.strip()
            if not line:
                continue
            m = transition_regex.match(line)
            if not m:
                continue
            chunk.append(m.groupdict())
            if len(chunk) >= chunk_size:
                yield pl.DataFrame(chunk)
                chunk.clear()

        if chunk:
            yield pl.DataFrame(chunk)

    def _load_one(self, path: "Path") -> pl.DataFrame:
        with open(path, "r") as src:
            df = pl.concat(self._load_chunks(src, chunk_size=100000))
        return df


@dataclass
class C18nKtraceConfig(Config):
    """
    Configure ktrace for c18n user probes.
    """
    c18n_utrace_enable: bool = config_field(True, desc="Enable or disable c18n tracing")


class C18nKtraceExec(ExecutionTask):
    """
    Add-on task that instruments a benchmark to run under ktrace with
    c18n user probes.
    """
    public = True
    task_namespace = "c18n"
    task_name = "ktrace"
    task_config_class = C18nKtraceConfig

    @output
    def trace_data(self):
        return RemoteBenchmarkIterationTarget(self, "c18n-trace", ext="txt", loader=C18nKtraceLoader)

    def run(self):
        self.script.extend_context({
            "c18n_utrace_config": self.config,
            "c18n_utrace_gen_output_path": self.trace_data.shell_path_builder()
        })
