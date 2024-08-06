from dataclasses import MISSING, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

from ..core.artefact import PLDataFrameLoadTask, RemoteBenchmarkIterationTarget
from ..core.config import Config, ConfigPath, config_field
from ..core.task import output
from ..core.tvrs import TVRSExecTask


@dataclass
class UnixBenchScenario(Config):
    test_name: str = config_field(MISSING, desc="Unixbench test to run")
    duration: int = config_field(10000, desc="Duration of an iteration (number of transactions)")
    args: List[str] = config_field(list, desc="Extra test arguments")


@dataclass
class UnixBenchConfig(Config):
    scenario_path: Optional[ConfigPath] = config_field(Path("pycheribenchplot/unixbench/scenarios"),
                                                       desc="Path to unixbench scenarios")
    unixbench_path: Optional[ConfigPath] = config_field(None, desc="Path of unixbench in the remote host")
    scenarios: Dict[str, UnixBenchScenario] = config_field(
        dict, desc="Inline scenarios. The key must match the name given by the `scenario` parameter")


class IngestUnixBenchStats(PLDataFrameLoadTask):
    """
    Loader for stats data that produces a standard polars dataframe.
    """
    task_namespace = "unixbench"
    task_name = "ingest-stats"

    def _load_one(self, path: Path) -> pl.DataFrame:
        """
        Load data for a benchmark run from the given target file.

        We only care about the actual time sample. Since we run individual
        iterations under hyperfine, this will be found in the "times" list.
        We also check the exit codes, to make sure there were no issues.
        """
        in_df = pl.read_json(path).with_columns(
            pl.col("results").list.first().struct.field("times"),
            pl.col("results").list.first().struct.field("exit_codes"))
        df = in_df.select(
            pl.col("results").list.first().struct.field("user"),
            pl.col("results").list.first().struct.field("system"),
            pl.col("times").list.first(),
            pl.col("times").list.len().alias("check_ntimes"),
            pl.col("exit_codes").list.eval(pl.element() != 0).list.any().alias("check_errs"))
        if (df["check_ntimes"] > 1).any():
            self.logger.error("Unexpected number of hyperfine iterations")
            raise RuntimeError("Input data error")

        if df["check_errs"].any():
            self.logger.error("Found non-zero exit codes for %s at %s", self.benchmark, path)
            raise RuntimeError("Input data error")

        return df.select("user", "system", "times")


class UnixBenchExec(TVRSExecTask):
    """
    Run unixbench tests.

    Note that this assumes a patched unixbench version that allows to set
    a fixed workload instead of a time-dependent one.

    The test to run is taken from the `scenario` parameter, which must
    point to a valid UnixBenchScenario configuration file.
    """
    task_namespace = "unixbench"
    task_name = "exec"
    task_config_class = UnixBenchConfig
    public = True
    scenario_config_class = UnixBenchScenario

    @output
    def timing(self):
        return RemoteBenchmarkIterationTarget(self, "timing", loader=IngestUnixBenchStats, ext="json")

    def run(self):
        super().run()
        self.script.set_template("unixbench.sh.jinja")
        self.script.extend_context({
            "unixbench_config": self.config,
            "unixbench_gen_output_path": self.timing.shell_path_builder()
        })
