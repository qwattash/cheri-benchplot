import json
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..core.artefact import PLDataFrameLoadTask, RemoteBenchmarkIterationTarget
from ..core.config import Config, ConfigPath, config_field
from ..core.task import ExecutionTask, output


def to_snake_case(name):
    """
    Convert column names from camelCase to snake_case
    """
    return re.sub(r"(?<=[a-z])(?=[A-Z])", "_", name).lower()


class LoadQpsData(PLDataFrameLoadTask):
    """
    Load and merge data from QPS runs.
    """
    task_namespace = "qps"
    task_name = "ingest"

    COLUMN_NAMES = [
        "qps",
        "message_count",
        "elapsed_time",
        "latency50",
        "latency90",
        "latency95",
        "latency99",
        "latency99_9",
        "req_size",
        "res_size",
    ]

    COLUMN_RENAME = {
        "latency999": "latency99_9",
        "resp_size": "res_size",
    }

    def _load_one(self, path: Path) -> pl.DataFrame:
        """
        Load data for a benchmark run for the given target file.

        This loads the QPS and latency metrics, along with a number
        of scenario parameters that are propagated to the output by
        the QPS driver.

        Columns:
         - qps: the QPS metric
         - message_count: Number of exchanged messages
         - elapsed_time: Benchmark time (s)
         - latency50: 50th percentile latency (ns)
         - latency90: 90th percentile latency (ns)
         - latency95: 95th percentile latency (ns)
         - latency99: 99th percentile latency (ns)
         - latency99_9: 99.9th percentile latency (ns)
         - req_size: request size (from scenario)
         - res_size: response size (from scenario)
        """
        self.logger.debug("Ingest QPS data for %s", path)
        with open(path, "r") as data_file:
            data = json.load(data_file)
        payload_config = data["scenario"]["clientConfig"].get("payloadConfig", {})

        params = {"reqSize": 0, "respSize": 0}
        payload_params = payload_config.get("simpleParams", {})
        params.update(payload_params)

        row = {**data["summary"], **params}
        df = pl.DataFrame(row).rename(to_snake_case).rename(self.COLUMN_RENAME)
        return df.select(self.COLUMN_NAMES)


@dataclass
class QpsExecConfig(Config):
    """
    Configure the QPS benchmark.
    """
    scenario_file: ConfigPath = config_field(Config.REQUIRED, desc="Scenario configuration to run")
    qps_path: ConfigPath = config_field(None, desc="Path of grpc qps binaries")
    client_procctl_args: list[str] = config_field(list, desc="proccontrol arguments for the client worker")
    client_cpu: list[int] = config_field(list, desc="Restrict client worker to the given CPUs")
    server_procctl_args: list[str] = config_field(list, desc="proccontrol arguments for the server worker")
    server_cpu: list[int] = config_field(list, desc="Restrict server worker to the given CPUs")
    driver_cpu: list[int] = config_field(list, desc="Restrict driver to the given CPUs")


class QpsExecTask(ExecutionTask):
    """
    Execute the GRPC QPS benchmark with the given scenario file.

    This will run two qps_worker processes, which may be pinned to different CPUs.
    The qps_json_driver is used to drive the benchmark.

    Integration with pmc is supported, where measurement is done on the server worker.
    When system-mode pmc is enabled, care must be taken to correctly pin workers to CPUs
    to isolate the counters effects.
    """
    task_namespace = "qps"
    task_name = "exec"
    public = True
    task_config_class = QpsExecConfig

    @output
    def qps_driver_output(self):
        return RemoteBenchmarkIterationTarget(self, "qps", ext="json", loader=LoadQpsData)

    def run(self):
        self.script.set_template("grpc.sh.jinja")
        self.script.extend_context({
            "qps_config": self.config,
            "qps_gen_output": self.qps_driver_output.shell_path_builder()
        })
