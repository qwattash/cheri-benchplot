import json
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from marshmallow import ValidationError, validates

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
    scenario_file: ConfigPath = config_field(
        Config.REQUIRED,
        desc="Scenario configuration to run. Note that this must be imported into assets first, "
        "as the path is assumed to be relative to the assets directory.")
    qps_path: ConfigPath = config_field(None, desc="Path of grpc qps binaries")
    client_procctl_args: list[str] = config_field(list, desc="proccontrol arguments for the client worker")
    client_cpu: list[int | str] = config_field(list, desc="Restrict client worker to the given CPUs")
    server_procctl_args: list[str] = config_field(list, desc="proccontrol arguments for the server worker")
    server_cpu: list[int | str] = config_field(list, desc="Restrict server worker to the given CPUs")
    driver_cpu: list[int | str] = config_field(list, desc="Restrict driver to the given CPUs")

    @validates("scenario_file")
    def check_scenario(self, data, **kwargs):
        if Path(data).is_absolute():
            raise ValidationError("Must be a relative path to the assets directory")


class QpsExecTask(ExecutionTask):
    """
    Execute the GRPC QPS benchmark with the given scenario file.

    This will run two qps_worker processes, which may be pinned to different CPUs.
    The qps_json_driver is used to drive the benchmark.

    Integration with pmc is supported, where measurement is done on the server worker.
    When system-mode pmc is enabled, care must be taken to correctly pin workers to CPUs
    to isolate the counters effects.

    The scenario supports templating. The :attr:`QpsExecConfig.scenario_file` is loaded and
    parameterization values are substituted according to Python :func:`format()`.
    """
    task_namespace = "qps"
    task_name = "exec"
    public = True
    task_config_class = QpsExecConfig

    @output
    def qps_driver_output(self):
        return RemoteBenchmarkIterationTarget(self, "qps", ext="json", loader=LoadQpsData)

    def _coerce(self, fmt_spec, value):
        try:
            match fmt_spec:
                case "d":
                    return int(value)
                case "s":
                    return str(value)
                case _:
                    return value
        except ValueError:
            return None

    def _bind_scenario(self, data):
        """
        Recursively substitute template parameters
        """
        if type(data) is dict:
            for k, v in data.items():
                data[k] = self._bind_scenario(v)
        elif type(data) is list:
            data = [self._bind_scenario(v) for v in data]
        elif type(data) is str:
            if m := re.match(r"\{([a-zA-Z0-9_-]+)(:[ds])?\}", data):
                key = m.group(1)
                fmt_spec = m.group(2)
                if fmt_spec:
                    fmt_spec = fmt_spec.strip(":")
                value = self.benchmark.parameters.get(key)
                value = self._coerce(fmt_spec, value)
                if value is None:
                    return data
                data = data.format_map({key: value})
                # Try to coerce again, in case we have something like "10"
                # that needs to turn into an integer.
                if res := self._coerce(fmt_spec, data):
                    return res
        return data

    def run(self):
        scenario_name = "qps_scenario.json"
        self.script.set_template("grpc.sh.jinja")
        self.script.extend_context({
            "qps_config": self.config,
            "qps_gen_output": self.qps_driver_output.shell_path_builder(),
            "qps_scenario": scenario_name
        })

        # Generate the scenario configuration. Note that this goes in the current
        # data output directory, alongside the runner script.
        scenario_file = self.benchmark.get_benchmark_data_path() / scenario_name
        with open(self.session.get_asset_root_path() / self.config.scenario_file) as src:
            scenario_template = json.load(src)
        scenario = self._bind_scenario(scenario_template)
        with open(scenario_file, "w+") as dst:
            json.dump(scenario, dst)
