from dataclasses import dataclass

from ..core.artefact import RemoteBenchmarkIterationTarget
from ..core.config import Config, ConfigPath, config_field
from ..core.task import ExecutionTask, output


@dataclass
class QPSExecConfig(Config):
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


class QPSExecTask(ExecutionTask):
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
    task_config_class = QPSExecConfig

    @output
    def qps_driver_output(self):
        return RemoteBenchmarkIterationTarget(self, "qps", ext="json")

    def run(self):
        self.script.set_template("grpc.sh.jinja")
        self.script.extend_context({
            "qps_config": self.config,
            "qps_gen_output": self.qps_driver_output.shell_path_builder()
        })
