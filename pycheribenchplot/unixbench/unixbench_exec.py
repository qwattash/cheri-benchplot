from dataclasses import dataclass
from enum import Enum

from ..core.config import Config, ConfigPath, config_field
from ..generic.timing import TimingConfig, TimingExecTask


class UnixBenchTest(Enum):
    """
    Supported unixbench tests
    """

    SYSCALL = "syscall"


@dataclass
class UnixBenchConfig(TimingConfig):
    test_name: UnixBenchTest = config_field(
        Config.REQUIRED, by_value=True, desc="Unixbench test to run"
    )
    unixbench_path: ConfigPath | None = config_field(
        None, desc="Path of unixbench in the remote host"
    )
    duration: int = config_field(
        10000, desc="Duration of an iteration (number of transactions)"
    )


class UnixBenchExec(TimingExecTask):
    """
    Run unixbench tests.

    Note that this assumes a patched unixbench version that allows to set
    a fixed workload instead of a time-dependent one.
    """

    task_namespace = "unixbench"
    task_name = "exec"
    task_config_class = UnixBenchConfig
    public = True

    def run(self):
        super().run()
        self.script.set_template("unixbench.sh.jinja")
        self.script.extend_context({"unixbench_config": self.config})
        self.script.register_global("UnixBenchTest", UnixBenchTest)
