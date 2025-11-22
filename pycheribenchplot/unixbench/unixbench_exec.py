from dataclasses import dataclass
from pathlib import Path

from ..core.config import Config, ConfigPath, config_field
from ..core.tvrs import TVRSExecTask
from ..generic.timing import TimingConfig, TimingExecTask


@dataclass
class UnixBenchScenario(Config):
    test_name: str = config_field(Config.REQUIRED, desc="Unixbench test to run")
    duration: int = config_field(
        10000, desc="Duration of an iteration (number of transactions)"
    )
    args: list[str] = config_field(list, desc="Extra test arguments")


@dataclass
class UnixBenchConfig(TimingConfig):
    scenario_path: ConfigPath | None = config_field(
        Path("pycheribenchplot/unixbench/scenarios"), desc="Path to unixbench scenarios"
    )
    unixbench_path: ConfigPath | None = config_field(
        None, desc="Path of unixbench in the remote host"
    )
    scenarios: dict[str, UnixBenchScenario] = config_field(
        dict,
        desc="Inline scenarios. The key must match the name given by the `scenario` parameter",
    )


class UnixBenchExec(TVRSExecTask, TimingExecTask):
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

    def run(self):
        super().run()
        self.script.set_template("unixbench.sh.jinja")
        self.script.extend_context({"unixbench_config": self.config})
