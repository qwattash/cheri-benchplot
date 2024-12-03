from dataclasses import MISSING, dataclass

from ..core.artefact import Target
from ..core.config import Config, config_field
from ..core.task import output
from ..core.tvrs import TVRSExecConfig, TVRSExecTask


@dataclass
class CheriBuildScenario(Config):
    """
    Common options for tasks that build source code to collect statistics
    """
    target: str = config_field(MISSING, desc="Cheribuild targe to build")
    kernel: str | None = config_field(None, desc="Shorthand to build a specific cheribsd kernel")
    args: list[str] = config_field(list, desc="Custom cheribuild arguments")


class CheriBuildCompilationDB(TVRSExecTask):
    """
    Build some binary and collect statistics furing the build process.
    """
    task_namespace = "compilationdb"
    task_name = "cheribuild"
    public = True
    task_config_class = TVRSExecConfig
    scenario_config_class = CheriBuildScenario

    @output
    def build_trace(self):
        return Target(self, "build-trace", ext="txt")

    def run(self):
        super().run()
        # Check dependencies
        cheribuild = self.session.user_config.cheribuild_path / "cheribuild.py"
        if not cheribuild.exists():
            self.logger.error("Missing cheribuild binary %s", cheribuild)
            raise ValueError("Invalid configuration")

        if self.benchmark.config.iterations > 1:
            self.logger.warning("Compilation DB task is running for multiple iterations")

        scenario_config = self.scenario()
        cli_args = scenario_config.args
        if scenario_config.target.startswith("cheribsd-") and scenario_config.kernel:
            cli_args += [f"--{scenario_config.target}/kernel-config", scenario_config.kernel]

        self.script.exec_iteration("compilationdb", template="compilationdb.hook.jinja")
        self.script.extend_context({
            "compilationdb_output": self.build_trace.single_path(),
            "cheribuild": cheribuild,
            "cheribuild_args": cli_args
        })
