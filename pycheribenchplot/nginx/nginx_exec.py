import shutil
import subprocess as sp
from dataclasses import dataclass
from pathlib import Path

from ..core.artefact import RemoteBenchmarkIterationTarget
from ..core.config import Config, ConfigPath, config_field
from ..core.shellgen import TemplateContextBase
from ..core.task import output
from ..core.tvrs import TVRSExecConfig, TVRSExecTask


@dataclass
class NginxScenario(Config):
    endpoint: str = config_field(Config.REQUIRED, desc="HTTP(s) endpoint for the test")
    nginx_config: str = config_field(
        Config.REQUIRED,
        desc="Nginx configuration file for this run, will be imported into assets.",
    )
    connections: int = config_field(50, desc="Concurrent connections")
    threads: int = config_field(1, desc="Number of wrk threads")
    requests: int = config_field(10000, desc="Number of transactions per iteration")
    args: list[str] = config_field(list, desc="Extra test arguments")
    nginx_cpu_affinity: str | None = config_field(None, desc="CPU to pin nginx server")
    wrk_cpu_affinity: str | None = config_field(None, desc="CPU to pin wrk client")


@dataclass
class NginxBenchConfig(TVRSExecConfig):
    prefix: ConfigPath | None = config_field(
        Path("/usr/local/nginx"), desc="Configured --prefix of nginx"
    )
    www_path: ConfigPath | None = config_field(
        None, desc="Path to the www directory to import to the session"
    )


class NginxBenchExec(TVRSExecTask):
    """
    Run wrk benchmark with nginx.

    The test to run is taken from the `scenario` parameter, which must
    point to a valid NginxScenario configuration.
    """

    task_namespace = "nginx"
    task_name = "exec"
    task_config_class = NginxBenchConfig
    public = True
    script_template = "nginx.sh.jinja"
    scenario_config_class = NginxScenario

    @output
    def report(self):
        return RemoteBenchmarkIterationTarget(self, "report", ext="json")

    def run(self):
        super().run()

        # Render the wrk lua script asset
        self.logger.info("Generate wrk report script")
        ctx = TemplateContextBase(self.logger)
        ctx.set_template("assets/nginx/wrk-msglimit-report.lua.jinja")
        ctx.extend_context({"nginx_request_limit": self.scenario().requests})
        lua_path = self.benchmark.get_benchmark_data_path() / "wrk-report.lua"
        with open(lua_path, "w+") as lua_file:
            ctx.render(lua_file)

        # Import the www directory to the asset path
        self.logger.info("Import www directory")
        if self.config.www_path:
            shutil.copytree(
                self.config.www_path,
                self.session.get_asset_root_path() / "www",
                dirs_exist_ok=True,
            )

        # Generate the server self-signed certificate
        crt_path = self.session.get_asset_root_path()
        if not (crt_path / "server.crt").exists():
            self.logger.info("Generate self-signed server certificate")
            sp.run(
                [
                    "openssl",
                    "req",
                    "-nodes",
                    "-new",
                    "-x509",
                    "-keyout",
                    crt_path / "server.key",
                    "-out",
                    crt_path / "server.crt",
                    "-subj",
                    "/C=UK/O=test/CN=*.cheri-nginx-benchmark.local",
                ],
                stdout=sp.DEVNULL,
                check=True,
            )

        # Render the nginx configuration
        self.logger.info("Generate server configuration")
        ctx = TemplateContextBase(self.logger)
        ctx.set_template(f"assets/nginx/{self.scenario().nginx_config}.conf.jinja")
        conf_path = self.benchmark.get_benchmark_data_path() / "nginx.conf"
        with open(conf_path, "w+") as conf_file:
            ctx.render(conf_file)

        self.script.extend_context(
            {
                "nginx_config": self.config,
                "nginx_gen_output_path": self.report.shell_path_builder(),
            }
        )
