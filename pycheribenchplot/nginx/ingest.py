import re
import json
from dataclasses import dataclass

from ..core.artefact import BenchmarkIterationTarget, PLDataFrameLoadTask
from ..core.config import Config, ConfigPath, InstanceCheriBSD
from ..core.task import DataGenTask, output

@dataclass
class IngestWRKConfig(Config):
    #: Path to the WRK results directory hierarchy
    path: ConfigPath
    #: Prefix for the jail names in the results directory
    jail_prefix: str = ""


class IngestWRKData(DataGenTask):
    """
    Task that takes data from an existing WRK benchmark run and loads it
    into a benchplot session.

    Note that we expect the data to be parameterized and the input
    data should be placed in a directory hierarchy as follows:
    <prefix><abi>-<variant>/<runtime>/<scenario>.<iteration>.json

    abi: one of {hybrid, purecap, benchmark}
    variant: one of {base, subobj, stackzero}
    runtime: one of {base, c18n, revoke}
    scenario: name of the WRK file being fetched
    iteration: iteration number
    """
    task_config_class= IngestWRKConfig
    task_namespace = "wrk"
    task_name = "ingest"
    public = True

    @output
    def data(self):
        return BenchmarkIterationTarget(self, "summary", ext="json",
                                        loader=PLDataFrameLoadTask)

    def run(self):
        user_abi = self.benchmark.config.instance.cheri_target
        if user_abi.is_hybrid_abi():
            abi = "hybrid"
        elif user_abi.is_purecap_abi():
            abi = "purecap"
        elif user_abi.is_benchmark_abi():
            abi = "benchmark"
        else:
            self.logger.error("Invalid platform config %s", user_abi)
            raise RuntimeError("Invalid configuration")

        params = self.benchmark.parameters
        if "variant" not in params:
            self.logger.error("Missing 'variant' parameter key in configuration")
            raise RuntimeError("Invalid configuration")
        variant = params["variant"]
        if "runtime" not in params:
            self.logger.error("Missing 'runtime' parameter key in configuration")
            raise RuntimeError("Invalid configuration")
        runtime = params["runtime"]
        if "scenario" not in params:
            self.logger.error("Missing 'scenario' parameter key in configuration")
            raise RuntimeError("Invalid configuration")
        scenario = params["scenario"]

        scenario_match = re.match(r".*_([0-9]+)b$", scenario)
        if scenario_match is None:
            self.logger.error("Invalid scenario name %s", scenario)
            raise RuntimeError("Invalid configuration")
        request_size = scenario_match.group(1)

        summary_paths = list(self.data.iter_paths())
        for i in range(self.benchmark.config.iterations):
            dst = summary_paths[i]
            # Note that we expect iteration # to be 1-based here
            data_file = self.config.path / f"{self.config.jail_prefix}{abi}-{variant}" / runtime / f"result_{scenario}.{i + 1}.json"
            self.logger.debug("Ingest %s => %s", data_file, dst)
            with open(data_file, "r") as fp:
                data = json.load(fp)
            summary = {
                "duration": data["summary"]["duration"],
                "requests": data["summary"]["requests"],
                "bytes": data["summary"]["bytes"],
                "request_size": int(request_size),
                "errstatus": data["errors"]["status"],
                "latency_min": data["latency"]["min"],
                "latency_max": data["latency"]["max"],
                "latency_mean": data["latency"]["mean"],
                "latency_std": data["latency"]["std"],
                "latency_pct50": data["latency"]["percentile50"],
                "latency_pct90": data["latency"]["percentile90"],
                "latency_pct95": data["latency"]["percentile95"],
                "latency_pct99": data["latency"]["percentile99"],
                "latency_pct999": data["latency"]["percentile999"]
            }
            with open(dst, "w+") as fp:
                json.dump(summary, fp)
