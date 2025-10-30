import json
from dataclasses import dataclass

import numpy as np

from ..core.artefact import BenchmarkIterationTarget, PLDataFrameLoadTask
from ..core.config import Config, ConfigPath, InstanceCheriBSD
from ..core.task import DataGenTask, output


@dataclass
class IngestQPSConfig(Config):
    #: Path to the QPS results directory hierarchy
    path: ConfigPath


class IngestQPSData(DataGenTask):
    """
    Task that takes data from an existing QPS benchmark run and loads
    it into a benchplot session.

    Note that we expect the data to be parameterized and the input
    data should be placed in a directory hierarchy as follows:
    grpc-<abi>-<variant>/<runtime>/<scenario>.<iteration>.json

    abi: one of {hybrid, purecap, benchmark}
    variant: one of {base, stackzero}
    runtime: one of {base, c18n, revoke}
    scenario: name of the QPS scenario
    iteration: iteration number
    """
    task_config_class = IngestQPSConfig
    task_namespace = "qps"
    task_name = "ingest-external"
    public = True

    @output
    def data(self):
        return BenchmarkIterationTarget(self, "summary", ext="json", loader=PLDataFrameLoadTask)

    @output
    def histogram(self):
        return BenchmarkIterationTarget(self, "latency-histogram", ext="json", loader=PLDataFrameLoadTask)

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

        summary_paths = list(self.data.iter_paths())
        hist_paths = list(self.histogram.iter_paths())
        for i in range(self.benchmark.config.iterations):
            dst = summary_paths[i]
            hist_dst = hist_paths[i]
            # Note that we expect iteration # to be 1-based here
            data_file = self.config.path / f"grpc-{abi}-{variant}" / runtime / f"summary_{scenario}.{i + 1}.json"
            self.logger.debug("Ingest %s => %s", data_file, dst)
            with open(data_file, "r") as fp:
                data = json.load(fp)
            with open(dst, "w+") as fp:
                summary = {}
                summary["qps"] = data["summary"]["qps"]
                summary["message_count"] = data["summary"].get("messageCount", None)
                summary["latency50"] = data["summary"]["latency50"]
                summary["latency90"] = data["summary"]["latency90"]
                summary["latency95"] = data["summary"]["latency95"]
                summary["latency99"] = data["summary"]["latency99"]
                summary["latency999"] = data["summary"]["latency999"]
                payload_conf = data["scenario"]["clientConfig"].get("payloadConfig", {})
                params = payload_conf.get("simpleParams", {})
                summary["req_size"] = params.get("reqSize", 0)
                summary["resp_size"] = params.get("respSize", 0)
                json.dump(summary, fp)
            self.logger.debug("Ingest %s => %s", data_file, hist_dst)
            with open(hist_dst, "w+") as fp:
                buckets = data["latencies"]["bucket"]
                # See grpc/test/core/util/histogram.cc
                client_conf = data["scenario"]["clientConfig"]
                # defaults from test/cpp/qps/histogram.h
                hmax = 6e9
                hres = 0.01
                if "histogramParams" in client_conf:
                    hp = client_conf["histogramParams"]
                    hmax = hp.get("maxPossible", hmax)
                    hres = hp.get("resolution", hres)
                num_buckets = (np.log(hmax) / np.log(1.0 + hres)) + 1
                buckets_start = np.logspace(0, len(buckets), num=int(num_buckets), base=(1 + hres), endpoint=False)
                assert len(buckets) == len(buckets_start)
                json.dump({
                    "buckets": buckets,
                    "start": list(buckets_start),
                }, fp)
