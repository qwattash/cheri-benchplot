import logging
import asyncio as aio
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd

from ..core.config import TemplateConfig, path_field
from ..core.manager import BenchmarkManager
from ..core.benchmark import (BenchmarkBase, BenchmarkDataSetConfig, BenchmarkType)
from ..core.instanced import InstancePlatform
from ..core.dataset import DatasetID
from ..core.excel import SpreadsheetSurface
from ..core.html import HTMLSurface
from .plot import *
from .dataset import NetperfData


@dataclass
class NetperfBenchmarkRunConfig(TemplateConfig):
    netperf_path: Path = path_field("opt/{cheri_target}/netperf/bin")
    netperf_ktrace_options: list[str] = field(default_factory=list)
    netperf_prime_options: list[str] = field(default_factory=list)
    netperf_options: list[str] = field(default_factory=list)
    netserver_options: list[str] = field(default_factory=list)
    qemu_log_output: str = "netperf-qemu-{uuid}.pb"


class NetperfBenchmark(BenchmarkBase):
    def __init__(self, manager, config, instance_config, run_id=None):
        super().__init__(manager, config, instance_config, run_id=run_id)
        self.netperf_config = NetperfBenchmarkRunConfig(**self.config.benchmark_options).bind(self)

        self.logger.debug("Looking for netperf binaries in %s", self.netperf_config.netperf_path)
        rootfs_netperf_base = self.rootfs / self.netperf_config.netperf_path
        rootfs_netperf_bin = list(rootfs_netperf_base.glob("*-netperf"))
        rootfs_netserver_bin = list(rootfs_netperf_base.glob("*-netserver"))
        self.netperf_bin = Path("/") / rootfs_netperf_bin[0].relative_to(self.rootfs)
        self.netserver_bin = Path("/") / rootfs_netserver_bin[0].relative_to(self.rootfs)
        self.logger.debug("Using %s %s", self.netperf_bin, self.netserver_bin)
        self.env = {"STATCOUNTERS_NO_AUTOSAMPLE": "1"}

        if self.config.output_file is None:
            self.logger.error("Malformed config: missing netperf output file")
            raise Exception("Benchmark config error")

    async def _run_procstat(self):
        await super()._run_procstat()
        # Grab the memory mapping for the process
        netperf_stopped = await self._run_bg_cmd(self.netperf_bin, ["-z"], env=self.env)
        await aio.sleep(5)  # Give some time to settle
        try:
            pid = self._remote_task_info[netperf_stopped].pid
            with open(self.procstat_output, "w+") as outfd:
                await self._run_cmd("procstat", ["-v", str(pid)], env=self.env, outfile=outfd)
            self.logger.debug("Collected procstat info")
        finally:
            await self._stop_bg_cmd(netperf_stopped)

    async def _run_benchmark(self):
        await super()._run_benchmark()
        netserver = await self._run_bg_cmd(self.netserver_bin, self.netperf_config.netserver_options, env=self.env)
        try:
            self.logger.info("Prime benchmark")
            await aio.sleep(5) # Give some time to settle
            await self._run_cmd(self.netperf_bin, self.netperf_config.netperf_prime_options, env=self.env)
            self.logger.info("Run benchmark iterations")
            with open(self.result_path / self.config.output_file, "w+") as outfd:
                await self._run_cmd(self.netperf_bin, self.netperf_config.netperf_options, outfile=outfd, env=self.env)
        finally:
            await self._stop_bg_cmd(netserver)
        self.logger.info("Gather results")
        for out in self.config.extra_files:
            await self._extract_file(out, self.result_path / out)
        if self.instance_config.platform == InstancePlatform.QEMU:
            # Grab the qemu perfetto log
            shutil.copy(self._reserved_instance.qemu_pftrace_file,
                        self.result_path / self.netperf_config.qemu_log_output)

    def verify(self):
        dset = self.get_dataset(DatasetID.NETPERF_DATA)
        # Check that all benchmarks report the same number of iterations
        if "Confidence Iterations Run" in dset.agg_df.columns:
            if len(dset.agg_df["Confidence Iterations Run"].unique()) > 1:
                self.logger.error("Benchmark iteration count does not match across samples")
        else:
            self.logger.warning(
                "Can not verify netperf iteration count, consider enabling the CONFIDENCE_ITERATION output")
        # Check that all benchmarks ran a consistent amount of sampling
        # functions in libstatcounters
        dset = self.get_dataset(DatasetID.QEMU_STATS_CALL_HIST)
        if dset:
            syms_index = dset.agg_df.index.get_level_values("symbol")
            cpu_start = syms_index == "cpu_start"
            cpu_stop = syms_index == "cpu_stop"
            statcounters_sample = syms_index == "statcounters_sample"
            check = dset.agg_df.loc[cpu_start, "call_count"].unique()
            if len(check) > 1:
                self.logger.error("netperf::cpu_start anomalous #calls %s", check)
            check = dset.agg_df.loc[cpu_stop, "call_count"].unique()
            if len(check) > 1:
                self.logger.error("netperf::cpu_stop anomalous #calls %s", check)
            check = dset.agg_df.loc[statcounters_sample, "call_count"].unique()
            if len(check) > 1:
                self.logger.error("libstatcounters::statcounters_sample anomalous #calls %s",
                                  check)


BenchmarkManager.register_benchmark(BenchmarkType.NETPERF, NetperfBenchmark)
