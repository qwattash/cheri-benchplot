import logging
import asyncio as aio
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import TemplateConfig, path_field
from ..core.dataset import (DatasetID, Field, DataField, StrField, IndexField, col2stat)
from ..core.csv import CSVDataSetContainer
from ..core.procstat import ProcstatDataset


@dataclass
class NetperfRunConfig(TemplateConfig):
    netperf_path: Path = path_field("opt/{cheri_target}/netperf/bin")
    netperf_ktrace_options: list[str] = field(default_factory=list)
    netperf_prime_options: list[str] = field(default_factory=list)
    netperf_options: list[str] = field(default_factory=list)
    netserver_options: list[str] = field(default_factory=list)


class NetperfProcstat(ProcstatDataset):
    """
    Netperf procstat runner and parser.
    This replaces the base procstat dataset in netperf.
    TODO support multiple outputs per dataset to split the generator/parser interfaces
    This will be tacked on the netperf generator instead.
    """
    benchmark_dataset_filter = [DatasetID.NETPERF_DATA]

    async def run_pre_benchmark(self):
        netperf = self.benchmark.get_dataset(DatasetID.NETPERF_DATA)
        netperf_stopped = await self.benchmark.run_bg_cmd(netperf.netperf_bin, ["-z"], env=netperf.run_env)
        await aio.sleep(5)  # Give some time to settle
        try:
            pid = self.benchmark.command_history[netperf_stopped].pid
            await self._run_procstat(pid)
        finally:
            await self.benchmark.stop_bg_cmd(netperf_stopped)


class NetperfData(CSVDataSetContainer):
    dataset_id = "netperf-data"
    run_options_class = NetperfRunConfig
    fields = [
        StrField("Socket Type"),
        StrField("Protocol"),
        StrField("Direction"),
        DataField("Elapsed Time (sec)"),
        DataField("Throughput"),
        StrField("Throughput Units"),
        Field("Local Send Socket Size Requested"),
        Field("Local Send Socket Size Initial"),
        Field("Local Send Socket Size Final"),
        Field("Local Recv Socket Size Requested"),
        Field("Local Recv Socket Size Initial"),
        Field("Local Recv Socket Size Final"),
        Field("Remote Send Socket Size Requested"),
        Field("Remote Send Socket Size Initial"),
        Field("Remote Send Socket Size Final"),
        Field("Remote Recv Socket Size Requested"),
        Field("Remote Recv Socket Size Initial"),
        Field("Remote Recv Socket Size Final"),
        Field("Local Send Size"),
        Field("Local Recv Size"),
        Field("Remote Send Size"),
        Field("Remote Recv Size"),
        IndexField("Request Size Bytes"),
        IndexField("Response Size Bytes"),
        Field("Local CPU Util %"),
        Field("Local CPU User %"),
        Field("Local CPU System %"),
        Field("Local CPU I/O %"),
        Field("Local CPU IRQ %"),
        Field("Local CPU swintr %"),
        StrField("Local CPU Util Method"),
        Field("Local Service Demand"),
        Field("Remote CPU Util %"),
        Field("Remote CPU User %"),
        Field("Remote CPU System %"),
        Field("Remote CPU I/O %"),
        Field("Remote CPU IRQ %"),
        Field("Remote CPU swintr %"),
        StrField("Remote CPU Util Method"),
        Field("Remote Service Demand"),
        StrField("Service Demand Units"),
        Field("Confidence Level Percent"),
        Field("Confidence Width Target"),
        Field("Confidence Iterations Run"),
        Field("Throughput Confidence Width (%)"),
        Field("Local CPU Confidence Width (%)"),
        Field("Remote CPU Confidence Width (%)"),
        DataField("Transaction Rate Tran/s"),
        DataField("Round Trip Latency usec/tran"),
        Field("Initial Burst Requests"),
        Field("Local Transport Retransmissions"),
        Field("Remote Transport Retransmissions"),
        Field("Transport MSS bytes"),
        Field("Local Send Throughput"),
        Field("Local Recv Throughput"),
        Field("Remote Send Throughput"),
        Field("Remote Recv Throughput"),
        Field("Local CPU Bind"),
        Field("Local CPU Count"),
        Field("Local Peak Per CPU Util %"),
        Field("Local Peak Per CPU ID"),
        Field("Local CPU Frequency MHz"),
        Field("Remote CPU Bind"),
        Field("Remote CPU Count"),
        Field("Remote Peak Per CPU Util %"),
        Field("Remote Peak Per CPU ID"),
        Field("Remote CPU Frequency MHz"),
        Field("Source Port"),
        StrField("Source Address"),
        Field("Source Family"),
        Field("Destination Port"),
        StrField("Destination Address"),
        Field("Destination Family"),
        Field("Local Send Calls"),
        Field("Local Recv Calls"),
        Field("Local Bytes Per Recv"),
        Field("Local Bytes Per Send"),
        Field("Local Bytes Sent"),
        Field("Local Bytes Received"),
        Field("Local Bytes Xferred"),
        Field("Local Send Offset"),
        Field("Local Recv Offset"),
        Field("Local Send Alignment"),
        Field("Local Recv Alignment"),
        Field("Local Send Width"),
        Field("Local Recv Width"),
        Field("Local Send Dirty Count"),
        Field("Local Recv Dirty Count"),
        Field("Local Recv Clean Count"),
        Field("Local NODELAY"),
        Field("Local Cork"),
        Field("Remote Send Calls"),
        Field("Remote Recv Calls"),
        Field("Remote Bytes Per Recv"),
        Field("Remote Bytes Per Send"),
        Field("Remote Bytes Sent"),
        Field("Remote Bytes Received"),
        Field("Remote Bytes Xferred"),
        Field("Remote Send Offset"),
        Field("Remote Recv Offset"),
        Field("Remote Send Alignment"),
        Field("Remote Recv Alignment"),
        Field("Remote Send Width"),
        Field("Remote Recv Width"),
        Field("Remote Send Dirty Count"),
        Field("Remote Recv Dirty Count"),
        Field("Remote Recv Clean Count"),
        Field("Remote NODELAY"),
        Field("Remote Cork"),
        StrField("Local Interface Vendor"),
        StrField("Local Interface Device"),
        StrField("Local Interface Subvendor"),
        StrField("Local Interface Subdevice"),
        StrField("Remote Interface Vendor"),
        StrField("Remote Interface Device"),
        StrField("Remote Interface Subvendor"),
        StrField("Remote Interface Subdevice"),
        Field("Local Interval Usecs"),
        Field("Local Interval Burst"),
        Field("Remote Interval Usecs"),
        Field("Remote Interval Burst"),
        Field("Local OS Security Type ID"),
        Field("Local OS Security Enabled Num"),
        Field("Result Tag"),
        StrField("Test UUID"),
        Field("Minimum Latency Microseconds"),
        Field("Maximum Latency Microseconds"),
        Field("50th Percentile Latency Microseconds"),
        Field("90th Percentile Latency Microseconds"),
        Field("99th Percentile Latency Microseconds"),
        Field("Mean Latency Microseconds"),
        Field("Stddev Latency Microseconds"),
        Field("Local Socket Priority"),
        Field("Remote Socket Priority"),
        StrField("Local Socket TOS"),
        StrField("Remote Socket TOS"),
        Field("Local Congestion Control Algorithm"),
        Field("Remote Congestion Control Algorithm"),
        Field("Local Fill File"),
        Field("Remote Fill File"),
        StrField("Command Line"),
        StrField("CHERI Netperf ABI"),
        StrField("CHERI Kernel ABI")
    ]

    def __init__(self, benchmark, dset_key, config):
        super().__init__(benchmark, dset_key, config)
        self.netserver_bin = None
        self.netperf_bin = None
        self.run_env = {"STATCOUNTERS_NO_AUTOSAMPLE": "1"}

    def raw_fields(self, include_derived=False):
        return NetperfData.fields

    def _load_csv(self, path: Path, **kwargs):
        kwargs["skiprows"] = 1
        return super()._load_csv(path, **kwargs)

    def aggregate(self):
        super().aggregate()
        self.agg_df = self.merged_df.copy()

    def output_file(self):
        return super().output_file().with_suffix(".csv")

    def get_addrspace_key(self):
        return self.netperf_bin.name

    def _set_netperf_option(self, flag, value):
        """
        Set a netperf CLI option if one is not specified by the configuration already
        """
        if flag in self.config.netperf_options:
            return
        self.config.netperf_options = [flag, value] + self.config.netperf_options

    def configure(self, opts):
        opts = super().configure(opts)
        # Resolve binaries here as the configuration is stable at this point
        rootfs_netperf_base = self.benchmark.rootfs / self.config.netperf_path
        rootfs_netperf_bin = rootfs_netperf_base / "netperf"
        rootfs_netserver_bin = rootfs_netperf_base / "netserver"
        # Paths relative to the remote root directory
        self.netperf_bin = Path("/") / rootfs_netperf_bin.relative_to(self.benchmark.rootfs)
        self.netserver_bin = Path("/") / rootfs_netserver_bin.relative_to(self.benchmark.rootfs)
        self.logger.debug("Using %s %s", self.netperf_bin, self.netserver_bin)
        # Determine any extra options for cooperation with other datasets
        pmc = self.benchmark.get_dataset(DatasetID.PMC)
        qemu = (self.benchmark.get_dataset(DatasetID.QEMU_STATS_BB_HIST)
                or self.benchmark.get_dataset(DatasetID.QEMU_STATS_BB_CALL)
                or self.benchmark.get_dataset(DatasetID.QEMU_CTX_CTRL))
        if pmc:
            self._set_netperf_option("-G", pmc.output_file().name)
            if not qemu:
                self._set_netperf_option("-g", "all")
        if qemu:
            self._set_netperf_option("-g", "qemu")
        return opts

    async def run_benchmark(self):
        await super().run_benchmark()
        netserver = await self.benchmark.run_bg_cmd(self.netserver_bin, self.config.netserver_options, env=self.run_env)
        try:
            self.logger.info("Prime benchmark")
            await aio.sleep(5)  # Give some time to settle
            await self.benchmark.run_cmd(self.netperf_bin, self.config.netperf_prime_options, env=self.run_env)
            self.logger.info("Run benchmark iterations")
            with open(self.output_file(), "w+") as outfd:
                await self.benchmark.run_cmd(self.netperf_bin,
                                             self.config.netperf_options,
                                             outfile=outfd,
                                             env=self.run_env)
            pidmap = self.benchmark.get_dataset(DatasetID.PIDMAP)
            if pidmap:
                # Sample PIDs before stopping netserver
                await pidmap.sample_system_pids()
        finally:
            await self.benchmark.stop_bg_cmd(netserver)
