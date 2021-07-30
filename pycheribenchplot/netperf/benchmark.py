
import logging
import asyncio as aio
import shutil
from enum import Enum
from pathlib import Path

import pandas as pd

from ..core.benchmark import BenchmarkBase
from ..core.instanced import InstancePlatform
from ..elf import ELFInfo, SymResolver
from ..qemu_stats import QEMUAddressRangeHistogram
from .config import NetperfBenchmarkRunConfig
from .plot import *
from .dataset import NetperfData


class NetperfBenchmark(BenchmarkBase):
    def __init__(self, manager, config, instance_config):
        super().__init__(manager, config, instance_config)
        self.netperf_config = self.config.benchmark_options

        self.logger.debug("Looking for netperf binaries in %s", self.netperf_config.netperf_path)
        rootfs_netperf_base = self.rootfs / self.netperf_config.netperf_path
        rootfs_netperf_bin = list(rootfs_netperf_base.glob("*-netperf"))
        rootfs_netserver_bin = list(rootfs_netperf_base.glob("*-netserver"))
        self.netperf_bin = Path("/") / rootfs_netperf_bin[0].relative_to(self.rootfs)
        self.netserver_bin = Path("/") / rootfs_netserver_bin[0].relative_to(self.rootfs)
        self.logger.debug("Using %s %s", self.netperf_bin, self.netserver_bin)
        self.env = {
            "STATCOUNTERS_NO_AUTOSAMPLE": "1"
        }

    def _bind_configs(self):
        qemu_outfile = "netperf-qemu-{uuid}.csv".format(uuid=self.uuid)
        pmc_outfile = "netperf-pmc-{uuid}.csv".format(uuid=self.uuid)
        self.register_template_subst(netperf_pmc_outfile=pmc_outfile,
                                     netperf_qemu_outfile=qemu_outfile)
        super()._bind_configs()

    async def _run_benchmark(self):
        await super()._run_benchmark()
        await self._run_bg_cmd(self.netserver_bin, self.netperf_config.netserver_options,
                               env=self.env)
        self.logger.info("Prime benchmark")
        await self._run_cmd(self.netperf_bin, self.netperf_config.netperf_prime_options,
                            env=self.env)
        self.logger.info("Run benchmark iterations")
        await self._run_cmd(self.netperf_bin, self.netperf_config.netperf_options,
                            outfile=self.result_path / self.config.output_file, env=self.env)
        self.logger.info("Gather results")
        for out in self.config.extra_files:
            await self._extract_file(out, self.result_path / out)
        if self.instance_config.platform == InstancePlatform.QEMU:
            # Grab the qemu log
            shutil.copy(self._reserved_instance.qemu_trace_file,
                        self.result_path / f"netperf-qemu-{self.uuid}.csv")


class _NetperfBenchmark(BenchmarkBase):

    def __init__(self, manager_config, config, instance_config, instance_daemon):
        super().__init__(manager_config, config, instance_config, instance_daemon)
        self.client_resolver = SymResolver()
        self.server_resolver = SymResolver()

        logging.info("Loading netperf binaries")
        netperf_path = list(self.options.rootfs.glob("opt/**/*-netperf"))
        if not netperf_path:
            parser.error("Missing netperf binary in rootfs {}".format(self.options.rootfs))
        self.client_resolver.register(ELFInfo(netperf_path[0]), 0)
        self.netperf_path = netperf_path[0]

        netserver_path = list(self.options.rootfs.glob("opt/**/*-netserver"))
        if not netserver_path:
            parser.error("Missing netserver binary in rootfs {}".format(self.options.rootfs))
        self.server_resolver.register(ELFInfo(netserver_path[0]), 0)
        self.netserver_path = netserver_path[0]
        # self.netperf_elf = ELFInfo()
        # self.netserver_elf = ELFInfo()

    def process(self):
        if self.options.config.name == "list":
            # list all configs and exit
            logging.info("Listing available netperf configurations:")
            for c in netperf_configs:
                if c.name == "list":
                    continue
                logging.info(c.get_description())
            exit(0)
        super().process()

    def run(self):
        # for kernel_variant in self.options.kernel_variants:
        #     netperf_bin = Path("/") / self.netperf_path.relative_to(self.options.rootfs)
        #     netserver_bin = Path("/") / self.netserver_path.relative_to(self.options.rootfs)
        #     instance = NetperfBenchmarkInstance(self.options, self.options.config, kernel_variant,
        #                                         netperf_bin, netserver_bin)
        #     self.runner.run_instance(instance)
        super().run()

    def plot(self):
        self.netperf = NetperfData(self.options)
        self.qemu_pc_samples = QEMUAddressRangeHistogram(self.options,
            prefix=self.options.qemu_pc_samples_prefix)
        super().plot()

    # async def run_on_instance(self, instance: BenchmarkInstance):
    #     # instance.run_cmd()
    #     return

    def _is_pmc_input(self, filepath):
        return filepath.name.startswith("netperf-pmc-")

    def _load_file(self, dirpath, filepath):
        super()._load_file(dirpath, filepath)

        if filepath.name.startswith("netperf-output-"):
            self.netperf.load(filepath)
        if filepath.name.startswith(self.options.qemu_pc_samples_prefix):
            self.qemu_pc_samples.load(filepath)

    def _load_pmc(self, filepath):
        match = re.match("netperf-pmc-([a-zA-Z0-9-]+)\.csv", filepath.name)
        assert match, "Malformed netperf output file name, requires netperf-pmc-<UUID>"
        dataset_id = match.group(1)

        self.pmc.load(dataset_id, filepath)

    def _process_data_sources(self):
        super()._process_data_sources()
        self.qemu_pc_samples.process()
        self.netperf.process()

    def _merge_stats(self):
        merged = pd.DataFrame(self.netperf.df)
        merged = merged.join(self.pmc.stats_df, on="__dataset_id")
        return merged

    def _compute_relative_stats(self):
        pass

    def _draw(self):
        print(self.options)
        if self.options.plot_type == NetperfPlot.ALL_BY_XFER_SIZE:
            plot = NetperfTXSizeStackPlot(self)
        elif self.options.plot_type == NetperfPlot.QEMU_PC_HIST:
            plot = NetperfQemuPCHist(self)
        else:
            assert False, "Not reached"
        plot.draw()
