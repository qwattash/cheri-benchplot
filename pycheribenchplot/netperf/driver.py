import logging
import asyncio as aio
from enum import Enum
from pathlib import Path

import pandas as pd

from ..core.benchmark import BenchmarkBase
from ..elf import ELFInfo, SymResolver
from ..qemu_stats import QEMUAddressRangeHistogram
from .config import NetperfBenchmarkRunConfig
from .plot import *
from .dataset import NetperfData

# class NetperfBenchmarkInstance(BenchmarkInstance):
#     def __init__(self, options, config, kernel_name, netperf_bin, netserver_bin):
#         super().__init__(options, config, kernel_name)
#         self.netperf_bin = netperf_bin
#         self.netserver_bin = netserver_bin

#     async def _run_benchmark(self):
#         await self.run_background_cmd(self.netserver_bin, self.config.netserver_options)
#         await self.run_cmd(self.netperf_bin, self.config.netperf_prime_options,
#                             outfile="/dev/null")
#         await self.run_cmd(self.netperf_bin, self.config.netperf_options, outfile=self.config.output_file)
#         await self.extract_file(self.config.output_file, self.options.output / self.config.output_file)
#         for out in self.config.extra_files:
#             await self.extract_file(out, self.options.output / out)


class NetperfBenchmark(BenchmarkBase):
    benchmark_name = "netperf"

    @classmethod
    def setup_config_options(cls, parser, command, run, plot):
        super().setup_config_options(parser, command, run, plot)
        run.add_argument("config",
                         type=_validate_config,
                         choices=netperf_configs,
                         help="Benchmark configuration to run, this is forwarded also to " +
                         "the remote script that invokes netperf")

        plot.add_argument("config",
                          type=_validate_config,
                          choices=netperf_configs,
                          help="The test configuration name that has been run " + " by run-benchmark.sh")
        plot.add_argument("plot_type", type=NetperfPlot, choices=list(NetperfPlot), help="Plot to draw")
        plot.add_argument("--qemu-pc-samples-prefix",
                          type=str,
                          default="netperf-qemu-",
                          help="Prefix to detect files with qemu PC samples")

    def __init__(self, options, parser):
        super().__init__(options, parser)
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
        self.qemu_pc_samples = QEMUAddressRangeHistogram(self.options, prefix=self.options.qemu_pc_samples_prefix)
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
