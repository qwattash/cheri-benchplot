import logging
import asyncio as aio
from enum import Enum

from ..netperf.benchmark import NetperfBenchmark
from .benchmark import BenchmarkManagerConfig, BenchmarkRunConfig, BenchmarkType
from .instanced import InstanceClient, InstanceConfig


class BenchmarkManager:
    def __init__(self, config: BenchmarkManagerConfig):
        self.config = config
        self.logger = logging.getLogger("benchplot")
        self.instance_manager = InstanceClient()
        self.loop = aio.get_event_loop()
        self.benchmark_instances = {}

    def create_benchmark(self, bench_config: BenchmarkRunConfig, instance: InstanceConfig):
        """Create a benchmark run on an instance"""
        if bench_config.type == BenchmarkType.NETPERF:
            bench_class = NetperfBenchmark
        else:
            self.logger.error("Invalid benchmark type %s", bench_config.type)
        bench = bench_class(self.config, bench_config, instance, self.instance_manager)
        bench.task = self.loop.create_task(bench.run())
        self.benchmark_instances[bench.uuid] = bench
        self.logger.debug("Created benchmark run %s on %s id=%s", bench_config.name, instance.name, bench.uuid)
        return bench

    async def _run_tasks(self):
        await aio.gather(*[bi.task for bi in self.benchmark_instances.values()])

    def run(self):
        for conf in self.config.benchmarks:
            self.logger.debug("Found benchmark %s", conf.name)
            for inst_conf in self.config.instances:
                self.create_benchmark(conf, inst_conf)
        try:
            self.loop.run_until_complete(self._run_tasks())
        except KeyboardInterrupt:
            self.logger.error("Shutdown")
        except Exception as ex:
            self.logger.error("Died %s", ex)
        finally:
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()
            self.logger.info("All tasks finished")



    def plot(self):
        pass

    # def initialize_cli(self, parser: ap.ArgumentParser):
    #     parser.add_argument("-s", "--stats", type=Path, nargs='+', required=True,
    #                         help="Path to benchmark stats results")
    #     parser.add_argument("-r", "--rootfs", type=Path, required=True,
    #                         help="Path to the CheriBSD rootfs to find related binaries")
    #     parser.add_argument("--cpu", type=BenchmarkCPU, choices=list(BenchmarkCPU),
    #                         required=True, default=BenchmarkCPU.FLUTE,
    #                         help="CPU on which the benchmark has run")
    #     parser.add_argument("-q", "--quiet", action="store_true")
    #     # XXX generate benchmark bundle directory if none is given
    #     parser.add_argument("-o", "--output", type=Path, help="Path to output directory",
    #                         default=Path.cwd())
    #     parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    #     bench = parser.add_subparsers(
    #         dest="benchmark", metavar="benchmark",
    #         help="Benchmark-specific options available by invoking it with -h",
    #         required=True)
    #     for driver in benchmark_manager.get_drivers():
    #         driver_parser = bench.add_parser(driver.benchmark_name)
    #         driver_commands = driver_parser.add_subparsers(dest="command", required=True)
    #         run = driver_commands.add_parser("run", description="Run {} benchmark".format(driver.benchmark_name))
    #         plot = driver_commands.add_parser("plot", description="Plot {} results".format(driver.benchmark_name))
    #         driver_parser.set_defaults(driver_class=driver)
    #         driver.setup_config_options(driver_parser, driver_commands, run, plot)
