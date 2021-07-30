import logging
import uuid
import asyncio as aio
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

import asyncssh

from .options import TemplateConfig, TemplateConfigContext
from .benchmark import BenchmarkRunConfig, BenchmarkType
from .instanced import InstanceClient, InstanceConfig
from ..netperf.benchmark import NetperfBenchmark


@dataclass
class BenchmarkManagerConfig(TemplateConfig):
    """
    Describe a set of benchmarks to run on each instance.
    This is the top-level configuration object loaded from the json config.
    """
    verbose: bool = False
    ssh_key: Path = Path("~/.ssh/id_rsa")
    output_path: Path = field(default_factory=Path.cwd)
    sdk_path: Path = Path("~/cheri/cherisdk")
    instances: list[InstanceConfig] = field(default_factory=list)
    benchmarks: list[BenchmarkRunConfig] = field(default_factory=list)


class BenchmarkManager(TemplateConfigContext):
    def __init__(self, config: BenchmarkManagerConfig):
        super().__init__()
        # The ID for this benchplot session
        self.session = uuid.uuid4()
        self.logger = logging.getLogger("benchplot")
        self.instance_manager = InstanceClient()
        self.loop = aio.get_event_loop()
        self.benchmark_instances = {}

        # Note: this will only bind the manager-specific options, the rest of the template arguments
        # will remain as they will need to be bound to specific benchmark instances.
        self.register_template_subst(session=self.session)
        self.config = config.bind(self)
        self.logger.info("Start benchplot session %s", self.session)
        # Adjust libraries log level
        if not self.config.verbose:
            ssh_logger = logging.getLogger("asyncssh")
            ssh_logger.setLevel(logging.WARNING)

    def create_benchmark(self, bench_config: BenchmarkRunConfig, instance: InstanceConfig):
        """Create a benchmark run on an instance"""
        if bench_config.type == BenchmarkType.NETPERF:
            bench_class = NetperfBenchmark
        else:
            self.logger.error("Invalid benchmark type %s", bench_config.type)
        bench = bench_class(self, bench_config, instance)
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
