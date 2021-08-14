import logging
import uuid
import json
import asyncio as aio
import traceback
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

import asyncssh

from .config import Config, TemplateConfig, TemplateConfigContext
from .benchmark import BenchmarkRunConfig, BenchmarkRunRecord, BenchmarkType
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


@dataclass
class BenchmarkManagerRecord(Config):
    session: uuid.UUID
    records: list[BenchmarkRunRecord] = field(default_factory=list)


class BenchmarkManager(TemplateConfigContext):
    def __init__(self, config: BenchmarkManagerConfig):
        super().__init__()
        # The ID for this benchplot session
        self.session = uuid.uuid4()
        self.logger = logging.getLogger("benchplot")
        self.instance_manager = InstanceClient()
        self.loop = aio.get_event_loop()
        self.benchmark_instances = {}
        self.queued_tasks = []

        # Note: this will only bind the manager-specific options, the rest of the template arguments
        # will remain as they will need to be bound to specific benchmark instances.
        self.register_template_subst(session=self.session)
        self.config = config.bind(self)
        self.logger.info("Start benchplot session %s", self.session)
        # Adjust libraries log level
        if not self.config.verbose:
            ssh_logger = logging.getLogger("asyncssh")
            ssh_logger.setLevel(logging.WARNING)
        matplotlib_logger = logging.getLogger("matplotlib")
        matplotlib_logger.setLevel(logging.WARNING)
        self.benchmark_records = BenchmarkManagerRecord(session=self.session)
        self.benchmark_records_path = self.config.output_path / "benchplot-run.json"

    def record_benchmark(self, record: BenchmarkRunRecord):
        self.benchmark_records.records.append(record)

    def _emit_records(self):
        self.logger.debug("Emit benchmark records")
        with open(self.benchmark_records_path, "w") as record_file:
            record_file.write(self.benchmark_records.to_json(indent=4))

    def create_benchmark(self, bench_config: BenchmarkRunConfig, instance: InstanceConfig, uid: uuid.UUID = None):
        """Create a benchmark run on an instance"""
        if bench_config.type == BenchmarkType.NETPERF:
            bench_class = NetperfBenchmark
        else:
            self.logger.error("Invalid benchmark type %s", bench_config.type)
        bench = bench_class(self, bench_config, instance, run_id=uid)
        self.benchmark_instances[bench.uuid] = bench
        self.logger.debug("Created benchmark run %s on %s id=%s", bench_config.name, instance.name, bench.uuid)
        return bench

    async def _run_tasks(self):
        await aio.gather(*self.queued_tasks)

    def _handle_run_command(self):
        for conf in self.config.benchmarks:
            self.logger.debug("Found benchmark %s", conf.name)
            for inst_conf in self.config.instances:
                bench = self.create_benchmark(conf, inst_conf)
                bench.task = self.loop.create_task(bench.run())
                self.queued_tasks.append(bench.task)

    async def _plot_task(self):
        self.benchmark_records = BenchmarkManagerRecord.load_json(self.benchmark_records_path)
        self.session = self.benchmark_records.session
        # Find all benchmark variants we were supposed to run
        # Note: this assumes that we aggregate to compare the same benchmark across OS configs,
        # it can be easily changed to also support comparison of different benchmark runs on
        # the same instance configuration if needed.
        # Load all datasets and for each benchmark and find the baseline instance for
        # each benchmark variant
        aggregate_baseline = {}
        aggregate_groups = defaultdict(list)
        for record in self.benchmark_records.records:
            bench = self.create_benchmark(record.run, record.instance, record.uuid)
            bench.load()
            if record.instance.baseline:
                if record.run.name in aggregate_baseline:
                    self.logger.error("Multiple baseline instances?")
                    raise Exception("Too many baseline specifiers")
                aggregate_baseline[record.run.name] = bench
            else:
                # Must belong to a group
                aggregate_groups[record.run.name].append(bench)
        if len(aggregate_baseline) != len(self.config.benchmarks):
            self.logger.error("Number of benchmark variants does not match " + "number of runs marked as baseline")
            raise Exception("Missing baseline")
        self.logger.debug("Benchmark aggregation groups: %s", aggregate_groups)
        self.logger.debug("Benchmark aggregation baselines: %s", aggregate_baseline)
        # Merge compatible benchmark datasets into the baseline instance
        for name, baseline_bench in aggregate_baseline.items():
            baseline_bench.merge(aggregate_groups[name])
        # From now on we ony operate on the merged data
        for bench in aggregate_baseline.values():
            bench.aggregate()
            bench.verify()
        # Now we have processed all the input data, do the plotting
        for bench in aggregate_baseline.values():
            bench.plot()

    def _handle_plot_command(self):
        self.logger.debug("Import records from %s", self.benchmark_records_path)
        if not self.benchmark_records_path.exists() or self.benchmark_records_path.is_dir():
            self.logger.error("Fatal: Invalid benchmark records file")
            exit(1)
        self.queued_tasks.append(self.loop.create_task(self._plot_task()))

    async def _shutdown_tasks(self):
        for t in self.queued_tasks:
            t.cancel()
        await aio.gather(*self.queued_tasks, return_exceptions=True)

    def run(self, command):
        if command == "run":
            self._handle_run_command()
        elif command == "plot":
            self._handle_plot_command()
        else:
            self.logger.error("Fatal: invalid command")
            exit(1)

        try:
            self.loop.run_until_complete(self._run_tasks())
            if command == "run":
                self._emit_records()
        except KeyboardInterrupt:
            self.logger.error("Shutdown")
            self.loop.run_until_complete(self._shutdown_tasks())
        except Exception as ex:
            self.logger.error("Died %s", ex)
            traceback.print_tb(ex.__traceback__)
            self.loop.run_until_complete(self._shutdown_tasks())
        finally:
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()
            self.logger.info("All tasks finished")
