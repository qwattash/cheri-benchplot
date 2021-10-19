import logging
import uuid
import json
import asyncio as aio
import traceback
import typing
import argparse as ap
import shutil
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

import termcolor
import asyncssh

from .util import new_logger
from .config import Config, TemplateConfig, TemplateConfigContext, path_field
from .benchmark import BenchmarkRunConfig, BenchmarkRunRecord, BenchmarkType
from .instanced import InstanceClient, InstanceConfig


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
    perfetto_path: Path = path_field("~/cheri/cheri-perfetto/build")
    instances: list[InstanceConfig] = field(default_factory=list)
    benchmarks: list[BenchmarkRunConfig] = field(default_factory=list)


@dataclass
class BenchmarkManagerRecord(Config):
    session: uuid.UUID
    records: list[BenchmarkRunRecord] = field(default_factory=list)


class BenchmarkManager(TemplateConfigContext):
    benchmark_runner_map = {}
    records_filename = "benchplot-run.json"

    @classmethod
    def register_benchmark(cls, type_: BenchmarkType, bench_class):
        cls.benchmark_runner_map[type_] = bench_class

    def __init__(self, config: BenchmarkManagerConfig):
        super().__init__()
        # Cached copy of the configuration template
        self._config_template = config
        # The ID for this benchplot session
        self.session = uuid.uuid4()
        self.logger = new_logger("manager")
        self.loop = aio.get_event_loop()
        self.instance_manager = InstanceClient(self.loop)
        self.benchmark_instances = {}
        self.failed_benchmarks = []
        self.queued_tasks = []

        self._init_session()
        self.logger.info("Start benchplot session %s", self.session)
        # Adjust libraries log level
        if not self.config.verbose:
            ssh_logger = logging.getLogger("asyncssh")
            ssh_logger.setLevel(logging.WARNING)
        matplotlib_logger = logging.getLogger("matplotlib")
        matplotlib_logger.setLevel(logging.WARNING)

    def record_benchmark(self, record: BenchmarkRunRecord):
        self.benchmark_records.records.append(record)

    def _init_session(self):
        """Session-ID dependant initialization"""
        # Note: this will only bind the manager-specific options, the rest of the template arguments
        # will remain as they will need to be bound to specific benchmark instances.
        self.register_template_subst(session=self.session)
        self.config = self._config_template.bind(self)
        self.session_output_path = self.config.output_path / f"benchplot-session-{str(self.session)}"
        self.benchmark_records = BenchmarkManagerRecord(session=self.session)
        self.benchmark_records_path = self.session_output_path / self.records_filename

    def _resolve_recorded_session(self, session: typing.Optional[uuid.UUID]):
        """
        Find recorded session to use for benchmark analysis.
        If a session ID is given, we try to locate the benchmark records for that session.
        If no session is given, default to the most recent session
        (by last-modified time of the record file).
        The resolved session is set as the current session ID and any dependent session state is
        re-initialized.
        If a session can not be resolved, raise an exception
        """
        self.logger.debug("Lookup session records in %s", self.config.output_path)
        resolved = None
        resolved_mtime = 0
        for next_dir in self._iter_output_session_dirs():
            record_file = next_dir / self.records_filename
            record = BenchmarkManagerRecord.load_json(record_file)
            if session is None:
                fstat = record_file.stat()
                if fstat.st_mtime > resolved_mtime:
                    resolved = record
            elif session == record.session:
                resolved = record
                break
        if resolved is None:
            self.logger.error("Can not resolve benchmark session %s in %s",
                              session if session is not None else "DEFAULT",
                              self.config.output_path)
            raise Exception("Benchmark session not found")
        self.session = resolved.session
        self._init_session()
        # Overwrite benchmark records with the resolved data
        self.benchmark_records = resolved

    def _iter_output_session_dirs(self):
        if not self.config.output_path.is_dir():
            self.logger.error("Output directory %s does not exist", self.config.output_path)
            raise OSError("Output directory not found")
        for next_dir in self.config.output_path.iterdir():
            if not next_dir.is_dir() or not (next_dir / self.records_filename).exists():
                continue
            yield next_dir

    def _emit_records(self):
        self.logger.debug("Emit benchmark records")
        with open(self.benchmark_records_path, "w") as record_file:
            record_file.write(self.benchmark_records.to_json(indent=4))

    def create_benchmark(self, bench_config: BenchmarkRunConfig, instance: InstanceConfig, uid: uuid.UUID = None):
        """Create a benchmark run on an instance"""
        try:
            bench_class = self.benchmark_runner_map[bench_config.type]
        except KeyError:
            self.logger.error("Invalid benchmark type %s", bench_config.type)
        bench = bench_class(self, bench_config, instance, run_id=uid)
        self.benchmark_instances[bench.uuid] = bench
        self.logger.debug("Created benchmark run %s on %s id=%s", bench_config.name, instance.name, bench.uuid)
        return bench

    async def _list_task(self):
        self.logger.debug("List recorded sessions at %s", self.config.output_path)
        for next_dir in self._iter_output_session_dirs():
            record_file = next_dir / self.records_filename
            records = BenchmarkManagerRecord.load_json(record_file)
            fstat = record_file.stat()
            mtime = datetime.fromtimestamp(fstat.st_mtime, tz=timezone.utc)
            print(termcolor.colored(f"SESSION {records.session} [{mtime:%d-%m-%Y %H:%M}]", "blue"))
            for bench_record in records.records:
                print(f"\t{bench_record.run.type}:{bench_record.run.name} on instance " +
                      f"{bench_record.instance.name} ({bench_record.uuid})")

    async def _clean_task(self):
        self.logger.debug("Clean all sessions from the output directory")
        for next_dir in self._iter_output_session_dirs():
            shutil.rmtree(next_dir)

    async def _run_tasks(self):
        await aio.gather(*self.queued_tasks)
        await self.instance_manager.stop()

    async def _plot_task(self):
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
        self.logger.debug("Benchmark aggregation groups: %s",
                          {k: map(lambda b: b.uuid, v)
                           for k, v in aggregate_groups.items()})
        self.logger.debug("Benchmark aggregation baselines: %s", {k: b.uuid for k, b in aggregate_baseline.items()})
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

    def _handle_run_command(self, args: ap.Namespace):
        self.instance_manager.start()
        self.session_output_path.mkdir(parents=True)
        for conf in self.config.benchmarks:
            self.logger.debug("Found benchmark %s", conf.name)
            for inst_conf in self.config.instances:
                bench = self.create_benchmark(conf, inst_conf)
                bench.task = self.loop.create_task(bench.run())
                self.queued_tasks.append(bench.task)

    def _handle_plot_command(self, args: ap.Namespace):
        self._resolve_recorded_session(args.session)
        self.queued_tasks.append(self.loop.create_task(self._plot_task()))

    def _handle_list_command(self, args: ap.Namespace):
        self.queued_tasks.append(self.loop.create_task(self._list_task()))

    def _handle_clean_command(self, args: ap.Namespace):
        self.queued_tasks.append(self.loop.create_task(self._clean_task()))

    async def _shutdown_tasks(self):
        for t in self.queued_tasks:
            t.cancel()
        await aio.gather(*self.queued_tasks, return_exceptions=True)
        await self.instance_manager.stop()

    def run(self, args: ap.Namespace):
        """Main entry point to execute benchmark tasks."""
        command = args.command
        if command == "run":
            self._handle_run_command(args)
        elif command == "plot":
            self._handle_plot_command(args)
        elif command == "list":
            self._handle_list_command(args)
        elif command == "clean":
            self._handle_clean_command(args)
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
