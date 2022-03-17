import argparse as ap
import asyncio as aio
import code
import itertools as it
import json
import logging
import multiprocessing
import shutil
import typing
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import asyncssh
import pandas
import termcolor
from git import Repo

from .analysis import AnalysisConfig, BenchmarkAnalysisRegistry
from .benchmark import BenchmarkBase, BenchmarkRunConfig, BenchmarkRunRecord
from .config import Config, TemplateConfig, TemplateConfigContext, path_field
from .dataset import DatasetRegistry
from .instance import InstanceConfig, InstanceManager
from .util import new_logger


@dataclass
class BenchplotUserConfig(Config):
    """
    User-environment configuration.
    This defines system paths for programs and source code we use
    """
    sdk_path: Path = path_field("~/cheri/cherisdk")
    build_path: Path = path_field("~/cheri/build")
    src_path: Path = path_field("~/cheri")
    perfetto_path: Path = path_field("~/cheri/cheri-perfetto/build")
    cheribuild_path: Path = path_field("~/cheri/cheribuild/cheribuild.py")
    cheribsd_path: Path = path_field("~/cheri/cheribsd")
    qemu_path: Path = path_field("~/cheri/qemu")


@dataclass
class BenchmarkSessionConfig(TemplateConfig):
    """
    Describe the benchmarks to run in the current benchplot session.
    """
    verbose: bool = False
    ssh_key: Path = Path("~/.ssh/id_rsa")
    output_path: Path = field(default_factory=Path.cwd)
    concurrent_instances: int = 0
    instances: list[InstanceConfig] = field(default_factory=list)
    benchmarks: list[BenchmarkRunConfig] = field(default_factory=list)


#class BenchmarkManagerConfig(TemplateConfig):
@dataclass
class BenchmarkManagerConfig(BenchplotUserConfig, BenchmarkSessionConfig):
    """
    Internal configuration object merging user and session top-level configurations.
    This is the top-level configuration object passed around as manager_config.
    """
    def __post_init__(self):
        super().__post_init__()
        # Verify basic sanity of the configuration
        if len(self.benchmarks) == 0:
            raise ValueError("No benchmarks specified in configuration")
        if len(self.instances) == 0:
            raise ValueError("No instances specified in configuration")
        if self.concurrent_instances == 0:
            self.concurrent_instances = multiprocessing.cpu_count()
        if self.concurrent_instances <= 0:
            raise ValueError("Negative max concurrent instances in configuration")


@dataclass
class BenchmarkManagerRecord(Config):
    session: uuid.UUID
    cheribsd_head: typing.Optional[str] = None
    qemu_head: typing.Optional[str] = None
    llvm_head: typing.Optional[str] = None
    records: list[BenchmarkRunRecord] = field(default_factory=list)


class BenchmarkManager(TemplateConfigContext):
    records_filename = "benchplot-run.json"

    def __init__(self, user_config: BenchplotUserConfig, config: BenchmarkSessionConfig):
        super().__init__()
        # Merge configurations
        manager_config = BenchmarkManagerConfig.merge(user_config, config)
        # Cached copy of the configuration template
        self._config_template = manager_config
        # The ID for this benchplot session
        self.session = uuid.uuid4()
        self.logger = new_logger("manager")
        self.loop = aio.get_event_loop()
        self.instance_manager = InstanceManager(self.loop, manager_config)
        self.benchmark_instances = {}
        self.failed_benchmarks = []
        # Task queue for the operations that the manager should run
        self.queued_tasks = []
        # Cleanup callbacks
        self.cleanup_callbacks = []

        self._init_session()
        self.logger.debug("Assign initial session %s", self.session)
        # Adjust libraries log level
        if not self.config.verbose:
            ssh_logger = logging.getLogger("asyncssh")
            ssh_logger.setLevel(logging.WARNING)
        matplotlib_logger = logging.getLogger("matplotlib")
        matplotlib_logger.setLevel(logging.WARNING)

        self.logger.debug("Registered datasets %s", [str(k) for k in DatasetRegistry.dataset_types.keys()])
        self.logger.debug("Registered analysis %s", BenchmarkAnalysisRegistry.analysis_steps)

    def record_benchmark(self, record: BenchmarkRunRecord):
        self.benchmark_records.records.append(record)

    def _get_repo_head(self, path):
        repo = Repo(path)
        head = repo.head.object.hexsha
        dirty = "-dirty" if repo.is_dirty() else ""
        return f"{head}{dirty}"

    def _init_session(self):
        """Session-ID dependant initialization"""
        # Note: this will only bind the manager-specific options, the rest of the template arguments
        # will remain as they will need to be bound to specific benchmark instances.
        self.register_template_subst(session=self.session)
        self.config = self._config_template.bind(self)
        self.session_output_path = self.config.output_path / f"benchplot-session-{str(self.session)}"
        self.benchmark_records = BenchmarkManagerRecord(session=self.session)
        self.benchmark_records_path = self.session_output_path / self.records_filename
        self.analysis_config = AnalysisConfig()
        self.plot_output_path = self.session_output_path / "plots"

        try:
            self.benchmark_records.cheribsd_head = self._get_repo_head(self.config.src_path / "cheribsd")
        except:
            self.logger.warning("Could not record CheriBSD HEAD state, consider setting `src_path`" +
                                " in the session configuration")
        try:
            self.benchmark_records.qemu_head = self._get_repo_head(self.config.src_path / "qemu")
        except:
            self.logger.warning("Could not record QEMU HEAD state, consider setting `src_path`" +
                                " in the session configuration")
        try:
            self.benchmark_records.llvm_head = self._get_repo_head(self.config.src_path / "llvm-project")
        except:
            self.logger.warning("Could not record LLVM HEAD state, consider setting `src_path`" +
                                " in the session configuration")

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
        sessions = []
        for next_dir in self._iter_output_session_dirs():
            record_file = next_dir / self.records_filename
            if not record_file.exists():
                continue
            record = BenchmarkManagerRecord.load_json(record_file)
            if session is not None and session == record.session:
                resolved = record
                break
            fstat = record_file.stat()
            mtime = datetime.fromtimestamp(fstat.st_mtime, tz=timezone.utc)
            sessions.append((mtime, record))
        if resolved is None and len(sessions):
            recent_session = sorted(sessions, key=lambda tup: tup[0], reverse=True)[0]
            resolved = recent_session[1]
        if resolved is None:
            self.logger.error("Can not resolve benchmark session %s in %s",
                              session if session is not None else "DEFAULT", self.config.output_path)
            raise Exception("Benchmark session not found")
        self.session = resolved.session
        self._init_session()
        # Overwrite benchmark records with the resolved data
        self.benchmark_records = resolved

    def _iter_output_session_dirs(self):
        if not self.config.output_path.is_dir():
            self.logger.error("Output directory %s does not exist", self.config.output_path)
            raise OSError("Output directory not found")
        for next_dir in self.config.output_path.glob("benchplot-session-*"):
            if not next_dir.is_dir():
                continue
            yield next_dir

    def _emit_records(self):
        self.logger.debug("Emit benchmark records")
        with open(self.benchmark_records_path, "w") as record_file:
            record_file.write(self.benchmark_records.to_json(indent=4))

    def create_benchmark(self, bench_config: BenchmarkRunConfig, instance: InstanceConfig, uid: uuid.UUID = None):
        """Create a benchmark run on an instance"""
        bench = BenchmarkBase(self, bench_config, instance, run_id=uid)
        self.benchmark_instances[bench.uuid] = bench
        self.logger.debug("Created benchmark run %s on %s id=%s", bench_config.name, instance.name, bench.uuid)
        return bench

    def _interactive_analysis(self):
        self.logger.info("Enter interactive analysis")
        local_env = {"pd": pandas, "manager": self}
        try:
            code.interact(local=local_env)
        except aio.CancelledError:
            raise
        except Exception as ex:
            self.logger.exception("Exiting interactive analysis with error")
        self.logger.info("Interactive analysis done")

    async def _run_tasks(self):
        await aio.gather(*self.queued_tasks)
        await self.instance_manager.shutdown()
        for cbk in self.cleanup_callbacks:
            cbk()

    async def _kill_tasks(self):
        for t in self.queued_tasks:
            t.cancel()
        await aio.gather(*self.queued_tasks, return_exceptions=True)
        await self.instance_manager.kill()
        for cbk in self.cleanup_callbacks:
            try:
                cbk()
            except ex:
                self.logger.exception("Cleanup callback failed, skipping")

    async def _list_task(self):
        self.logger.debug("List recorded sessions at %s", self.config.output_path)
        sessions = []
        for next_dir in self._iter_output_session_dirs():
            record_file = next_dir / self.records_filename
            if not record_file.exists():
                continue
            records = BenchmarkManagerRecord.load_json(record_file)
            fstat = record_file.stat()
            mtime = datetime.fromtimestamp(fstat.st_mtime, tz=timezone.utc)
            sessions.append((mtime, records))
        sessions = sorted(sessions, key=lambda tup: tup[0], reverse=True)
        for mtime, records in sessions:
            if records == sessions[0][1]:
                is_default = " (default)"
            else:
                is_default = ""
            print(termcolor.colored(f"SESSION {records.session} [{mtime:%d-%m-%Y %H:%M}]{is_default}", "blue"))
            for bench_record in records.records:
                print(f"\t{bench_record.run.name}:{bench_record.run.benchmark_dataset.type} on instance " +
                      f"{bench_record.instance.name} ({bench_record.uuid})")

    async def _clean_task(self):
        self.logger.debug("Clean all sessions from the output directory")
        for next_dir in self._iter_output_session_dirs():
            shutil.rmtree(next_dir)

    async def _analysis_task(self, interactive_step=None, config_path: Path = None):
        # Find all benchmark variants we were supposed to run
        # Note: this assumes that we aggregate to compare the same benchmark across OS configs,
        # it can be easily changed to also support comparison of different benchmark runs on
        # the same instance configuration if needed.
        # Load all datasets and for each benchmark and find the baseline instance for
        # each benchmark variant
        aggregate_baseline = {}
        aggregate_groups = defaultdict(list)

        if config_path:
            self.analysis_config = AnalysisConfig.load_json(config_path)

        # Ensure that the plot subdir exists
        self.plot_output_path.mkdir(exist_ok=True)

        for record in self.benchmark_records.records:
            bench = self.create_benchmark(record.run, record.instance, record.uuid)
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
        # Load datasets concurrently
        loading_tasks = []
        self.logger.info("Loading datasets")
        try:
            for bench in it.chain(aggregate_baseline.values(), it.chain.from_iterable(aggregate_groups.values())):
                if interactive_step == "load":
                    executor_fn = bench.load
                else:
                    executor_fn = bench.load_and_pre_merge
                bench.task = self.loop.run_in_executor(None, executor_fn)
                loading_tasks.append(bench.task)
                # Wait for everything to have loaded
            await aio.gather(*loading_tasks)
            if interactive_step == "load" or interactive_step == "pre-merge":
                self._interactive_analysis()
                return
        except aio.CancelledError as ex:
            # Cancel any pending loading
            for task in loading_tasks:
                task.cancel()
            await aio.gather(*loading_tasks, return_exceptions=True)
            raise ex
        self.logger.info("Merge datasets")
        self.logger.debug("Benchmark aggregation baselines: %s", {k: b.uuid for k, b in aggregate_baseline.items()})
        # Merge compatible benchmark datasets into the baseline instance
        for name, baseline_bench in aggregate_baseline.items():
            baseline_bench.merge(aggregate_groups[name])
        if interactive_step == "merge":
            self._interactive_analysis()
            return
        # From now on we ony operate on the merged data
        self.logger.info("Aggregate datasets")
        for bench in aggregate_baseline.values():
            bench.aggregate()
        if interactive_step == "aggregate":
            self._interactive_analysis()
            return
        self.logger.info("Run analysis steps")
        for bench in aggregate_baseline.values():
            bench.analyse()

    def _handle_run_command(self, args: ap.Namespace):
        self.session_output_path.mkdir(parents=True)
        self.logger.info("Start benchplot session %s", self.session)
        for conf in self.config.benchmarks:
            self.logger.debug("Found benchmark %s", conf.name)
            for inst_conf in self.config.instances:
                bench = self.create_benchmark(conf, inst_conf)
                bench.task = self.loop.create_task(bench.run())
                self.queued_tasks.append(bench.task)

    def _handle_analysis_command(self, args: ap.Namespace):
        self._resolve_recorded_session(args.session)
        self.logger.info("Using recorded session %s", self.session)
        task = self.loop.create_task(self._analysis_task(args.interactive, args.analysis_config))
        self.queued_tasks.append(task)

    def _handle_interactive_analysis_command(self, args: ap.Namespace):
        self._resolve_recorded_session(args.session)

    def _handle_list_command(self, args: ap.Namespace):
        self.queued_tasks.append(self.loop.create_task(self._list_task()))

    def _handle_clean_command(self, args: ap.Namespace):
        self.queued_tasks.append(self.loop.create_task(self._clean_task()))

    def run(self, args: ap.Namespace):
        """Main entry point to execute benchmark tasks."""
        command = args.command
        if command == "run":
            self._handle_run_command(args)
        elif command == "analyse":
            self._handle_analysis_command(args)
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
            self.loop.run_until_complete(self._kill_tasks())
        except Exception as ex:
            self.logger.exception("Died because of error: %s", ex)
            self.loop.run_until_complete(self._kill_tasks())
        finally:
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()
            self.logger.info("All tasks finished")
