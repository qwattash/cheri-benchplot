
import logging
import pandas as pd
import uuid
import typing
import asyncio as aio
import asyncssh
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from subprocess import PIPE

from .instanced import InstanceConfig, BenchmarkInfo
from .options import OptionConfig, TemplateConfig, TemplateConfigContext

from .cpu import BenchmarkCPU
from ..pmc import PMCStatData
from ..elf import ELFInfo, SymResolver
from ..netperf.config import NetperfBenchmarkRunConfig

class BenchmarkType(Enum):
    NETPERF = "netperf"

    def __str__(self):
        return self.value


@dataclass
class BenchmarkRunConfig(TemplateConfig):
    name: str
    type: BenchmarkType
    benchmark_options: typing.Union[NetperfBenchmarkRunConfig]
    desc: str = ""
    tags: list[str] = field(default_factory=list)
    env: dict = field(default_factory=dict)
    plots: list[str] = field(default_factory=list)
    output_file: str = "/dev/null"
    extra_files: list[str] = field(default_factory=list)


@dataclass
class BenchmarkManagerConfig(OptionConfig):
    """Describe a set of benchmarks to run on each instance"""
    verbose: bool = False
    ssh_key: Path = Path("~/.ssh/id_rsa")
    output_path: Path = field(default_factory=Path.cwd)
    sdk_path: Path = Path("~/cheri/cherisdk")
    instances: list[InstanceConfig] = field(default_factory=list)
    benchmarks: list[BenchmarkRunConfig] = field(default_factory=list)


class BenchmarkBase(TemplateConfigContext):
    """
    Base class for all the benchmarks
    """

    def __init__(self, manager_config, config, instance_config, instance_daemon):
        self.uuid = uuid.uuid4()
        self.daemon = instance_daemon
        self.manager_config = manager_config
        self.instance_config = instance_config
        self.config = config
        # Do the binding separately so that template params can access the non-template fields
        self.instance_config = self.instance_config.bind(self)
        self.config = self.config.bind(self)

        rootfs_path = self.manager_config.sdk_path / f"rootfs-{self.instance_config.cheri_target}"
        rootfs_path = rootfs_path.expanduser()
        if not rootfs_path.exists() or not rootfs_path.is_dir():
            raise Exception(f"Invalid rootfs path {rootfs_path} for benchmark instance")
        self.rootfs = rootfs_path

        self.result_path = self.manager_config.output_path / str(self.uuid)
        self.result_path.mkdir(parents=True)

        self.logger = logging.getLogger(f"{config.name}:{instance_config.name}:{self.uuid}")
        self._reserved_instance = None  # BenchmarkInfo of the instance the daemon has reserved us
        self._conn = None  # Connection to the CheriBSD instance
        self._command_tasks = []  # Commands being run on the instance

    def conf_template_params(self):
        params = super().conf_template_params()
        params.update(uuid=self.uuid,
                      cheri_target=self.instance_config.cheri_target)
        return params

    async def _cmd_io(self, proc_task, callback):
        try:
            while proc_task.returncode is None:
                out = await proc_task.stdout.readline()
                try:
                    if callback:
                        callback(out)
                except aio.CancelledError as ex:
                    raise ex
                except Exception as ex:
                    self.logger.error("Error while processing output for %s: %s",
                                      proc_task.command, ex)
                self.logger.debug(out)
        except aio.CancelledError as ex:
            proc_task.terminate()
            raise ex
        finally:
            self.logger.info("Background task %s done", proc_task.command)


    async def _run_bg_cmd(self, command: str, args: list, env={}, iocallback=None):
        """Run a background command without waiting for termination"""
        cmdline = f"{command} " + " ".join(args)
        env_str = [f"{k}={v}" for k,v in env.items()]
        self.logger.debug("exec background: %s env=%s", cmdline, env)
        proc_task = await self._conn.create_process(cmdline, env=env_str)
        self._command_tasks.append(aio.create_task(self._cmd_io(proc_task, iocallback)))
        return proc_task

    async def _run_cmd(self, command: str, args: list, env={}, outfile=PIPE):
        """Run a command and wait for the process to complete"""
        cmdline = f"{command} " + " ".join(args)
        env_str = [f"{k}={v}" for k,v in env.items()]
        self.logger.debug("exec: %s env=%s", cmdline, env)
        result = await self._conn.run(cmdline, env=env_str, stdout=outfile)
        if result.returncode != 0:
            if outfile:
                cmdline += f" >> {outfile}"
            self.logger.error("Failed to run %s: %s", command, result.stderr)
        else:
            self.logger.debug("%s done: %s", command, result.stdout)
        return result.returncode

    async def _extract_file(self, guest_src: Path, host_dst: Path):
        """Extract file from instance"""
        src = (self._conn, guest_src)
        await asyncssh.scp(src, host_dst)

    async def _connect_instance(self, info: BenchmarkInfo):
        conn = await asyncssh.connect(info.ssh_host, port=info.ssh_port, known_hosts=None,
                                      client_keys=[self.manager_config.ssh_key], username="root",
                                      passphrase="")
        self.logger.debug("Connected to instance")
        return conn

    async def _run_benchmark(self):
        self.logger.info("Running benchmark")

    async def run(self):
        self.logger.info("Waiting for instance")
        self._reserved_instance = await self.daemon.request_instance(self.uuid, self.instance_config)
        if self._reserved_instance is None:
            self.logger.error("Can not reserve instance, bailing out...")
            return
        try:
            self._conn = await self._connect_instance(self._reserved_instance)
            await self._run_benchmark()
            self.logger.info("Benchmark completed successfully")
            # Stop all pending background processes
            for t in self._command_tasks:
                t.cancel()
            await aio.gather(*self._command_tasks, return_exceptions=True)
        except Exception as ex:
            self.logger.error("Benchmark run failed: %s", ex)
        finally:
            await self.daemon.release_instance(self.uuid, self._reserved_instance)

    def plot(self):
        pass


class _BenchmarkBase:

    def plot(self):
        # Common libpmc input
        self.pmc = PMCStatData.get_pmc_for_cpu(self.cpu, self.options, self)
        """Entry point for plotting benchmark results or analysis files"""
        for dirpath in self.options.stats:
            if not dirpath.exists():
                fatal("Source directory {} does not exist".format(dirpath))
            self._load_dir(dirpath)
        logging.info("Process data")
        self._process_data_sources()
        self.merged_raw_data = self._merge_raw_data()
        self.merged_stats = self._merge_stats()
        logging.info("Generate relative data for baseline %s",
                     self.options.baseline)
        self._compute_relative_stats()
        logging.info("Generate plots")
        self._draw()

    def pmc_map_index(self, df):
        """
        Map the progname and archname columns from statcounters
        to new columns to be used as part of the index, or an empty
        dataframe.
        """
        return pd.DataFrame()

    def _merge_raw_data(self):
        return self.pmc.df

    def _merge_stats(self):
        return self.pmc.stats_df

    def _process_data_sources(self):
        self.pmc.process()

    def _load_dir(self, path):
        for fpath in path.glob("*.csv"):
            self._load_file(path, fpath)

    def _load_file(self, dirpath, filepath):
        logging.info("Loading %s", filepath)
        if self._is_pmc_input(filepath):
            self._load_pmc(filepath)

