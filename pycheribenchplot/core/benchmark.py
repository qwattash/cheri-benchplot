import asyncio as aio
import operator as op
import shlex
import typing
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import asyncssh
import pandas as pd

from .analysis import BenchmarkAnalysisRegistry
from .config import AnalysisConfig, DatasetArtefact, DatasetConfig, DatasetName
from .dataset import DataSetContainer, DatasetRegistry
from .elf import AddressSpaceManager, DWARFHelper
from .instance import InstanceConfig, InstanceInfo, InstanceManager
from .shellgen import ShellScriptBuilder
from .util import new_logger, timing


class BenchmarkError(Exception):
    def __init__(self, benchmark, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark = benchmark
        self.benchmark.logger.error(str(self))

    def __str__(self):
        msg = super().__str__()
        return f"BenchmarkError: {msg} on benchmark instance {self.benchmark.uuid}"


class Benchmark:
    """
    New benchmark handler implementation

    :param session: The parent session
    :param config: The benchmark run configuration allocated to this handler.
    """
    def __init__(self, session: "PipelineSession", config: "BenchmarkRunConfig"):
        self.session = session
        self.config = config
        self.logger = new_logger(f"{self.config.name}:{self.config.instance.name}")

        # Map uuids to benchmarks that have been merged into the current instance (which is the baseline)
        # so that we can look them up if necessary
        self.merged_benchmarks = {}
        # Cross-benchmark parameterisation merged instances
        self.cross_merged = {}
        # Symbol mapping handler for this benchmark instance
        self.sym_resolver = AddressSpaceManager(self)
        # Dwarf information extraction helper
        self.dwarf_helper = DWARFHelper(self)

        self.get_benchmark_data_path().mkdir(exist_ok=True)
        self._dataset_modules = self._collect_dataset_modules()
        self._dataset_runner_modules = self._collect_runner_modules()
        self._configure_datasets()

    def __str__(self):
        return f"{self.config.name}({self.uuid})"

    def __repr__(self):
        return str(self)

    def _find_dataset_module(self, config: DatasetConfig):
        """Resolve the parser for the given dataset"""
        ds_name = DatasetName(config.handler)
        handler_class = DatasetRegistry.resolve_name(ds_name)
        handler = handler_class(self, config)
        return handler

    def _collect_dataset_modules(self) -> typing.Dict[DatasetArtefact, DataSetContainer]:
        """
        Helper to resolve all dataset handlers for this benchmark setup.
        This will return a dictionary mapping DatasetName => :class:`DataSetContainer`

        :return: A mapping of dataset ID to the dataset module responsible to analyse the data.
        """
        modules = {}
        # The main benchmark dataset
        # XXX we should mark these somehow, to avoid confusion
        modules[self.config.benchmark.handler] = self._find_dataset_module(self.config.benchmark)
        # Implicit auxiliary datasets
        modules[DatasetName.PIDMAP] = self._find_dataset_module(DatasetConfig(handler=DatasetName.PIDMAP))
        # Extra datasets configured
        for aux_config in self.config.aux_dataset_handlers:
            assert aux_config.handler not in modules, "Duplicate AUX dataset handler"
            modules[aux_config.handler] = self._find_dataset_module(aux_config)
        return modules

    def _collect_runner_modules(self) -> typing.Dict[DatasetArtefact, DataSetContainer]:
        """
        Helper to resolve all dataset handlers that will produce data during a benchmark run.
        This is necessary to deduplicate any handler that require the same input data but we
        only need to generate the data once.
        If multiple datasets have the same dataset_source_id, assume it to be equivalent and
        just pick one of them (this is the case for commands that produce multiple datasets).

        :return: A mapping of dataset artefact ID to the dataset module responsible to generate the data
        """
        runners = {}
        for dset in self._dataset_modules.values():
            if dset.dataset_source_id not in runners:
                runners[dset.dataset_source_id] = dset
        return runners

    def _collect_analysers(self, analysis_config: AnalysisConfig, cross_analysis: bool):
        """
        Helper to resolve all analysis steps that need to be run for this benchmark set.
        Note that this will be called only on the instance where the merged dataframes are built.
        """
        analysers = []
        datasets = set(self._dataset_modules.keys())
        for handler_class in BenchmarkAnalysisRegistry.analysis_steps.values():
            if handler_class.cross_analysis != cross_analysis:
                continue
            # First check if we have the minimal requirements
            if callable(handler_class.require):
                checker = handler_class.require
            else:
                checker = lambda dss, conf: handler_class.require.issubset(dss)
            if not checker(datasets, analysis_config):
                continue
            # Check for config enable
            if (handler_class.name in analysis_config.enable
                    or handler_class.tags.issubset(analysis_config.enable_tags)):
                analysers.append(handler_class(self))
        return analysers

    def _configure_datasets(self):
        """Resolve platform options for the instance configuration and finalize dataset configuration"""
        opts = self.config.instance.platform_options
        for dset in self._dataset_modules.values():
            opts = dset.configure(opts)

    def _build_remote_script(self) -> ShellScriptBuilder:
        """
        Helper to build a new benchmark run script based on the current configuration.

        :return: The local path to the runner script
        """
        script = ShellScriptBuilder(self)

        bench_handler = self.get_dataset(DatasetName(self.config.benchmark.handler))
        assert bench_handler, "Missing benchmark dataset"
        pre_generators = sorted(self._dataset_runner_modules.values(), key=lambda ds: ds.dataset_run_order)
        post_generators = sorted(self._dataset_runner_modules.values(),
                                 key=lambda ds: ds.dataset_run_order,
                                 reverse=True)
        self.logger.info("Generate benchmark script")

        self._gen_hooks(script, "pre_benchmark")
        for dset in pre_generators:
            dset.gen_pre_benchmark(script)

        for i in range(self.config.iterations):
            self.logger.info("Generate benchmark iteration %d", i)
            for dset in pre_generators:
                dset.configure_iteration(script, i)
            self._gen_hooks(script, "pre_benchmark_iter")
            for dset in pre_generators:
                dset.gen_pre_benchmark_iter(script, i)
            # Only run the benchmark step for the given benchmark_dataset
            bench_handler.gen_benchmark(script, i)
            self._gen_hooks(script, "post_benchmark_iter")
            for dset in post_generators:
                dset.gen_post_benchmark_iter(script, i)

        self._gen_hooks(script, "post_benchmark")
        for dset in post_generators:
            dset.gen_post_benchmark(script)
        for dset in post_generators:
            dset.gen_pre_extract_results(script)

        script_path = self.get_run_script_path()
        with open(script_path, "w+") as script_file:
            script.to_shell_script(script_file)
        script_path.chmod(0o755)
        return script

    def _gen_hooks(self, script: ShellScriptBuilder, name: str):
        """
        Generate script commands from configured hooks. The given name is the
        hook key in the configuration dictionary.

        :param name: Hook name (as appears in the configuration file)
        """
        commands = self.config.command_hooks.get(name, [])
        for cmd in commands:
            cmd_args = shlex.split(cmd)
            script.gen_cmd(cmd_args[0], cmd_args[1:])

    async def _extract_results(self, script: ShellScriptBuilder, instance: InstanceInfo):
        """
        Helper to extract results from a connected instance

        :param script: The script generator
        :param instance: Instance handler from the :class:`InstanceManager`
        """
        for remote_path, local_path, custom_extract_fn in script.get_extract_files():
            self.logger.debug("Extract %s -> %s", remote_path, local_path)
            if custom_extract_fn:
                await custom_extract_fn(instance, script, remote_path, local_path)
            else:
                await instance.extract_file(remote_path, local_path)

        # Extract also the implicit command history
        cmd_history = script.command_history_path()
        remote_cmd_history = script.local_to_remote_path(cmd_history)
        self.logger.debug("Extract %s -> %s", remote_cmd_history, cmd_history)
        await instance.extract_file(remote_cmd_history, cmd_history)

    def _load_kernel_symbols(self):
        # Boot kernel location
        kernel = self.cheribsd_rootfs_path / "boot" / f"kernel.{self.config.instance.kernel}" / "kernel.full"
        if not kernel.exists():
            # FPGA kernels fallback location
            kernel = self.cheribsd_rootfs_path / f"kernel.{self.config.instance.kernel}.full"
        if not kernel.exists():
            self.logger.warning("Kernel name not found in kernel.<CONF> directories, using the default kernel")
            kernel = self.cheribsd_rootfs_path / "boot" / "kernel" / "kernel.full"
        self.sym_resolver.register_address_space("kernel.full", shared=True)
        self.sym_resolver.add_mapped_binary("kernel.full", 0, kernel)
        arch_pointer_size = self.config.instance.kernel_pointer_size
        self.dwarf_helper.register_object("kernel.full", kernel, arch_pointer_size)

    @property
    def uuid(self):
        return self.config.uuid

    @property
    def g_uuid(self):
        return self.config.g_uuid

    @property
    def user_config(self):
        return self.session.user_config

    @property
    def analysis_config(self):
        return self.session.analysis_config

    @property
    def cheribsd_rootfs_path(self):
        rootfs_path = self.user_config.sdk_path / f"rootfs-{self.config.instance.cheri_target}"
        if not rootfs_path.exists() or not rootfs_path.is_dir():
            raise ValueError(f"Invalid rootfs path {rootfs_path} for benchmark instance")
        return rootfs_path

    def get_run_script_path(self) -> Path:
        """
        :return: The path to the run script to import to the guest for this benchmark
        """
        return self.get_benchmark_data_path() / "runner-script.sh"

    def get_benchmark_data_path(self) -> Path:
        """
        :return: The output directory for run data corresponding to this benchmark configuration.
        """
        return self.session.get_data_root_path() / f"{self.uuid}"

    def get_plot_path(self):
        """
        Get base plot path for the current benchmark instance.
        Every plot should use this path as the base path to generate plots.
        """
        return self.session.get_plot_root_path() / self.config.name

    def get_benchmark_iter_data_path(self, iteration: int) -> Path:
        """
        :return: The output directory for run data corresponding to this benchmark configuration and
        a specific benchmark iteration.
        """
        iter_path = self.get_benchmark_data_path() / str(iteration)
        iter_path.mkdir(exist_ok=True)
        return iter_path

    def get_dataset(self, name: DatasetName) -> typing.Optional[DataSetContainer]:
        """
        :return: The instance within the current benchmark for given dataset module name.
        """
        return self._dataset_modules.get(name, None)

    def get_dataset_by_artefact(self, ds_id: DatasetArtefact) -> typing.Optional[DataSetContainer]:
        """
        Lookup a generic dataset by the artefact ID.
        Note that this will fail if there are multiple matches
        XXX do we need this?
        """
        match = [dset for dset in self._dataset_modules.values() if dset.dataset_source_id == ds_id]
        if len(match) > 1:
            raise KeyError("Multiple matching dataset for artefact %s", ds_id)
        if len(match):
            return match[0]
        return None

    def get_merged_benchmarks(self) -> typing.Dict[UUID, "Benchmark"]:
        """
        Find the set of benchmarks that have been merged or will be merged to the current benchmark
        instance. This is relevant only for baseline instances, as other instances will be merged
        to the corresponding baseline :class:`Benchmark` instance.

        :return: a dict mapping the UUID to the benchmark instances that are merged to the current
        instance. This includes the current instance.
        """
        assert self.g_uuid == self.session.baseline_g_uuid,\
            "Benchmark groups are meaningful only on the baseline instance"
        row = self.session.benchmark_matrix[self.g_uuid] == self
        assert row.sum() == 1
        return {b.uuid: b for b in self.session.benchmark_matrix.loc[row, :]}

    def get_benchmark_groups(self) -> typing.Dict[UUID, "Benchmark"]:
        """
        Return the benchmark matrix columns as a mapping of group UUIDs to the list of benchmarks
        for each group (instance configuration).
        """
        assert self.g_uuid == self.session.baseline_g_uuid,\
            "Benchmark groups are meaningful only on the baseline instance"
        groups = self.session.benchmark_matrix.to_dict(orient="list")
        return groups

    def register_mapped_binary(self, map_addr: int, path: Path, pid: int):
        """
        Add a binary to the symbolizer for this benchmark.
        The symbols will be associated with the given PID address space

        :param map_addr: Base load address
        :param path: Host path to the binary ELF file
        :param pid: PID of the process associated to the binary
        """
        # bench_dset = self.get_dataset(self.config.benchmark_dataset.type)
        # addrspace = bench_dset.get_addrspace_key()
        pidmap = self.get_dataset_by_artefact(DatasetArtefact.PIDMAP)
        assert pidmap is not None
        cmd = pidmap.df[pidmap.df["pid"] == pid]
        if len(cmd) > 1:
            # XXX PID reuse is not supported yet
            self.logger.warning("Multiple commands for pid %d (%s), PID reuse not yet supported", pid, cmd)
            return
        elif len(cmd) == 0:
            self.logger.warning("No commands for pid %d, can not register binary", pid)
            return
        cmd = Path(cmd["command"][0])
        addrspace_key = cmd.name
        self.logger.debug("Register binary mapping %s at 0x%x for %s", path, map_addr, addrspace_key)
        # XXX We should use both PID and command name to distinguish between different runs?
        self.sym_resolver.register_address_space(cmd.name)
        self.sym_resolver.add_mapped_binary(cmd.name, map_addr, path)
        self.dwarf_helper.register_object(path.name, path)

    async def run(self, instance_manager: InstanceManager) -> bool:
        """
        Run the benchmark and extract the results.
        This involves the steps:
        1. Build the remote run script
        2. Ask the instance manager for an instance that we can run on
        3. Connect to the instance, upload the run script and run it to completion.
        4. Extract results

        :param instance_manager: The global instance manager
        :return: True if the benchmark completed successfully
        """
        script_path = self.get_run_script_path()
        remote_script = Path(f"{self.config.name}-{self.uuid}.sh")
        script = self._build_remote_script()
        self.logger.info("Waiting for instance")
        # instance_reservation = instance_manager.request(self.uuid, self.config.instance)
        # async with instance_reservation as reserved_instance:

        instance = await instance_manager.request_instance(self.uuid, self.config.instance)
        try:
            await instance.connect()
            self.logger.debug("Import script file host:%s => guest:%s", script_path, remote_script)
            await instance.import_file(script_path, remote_script, preserve=True)
            # await instance.import_file("/home/qwattash/cheri/cheri-benchplot/test.sh", remote_script, preserve=True)
            self.logger.info("Execute benchmark script verbose=%s", self.user_config.verbose)
            with timing("Benchmark script completed", logger=self.logger):
                await instance.run_cmd("sh", [remote_script], verbose=self.user_config.verbose)
            await self._extract_results(script, instance)

            self.logger.info("Generate extra datasets")
            for dset in sorted(self._dataset_runner_modules.values(), key=op.attrgetter("dataset_run_order")):
                await dset.after_extract_results(script, instance)

            # Record successful run and cleanup any pending background task
            # self._record_benchmark_run()
            return True
        except Exception as ex:
            self.logger.exception("Benchmark run failed: %s", ex)
        finally:
            self.logger.info("Benchmark run completed")
            await instance_manager.release_instance(self.uuid)
        return False

    def load(self):
        """
        Setup benchmark metadata and load results into datasets from the currently assigned run configuration.
        Note: this runs in the aio loop executor asyncronously
        """
        self._load_kernel_symbols()
        # Always load the pidmap dataset first
        self.logger.info("Loading pidmap data")
        self._dataset_modules[DatasetName.PIDMAP].load()
        for dset in self._dataset_modules.values():
            if dset.dataset_source_id == DatasetArtefact.PIDMAP:
                # We loaded it before
                continue
            self.logger.info("Loading %s data, dropping first %d iterations", dset.name, self.config.drop_iterations)
            for i in range(self.config.iterations):
                try:
                    if i >= self.config.drop_iterations:
                        dset.load_iteration(i)
                except Exception:
                    self.logger.exception("Failed to load data for %s iteration %d", dset.name, i)
            dset.load()

    def pre_merge(self):
        """
        Perform pre-processing step for all datasets. This may generate derived fields.
        Note: this runs in the aio loop executor asyncronously
        """
        for dset in self._dataset_modules.values():
            self.logger.info("Pre-process %s", dset.name)
            dset.pre_merge()

    def load_and_pre_merge(self):
        """Shortcut to perform both the load and pre-merge steps"""
        self.load()
        self.pre_merge()

    def merge(self, others: typing.List["Benchmark"]):
        """
        Merge datasets from compatible runs into a single dataset.
        Note that this is called only on the baseline benchmark instance
        """
        self.logger.debug("Merge datasets %s onto baseline %s", [str(b) for b in others], self.uuid)
        for dset in self._dataset_modules.values():
            dset.init_merge()
        for bench in others:
            self.logger.debug("Merge %s(%s) instance='%s'", bench.config.name, bench.uuid, bench.config.instance.name)
            self.merged_benchmarks[bench.uuid] = bench
            for parser_id, dset in bench._dataset_modules.items():
                self._dataset_modules[parser_id].merge(dset)
        for dset in self._dataset_modules.values():
            dset.post_merge()

    def aggregate(self):
        """
        Generate dataset aggregates (e.g. mean and quartiles)
        Note that this is called only on the baseline benchmark instance
        """
        self.logger.debug("Aggregate datasets %s", self.config.name)
        for dset in self._dataset_modules.values():
            dset.aggregate()
            dset.post_aggregate()

    def analyse(self, analysis_config: AnalysisConfig):
        """
        Run analysis steps on this benchmark. This includes plotting.
        Currently there is no ordering guarantee among analysis steps.
        Note that this is called only on the baseline benchmark instance
        """
        self.logger.debug("Analise %s", self.config.name)
        analysers = self._collect_analysers(analysis_config, cross_analysis=False)
        self.logger.debug("Resolved analysis steps %s", [str(a) for a in analysers])
        for handler in analysers:
            handler.process_datasets()

    def cross_merge(self, others: typing.List["Benchmark"]):
        """
        Merge the aggregated frames from all other benchmark parametrized variants.
        """
        self.logger.debug("Cross merge parameterized variants %s onto %s", [b for b in others], self.uuid)
        for dset in self._dataset_modules.values():
            dset.init_cross_merge()
        for bench in others:
            self.logger.debug("Merge %s(%s) param-set: %s", bench.config.name, bench.uuid, bench.config.parameters)
            self.cross_merged[bench.uuid] = bench
            for parser_id, dset in bench._dataset_modules.items():
                self._dataset_modules[parser_id].cross_merge(dset)
        for dset in self._dataset_modules.values():
            dset.post_cross_merge()

    def cross_analysis(self, analysis_config: AnalysisConfig):
        """
        Perform any analysis steps on the merged frame with all the parameterized
        benchmark variants.
        """
        self.logger.debug("Cross-analise")
        analysers = self._collect_analysers(analysis_config, cross_analysis=True)
        self.logger.debug("Resolved cross analysis steps %s", [str(a) for a in analysers])
        for handler in analysers:
            handler.process_datasets()
