import asyncio as aio
import shutil
import typing
from collections import defaultdict, namedtuple
from contextlib import AbstractContextManager
from dataclasses import asdict, fields
from pathlib import Path
from uuid import UUID

import pandas as pd

from .benchmark import Benchmark
from .config import (AnalysisConfig, Config, DatasetConfig, PipelineConfig, SessionRunConfig, TemplateConfigContext)
from .dataset import DatasetName, DatasetRegistry
from .instance import InstanceManager
from .perfetto import TraceProcessorCache
from .util import new_logger

SESSION_RUN_FILE = "session-run.json"


class PipelineSession:
    """
    Represent a benchmarking session.
    A session is the primary container for the benchmark assets, results and
    intermediary files.
    When created, a session has at minimum a session descriptor JSON file.
    The session descriptor encodes the session configuration expanded from the
    main benchplot configuration. This instructs the session about the components
    to run and how.
    """
    @classmethod
    def make_new(cls, mgr: "PipelineManager", session_path: Path, config: PipelineConfig) -> "PipelineSession":
        """
        Create a new session and initialize the directory hierarchy

        :param mgr: The parent pipeline manager
        :param name: The name of the new session
        :param config: The session configuration
        :return: A new session instance
        """
        if session_path.exists():
            mgr.logger.error("Session directory already exists for session %s at %s", name, session_path)
            raise ValueError("New session path already exists")
        run_config = SessionRunConfig.generate(mgr, config)
        run_config.name = name
        session_path.mkdir()
        with open(session_path / SESSION_RUN_FILE, "w") as runfile:
            runfile.write(run_config.emit_json())
        return PipelineSession(mgr, run_config, session_path=session_path)

    @classmethod
    def is_session(cls, path: Path) -> bool:
        """
        Check if the given path corresponds to a session.
        This currently checks for the session runfile.

        :param path: The path to a directory
        :return: True if the path contains a session information
        """
        if (path / SESSION_RUN_FILE).exists():
            return True
        return False

    @classmethod
    def from_path(cls, mgr: "PipelineManager", path: Path) -> "PipelineSession":
        """
        Load a session from the given path.

        :param mgr: The parent pipeline manager
        :param path: The session directory path
        :return: The corresponding :class:`PipelineSession`
        """
        assert path.exists()
        assert cls.is_session(path)

        config = SessionRunConfig.load_json(path / SESSION_RUN_FILE)
        return PipelineSession(mgr, config, session_path=path)

    def __init__(self, manager: "PipelineManager", config: SessionRunConfig, session_path: Path = None):
        super().__init__()
        self.manager = manager
        # Now resolve the configuration templates, before doing anything in the session
        self.config = self._resolve_config_template(config)
        self.logger = new_logger(f"session-{self.config.name}")
        if session_path is None:
            self.session_root_path = self.manager.user_config.session_path / self.config.name
        else:
            self.session_root_path = session_path
        # Analysis step configuration, only set when running analysis
        self.analysis_config = None
        # Benchmark baseline instance group UUID
        self.baseline_g_uuid = None
        # Benchmark analysis matrix, only set when running analysis
        self.benchmark_matrix = None

    def __str__(self):
        return f"Session({self.uuid}) [{self.name}]"

    def _resolve_run_options(self, config: DatasetConfig) -> typing.Optional[Config]:
        """
        Resolve the configuration type for a :class:`DatasetConfig` containing run options.

        :param config: A single dataset configuration
        :return: A Config object to be used as the new :attr:`DatasetConfig.run_options`.
        If the dataset does not specify a run_options_class, return a dict with the same content as
        the original run_options dict.
        """
        dataset_class = DatasetRegistry.resolve_name(config.handler)
        if not dataset_class.run_options_class:
            return config.run_options
        run_opts = dataset_class.run_options_class(**config.run_options)
        return run_opts

    def _resolve_config_template(self, config: SessionRunConfig) -> SessionRunConfig:
        """
        Resolves the templates for the given session run configuration,
        using the current user configuration.
        """
        ctx = TemplateConfigContext()
        ctx.register_template_subst(**asdict(self.manager.user_config))
        # Register substitutions for known stable fields in the run configuration
        ctx.register_template_subst(session=config.uuid)
        ctx.register_template_subst(session_name=config.name)
        new_config = config.bind(ctx)
        # Now scan through all the configurations and subsitute per-benchmark/instance fields
        new_bench_conf = []
        for bench_conf in new_config.configurations:
            ctx.register_template_subst(uuid=bench_conf.uuid,
                                        g_uuid=bench_conf.g_uuid,
                                        iterations=bench_conf.iterations,
                                        drop_iterations=bench_conf.drop_iterations,
                                        remote_ouptut_dir=bench_conf.remote_output_dir)
            ctx.register_template_subst(**bench_conf.parameters)
            inst_conf = bench_conf.instance
            ctx.register_template_subst(kernel=inst_conf.kernel,
                                        baseline=inst_conf.baseline,
                                        platform=inst_conf.platform.value,
                                        cheri_target=inst_conf.cheri_target.value,
                                        kernelabi=inst_conf.kernelabi.value)
            # Resolve run_options for each DatasetConfig
            bench_conf.benchmark.run_options = self._resolve_run_options(bench_conf.benchmark)
            for aux_conf in bench_conf.aux_dataset_handlers:
                aux_conf.run_options = self._resolve_run_options(aux_conf)
            new_bench_conf.append(bench_conf.bind(ctx))
        new_config.configurations = new_bench_conf
        return new_config

    @property
    def name(self):
        return self.config.name

    @property
    def uuid(self):
        return self.config.uuid

    @property
    def user_config(self):
        return self.manager.user_config

    def get_data_root_path(self) -> Path:
        """
        :return: The root raw data directory, used to store artifacts from benchmark runs.
        """
        return self.session_root_path / "run"

    def get_plot_root_path(self) -> Path:
        """
        :return: The root path for the generated plots and tables
        """
        return self.session_root_path / "plots"

    def clean(self):
        """
        Clean all output files for a session and ensure that the
        directory hierarchy for the output data is created.
        This will also clean the raw data and cache files.
        """
        data_root = self.get_data_root_path()
        if data_root.exists():
            shutil.rmtree(data_root)
        data_root.mkdir()

    def run(self) -> AbstractContextManager:
        """
        Prepare to run the session.

        :return: A context manager to wrap the main loop and manage
        the tasks for the session.
        """
        ctx = SessionRunContext(self)
        for benchmark_config in self.config.configurations:
            self.logger.debug("Found benchmark run config %s", benchmark_config)
            benchmark = Benchmark(self, benchmark_config)
            ctx.schedule_benchmark(benchmark)
        return ctx

    def analyse(self, analysis_config: AnalysisConfig, interactive: str | None) -> AbstractContextManager:
        """
        Prepare data analysis.

        :return: A context manager to wrap the main loop and manage
        the analysis tasks.
        """
        self.get_plot_root_path().mkdir(exist_ok=True)
        self.analysis_config = analysis_config

        # Resolve benchmark matrix
        # Rows are different parameterizations of the benchmark. Indexed by param keys.
        # Columns are different instances on which the benchmark is run. Indexed by dataset_gid.
        # First pass, just collect instances and parameters
        instances = {}
        baseline = None
        parameters = defaultdict(list)
        for benchmark_config in self.config.configurations:
            self.logger.debug("Found benchmark run config %s", benchmark_config)
            instances[benchmark_config.g_uuid] = benchmark_config.instance.name
            if (analysis_config.baseline_gid == benchmark_config.g_uuid
                    or (analysis_config.baseline_gid is None and benchmark_config.instance.baseline)):
                if baseline and baseline != benchmark_config.g_uuid:
                    self.logger.error("Multiple baseline instances?")
                    raise Exception("Too many baseline specifiers")
                baseline = benchmark_config.g_uuid
            for k, p in benchmark_config.parameters.items():
                parameters[k].append(p)
        if parameters:
            index_frame = pd.DataFrame(parameters)
            index = pd.MultiIndex.from_frame(index_frame)
            if not index.is_unique:
                index = index.unique()
        else:
            # If there is no parameterization, just use a flat index, there will be only
            # one row in the benchmark matrix.
            index = pd.Index([0], name="index")

        # Second pass, fill the matrix
        bench_matrix = pd.DataFrame(index=index, columns=instances.keys())
        BenchParams = namedtuple("BenchParams", bench_matrix.index.names)
        for benchmark_config in self.config.configurations:
            benchmark = Benchmark(self, benchmark_config)
            if benchmark_config.parameters:
                # Note: config.parameters is unordered, use namedtuple to ensure
                # the key ordering
                i = BenchParams(**benchmark_config.parameters)
                bench_matrix.loc[i, benchmark_config.g_uuid] = benchmark
            else:
                bench_matrix[benchmark_config.g_uuid] = benchmark
            benchmark.get_plot_path().mkdir(exist_ok=True)

        assert not bench_matrix.isna().any().any(), "Incomplete benchmark matrix"
        if not baseline:
            self.logger.error("Missing baseline instance")
            raise RuntimeError("Missing baseline")
        self.logger.debug("Benchmark baseline %s (%s)", instances[baseline], baseline)
        for i, row in bench_matrix.iterrows():
            if isinstance(i, tuple):
                i = BenchParams(*i)
            self.logger.debug("Benchmark matrix %s = %s", i, row.values)
        if bench_matrix.shape[0] * bench_matrix.shape[1] != len(self.config.configurations):
            self.logger.error("Malformed benchmark matrix")
            raise RuntimeError("Malformed benchmark matrix")

        self.benchmark_matrix = bench_matrix
        self.baseline_g_uuid = baseline
        ctx = SessionAnalysisContext(self, analysis_config, bench_matrix, baseline)
        if interactive:
            ctx.set_interactive(interactive)
        return ctx


class SessionAnalysisContext(AbstractContextManager):
    """
    Handle the context for a session analysis.
    This abstract the cleanup process and resource allocation for
    the analysis phase.

    :param session: The parent pipeline session
    """
    def __init__(self, session: PipelineSession, analysis_config: AnalysisConfig, benchmark_matrix: pd.DataFrame,
                 baseline: UUID):
        self.session = session
        self.analysis_config = analysis_config
        self.interactive_step = None
        self.loop = aio.get_event_loop()
        self.logger = session.logger
        self._tp_cache = TraceProcessorCache.get_instance(session)
        self._baseline_g_uuid = baseline
        self._benchmark_matrix = benchmark_matrix
        self._tasks = []

    def __exit__(self, ex_type, ex, traceback):
        self.logger.debug("SessionAnalysisContext cleanup")

        if ex and self._tasks:
            self.loop.run_until_complete(self._dirty_shutdown())
        self._tp_cache.shutdown()
        self.logger.info("Analysis complete")

    async def _dirty_shutdown(self):
        """
        Cancel all tasks can wait for cleanup
        """
        for task in self._tasks:
            self.logger.debug("Cancel task %s", task)
            task.cancel()
        await aio.gather(*self._tasks, return_exceptions=True)

    def _interactive_analysis(self):
        self.logger.info("Enter interactive analysis")
        local_env = {"pd": pandas, "ctx": self}
        try:
            code.interact(local=local_env)
        except aio.CancelledError:
            raise
        except Exception as ex:
            self.logger.exception("Exiting interactive analysis with error")
        self.logger.info("Interactive analysis done")

    def set_interactive(self, step: str | None):
        self.logger.info("Interactive analysis will stop at %s", step)
        self.interactive_step = step

    async def main(self):
        """
        Main analysis processing steps.
        Analysis is split in the following phases:
        1. Load data concurrently
        2. Perform any post-load cleanup in the pre_merge step
        3. Merge datasets for the same parameterizations into a single dataframe.
        4. Perform any post-merge processing
        5. Compute aggregated columns (e.g. mean, quartiles...)
        6. Perform post-aggregation processing, usually to produce deltas
        7. Merge the aggregated data into a single dataframe
        8. Perform analysis across all parameterizations
        """
        self.logger.info("Loading datasets")
        for bench in self._benchmark_matrix.values.ravel():
            if self.interactive_step:
                executor_fn = bench.load
            else:
                executor_fn = bench.load_and_pre_merge
            self._tasks.append(self.loop.run_in_executor(None, executor_fn))
        await aio.gather(*self._tasks)
        self._tasks.clear()
        if self.interactive_step == "load" or self.interactive_step == "pre-merge":
            self._interactive_analysis()
            return

        self.logger.info("Merge datasets by param set")
        to_merge = self._benchmark_matrix.columns.difference([self._baseline_g_uuid])
        for params, row in self._benchmark_matrix.iterrows():
            self.logger.debug("Merge param set %s", params)
            row[self._baseline_g_uuid].merge(row[to_merge])
        if self.interactive_step == "merge":
            self._interactive_analysis()
            return
        # From now on we ony operate on the baseline dataset containing the merged data
        self.logger.info("Aggregate datasets")
        for params, row in self._benchmark_matrix.iterrows():
            self.logger.debug("Aggregate param set %s", params)
            row[self._baseline_g_uuid].aggregate()
        if self.interactive_step == "aggregate":
            self._interactive_analysis()
            return
        self.logger.info("Run analysis steps")
        for params, row in self._benchmark_matrix.iterrows():
            self.logger.debug("Analyse param set %s", params)
            row[self._baseline_g_uuid].analyse(self.analysis_config)

        self.logger.info("Run cross-param analysis")
        # Just pick a random instance to perform the cross-parameterization merge and
        # analysis
        to_merge = self._benchmark_matrix[self._baseline_g_uuid]
        target = to_merge.iloc[0]
        target.cross_merge(to_merge.iloc[1:])
        target.cross_analysis(self.analysis_config)


class SessionRunContext(AbstractContextManager):
    """
    Handle the context for a single session run.
    This abstract benchmark scheduling and cleanup.

    :param session: The parent pipeline session
    """
    def __init__(self, session: PipelineSession):
        self.session = session
        self.loop = aio.get_event_loop()
        self.logger = session.logger
        self.instance_manager = InstanceManager(self.session)
        self._tasks = []

    def __exit__(self, ex_type, ex, traceback):
        self.logger.debug("SessionRunContext cleanup")

        if isinstance(ex, KeyboardInterrupt):
            self.logger.debug("SessionRunContext user SIGINT cleanup")
            self.loop.run_until_complete(self._dirty_shutdown())
        elif isinstance(ex, Exception):
            self.logger.debug("SessionRunContext error cleanup")
            self.loop.run_until_complete(self._dirty_shutdown())
        else:
            # normal shutdown
            self.logger.debug("SessionRunContext regular cleanup")
            self.loop.run_until_complete(self._clean_shutdown())
        self.logger.info("Session shutdown complete")

    async def _dirty_shutdown(self):
        """
        Cancel all tasks and wait for the cleanup
        """
        for t in self._tasks:
            self.logger.debug("Cancel task %s", t.get_name())
            t.cancel()
        await aio.gather(*self._tasks, return_exceptions=True)
        await self.instance_manager.kill()

    async def _clean_shutdown(self):
        """
        Stop gracefully background tasks.
        """
        for task in self._tasks:
            if not task.done():
                await self._dirty_shutdown()
                return
        await self.instance_manager.shutdown()

    def schedule_benchmark(self, bench):
        run_task = self.loop.create_task(bench.run(self.instance_manager))
        run_task.set_name(f"Task[{bench.config.name}/{bench.config.instance.name}]")
        self._tasks.append(run_task)

    async def main(self):
        await aio.gather(*self._tasks)
