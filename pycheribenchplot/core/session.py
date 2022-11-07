import re
import shutil
import typing
from collections import defaultdict, namedtuple
from contextlib import AbstractContextManager
from dataclasses import asdict, fields
from enum import Enum
from pathlib import Path
from uuid import UUID

import pandas as pd

from .benchmark import Benchmark, BenchmarkExecMode, ExecTaskConfig
from .config import (AnalysisConfig, Config, PipelineConfig, SessionRunConfig, TaskTargetConfig, TemplateConfigContext)
from .instance import InstanceManager
from .model import UUIDType
from .task import AnalysisTask, TaskRegistry, TaskScheduler
from .util import new_logger

SESSION_RUN_FILE = "session-run.json"


class SessionAnalysisMode(Enum):
    """
    Analysis mode determines the type of run strategy to use for the
    processing steps.
    """
    BATCH = "batch"
    INTERACTIVE_LOAD = "interactive-load"
    INTERACTIVE_PREMERGE = "interactive-pre-merge"
    INTERACTIVE_MERGE = "interactive-merge"
    INTERACTIVE_AGG = "interactive-aggregate"
    INTERACTIVE_XMERGE = "interactive-xmerge"


#: Constant name to mark the benchmark matrix index as unparameterized
UNPARAMETERIZED_INDEX_NAME = "RESERVED__unparameterized_index"


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
        :param session_path: The session_path of the new session
        :param config: The session configuration
        :return: A new session instance
        """
        if session_path.exists():
            mgr.logger.error("Session directory already exists for session %s", session_path)
            raise ValueError("New session path already exists")
        run_config = SessionRunConfig.generate(mgr, config)
        run_config.name = session_path.name
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
        # XXX debatable whether we want the manager here
        self.manager = manager
        # Now resolve the configuration templates, before doing anything in the session
        #: Main session configuration
        self.config = self._resolve_config_template(config)
        self.logger = new_logger(f"session-{self.config.name}")
        if session_path is None:
            session_root_path = self.manager.user_config.session_path / self.config.name
        else:
            session_root_path = session_path
        #: Session root path, where all the session data will be stored
        self.session_root_path = session_root_path.expanduser().absolute()
        self._ensure_dir_tree()
        #: A dataframe that organises the set of benchmarks to run or analyse.
        self.benchmark_matrix = self._resolve_benchmark_matrix()
        #: Benchmark baseline instance group UUID.
        self.baseline_g_uuid = self._resolve_baseline()
        #: Task scheduler
        self.scheduler = TaskScheduler(self)

    def __str__(self):
        return f"Session({self.uuid}) [{self.name}]"

    def _resolve_task_options(self, config: TaskTargetConfig) -> typing.Optional[Config]:
        """
        Resolve the configuration type for a :class:`TaskTargetConfig` containing run options.

        :param config: A single dataset configuration
        :return: A Config object to be used as the new :attr:`TaskTargetConfig.task_options`.
        If the exec task does not specify a task_config_class, return a dict with the same content as
        the original run_options dict.
        """
        # Existence of the exec task is ensured by configuration validation
        exec_task = TaskRegistry.public_tasks[config.namespace or "exec"][config.name]
        if exec_task.task_config_class:
            return exec_task.task_config_class.schema().load(config.task_options)
        return config.task_options

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
            # Resolve run_options for each TaskTargetConfig
            bench_conf.benchmark.task_options = self._resolve_task_options(bench_conf.benchmark)
            for aux_conf in bench_conf.aux_dataset_handlers:
                aux_conf.task_options = self._resolve_task_options(aux_conf)
            new_bench_conf.append(bench_conf.bind(ctx))
        new_config.configurations = new_bench_conf
        return new_config

    def _resolve_baseline(self):
        """
        Resolve the baseline benchmark run group ID.
        This is necessary to identify the benchmark run that we compare against
        (actually the column in the benchmark matrix we compare against).

        :return: The baseline group ID.
        """
        baseline = None
        for benchmark_config in self.config.configurations:
            if benchmark_config.instance.baseline:
                if baseline and baseline != benchmark_config.g_uuid:
                    self.logger.error("Multiple baseline instances?")
                    raise RuntimeError("Too many baseline specifiers")
                baseline = benchmark_config.g_uuid
        if not baseline:
            self.logger.error("Missing baseline instance")
            raise RuntimeError("Missing baseline")
        baseline_conf = self.benchmark_matrix[baseline].iloc[0].config
        self.logger.debug("Benchmark baseline %s (%s)", baseline_conf.instance.name, baseline)
        return baseline

    def _resolve_benchmark_matrix(self) -> typing.Tuple[pd.DataFrame, UUID]:
        """
        Generate the benchmark matrix from the benchmark configurations.
        In the resulting dataframe:
         - rows are different parameterizations of the benchmark, indexed by param keys.
         - columns are different instances on which the benchmark is run, indexed by dataset_gid.

        :return: The benchmark matrix as a pandas dataframe
        """
        # First pass, just collect instances and parameters
        instances = {}
        parameters = defaultdict(list)
        for benchmark_config in self.config.configurations:
            self.logger.debug("Found benchmark run config %s", benchmark_config)
            instances[benchmark_config.g_uuid] = benchmark_config.instance.name
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
            index = pd.Index([0], name=UNPARAMETERIZED_INDEX_NAME)

        # Second pass, fill the matrix
        bench_matrix = pd.DataFrame(index=index, columns=instances.keys())
        for index_name in bench_matrix.index.names:
            if not re.fullmatch(r"[a-zA-Z0-9_]+", index_name):
                self.logger.exception(
                    "Benchmark parameter keys must be valid python property names, only [a-zA-Z0-9_] allowed")
                raise ValueError("Invalid parameter key")
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
        for i, row in bench_matrix.iterrows():
            if isinstance(i, tuple):
                i = BenchParams(*i)
            self.logger.debug("Benchmark matrix %s = %s", i, row.values)
        if bench_matrix.shape[0] * bench_matrix.shape[1] != len(self.config.configurations):
            self.logger.error("Malformed benchmark matrix")
            raise RuntimeError("Malformed benchmark matrix")

        return bench_matrix

    def _ensure_dir_tree(self):
        """
        Build the session directory tree.
        """
        self.get_data_root_path().mkdir(exist_ok=True)
        self.get_asset_root_path().mkdir(exist_ok=True)
        self.get_plot_root_path().mkdir(exist_ok=True)

    @property
    def name(self):
        return self.config.name

    @property
    def uuid(self):
        return self.config.uuid

    @property
    def user_config(self):
        return self.manager.user_config

    @property
    def parameter_keys(self) -> list[str]:
        """
        The set of parameter keys that index the rows of the benchmark matrix.
        """
        names = self.benchmark_matrix.index.names
        if len(names) == 1 and names[0] == UNPARAMETERIZED_INDEX_NAME:
            return []
        else:
            return list(names)

    def machine_configuration_name(self, g_uuid: UUID) -> str:
        """
        Retrieve the human-readable form of the machine configuration identified by a given UUID.
        This is useful for producing readable output.
        """
        col = self.benchmark_matrix[g_uuid]
        # column shares that same instance configuration, just grab the first
        instance_config = col.iloc[0].config.instance
        return instance_config.name

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

    def get_asset_root_path(self) -> Path:
        """
        :return: The root path for binary assets e.g. kernel images
        """
        return self.session_root_path / "assets"

    def clean(self):
        """
        Clean all output files for a session and ensure that the
        directory hierarchy for the output data is created.
        This will also clean the raw data and cache files.
        """
        data_root = self.get_data_root_path()
        asset_root = self.get_asset_root_path()
        if data_root.exists():
            shutil.rmtree(data_root)
        if asset_root.exists():
            shutil.rmtree(asset_root)
        self._ensure_dir_tree()

    def run(self, mode: str = "full"):
        """
        Run the session benchmark execution task stack

        :param mode: Alternate run mode for partial or pretend runs.
        """
        exec_config = ExecTaskConfig(mode=BenchmarkExecMode(mode))
        instance_manager = InstanceManager(self)
        self.scheduler.register_resource(instance_manager)

        for col in self.benchmark_matrix:
            for bench in self.benchmark_matrix[col]:
                bench.schedule_exec_tasks(self.scheduler, exec_config)
        self.scheduler.run()

    def analyse(self, analysis_config: AnalysisConfig, mode: str = "batch"):
        """
        Run the session analysis tasks requested.
        The analysis pipeline is slighly different from the execution pipeline.
        This is because the same data can be used for different analysis targets and
        the user should be able to specify the set of targets we need to run.
        Here the analysis configuration should specify a set of public task names that
        we collect and schedule

        :param analysis_config: The analysis configuration
        :param mode: The mode to use for the analysis task scheduling. (XXX implement, currently ignored)
        """
        # session_analysis_mode = SessionAnalysisMode(mode)
        # Override the baseline ID if configured
        if (analysis_config.baseline_gid is not None and analysis_config.baseline_gid != self.baseline_g_uuid):
            self.baseline_g_uuid = analysis_config.baseline_gid
            try:
                baseline_conf = self.benchmark_matrix[self.baseline_g_uuid].iloc[0].config
            except KeyError:
                self.logger.error("Invalid baseline instance ID %s", self.baseline_g_uuid)
                raise
            self.logger.info("Using alternative baseline %s (%s)", baseline_conf.instance.name, self.baseline_g_uuid)

        for handler in analysis_config.handlers:
            task_klass = TaskRegistry.public_tasks[handler.namespace].get(handler.name)
            if task_klass is None:
                self.logger.error("Invalid task name specification %s", handler.handler)
                raise ValueError("Invalid task name")
            if not issubclass(task_klass, AnalysisTask):
                self.logger.error("Analysis process only supports scheduling of AnalysisTasks, found %s", task_klass)
                raise TypeError("Invalid task type")
            task = task_klass(self, analysis_config, task_config=handler.task_options)
            self.scheduler.add_task(task)
        self.scheduler.run()
