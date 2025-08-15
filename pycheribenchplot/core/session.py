import logging
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Type
from uuid import UUID

import polars as pl
from typing_extensions import Self

from .analysis import AnalysisTask, DatasetAnalysisTaskGroup
from .benchmark import Benchmark, ExecTaskConfig
from .config import (AnalysisConfig, BenchplotUserConfig, ConfigContext, InstanceConfig, PipelineConfig,
                     SessionRunConfig)
from .instance import InstanceManager
from .shellgen import TemplateContextBase
from .task import (ExecutionTask, SessionExecutionTask, TaskRegistry, TaskScheduler)
from .util import new_logger

#: Constant name of the generated session configuration file
SESSION_RUN_FILE = "session-run.json"
benchplot_logger = logging.getLogger("cheri-benchplot")

# XXX Is `target` reserved as well?
RESERVED_PARAMETER_NAMES = ["descriptor"]


class Session:
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
    def make_new(cls, user_config: BenchplotUserConfig, config: PipelineConfig, session_path: Path) -> Self:
        """
        Create a new session and initialize the directory hierarchy

        :param user_config: The user configuration file for local configuration options
        :param config: The session configuration
        :param session_path: The session_path of the new session
        :return: A new session instance
        """
        if session_path.exists():
            benchplot_logger.error("Session directory already exists for session %s", session_path)
            raise ValueError("New session path already exists")
        run_config = SessionRunConfig.generate(user_config, config)
        run_config.name = session_path.name
        session_path.mkdir()
        with open(session_path / SESSION_RUN_FILE, "w") as runfile:
            runfile.write(run_config.emit_json())
        return Session(user_config, run_config, session_path=session_path)

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
    def from_path(cls, user_config: BenchplotUserConfig, path: Path) -> Self | None:
        """
        Load a session from the given path.

        :param mgr: The parent pipeline manager
        :param path: The session directory path
        :return: The corresponding :class:`Session`
        """
        benchplot_logger.debug("Scanning %s for valid session", path)

        if not path.exists() or not cls.is_session(path):
            benchplot_logger.debug("Session for %s not found", path)
            return None

        config = SessionRunConfig.load_json(path / SESSION_RUN_FILE)
        session = Session(user_config, config, session_path=path)

        benchplot_logger.debug("Resolved session %s => %s", path, session)
        return session

    def __init__(self, user_config: BenchplotUserConfig, config: SessionRunConfig, session_path: Path = None):
        super().__init__()
        # Pre-initialization logger
        self.logger = new_logger("session-init")
        #: The local user configuration
        self.user_config = user_config
        #: Main session configuration
        self.config = self._resolve_config_template(config)
        #: Root logger for the session, initialized as soon as we know we have a stable session name
        self.logger = new_logger(f"session-{self.config.name}")
        if session_path is None:
            session_root_path = self.user_config.session_path / self.config.name
        else:
            session_root_path = session_path
        #: Session root path, where all the session data will be stored
        self.session_root_path = session_root_path.expanduser().absolute()
        self._ensure_dir_tree()
        #: Mapping from target names to platform configurations.
        #: Note that this should be readonly.
        self.platform_map = {}
        #: A dataframe that organises the set of benchmarks to run or analyse.
        self.parameterization_matrix = self._resolve_parameterization_matrix()

        # Before using the workers configuration, check if we are overriding it
        if self.user_config.concurrent_workers:
            self.logger.debug("Overriding maximum workers count from user configuration (max=%d)",
                              self.user_config.concurrent_workers)
            self.config.concurrent_workers = self.user_config.concurrent_workers

        #: Task scheduler
        self.scheduler = TaskScheduler(self)

    def __str__(self):
        return f"Session({self.uuid}) [{self.name}]"

    def _resolve_config_template(self, config: SessionRunConfig) -> SessionRunConfig:
        """
        Resolves the templates for the given session run configuration,
        using the current user configuration.
        """
        ctx = ConfigContext()
        ctx.add_namespace(self.user_config, "user")
        # Relative path of the assets dir with respect to the benchmark runner scripts
        ctx.add_values(assets="../../assets")
        return config.bind(ctx)

    def _resolve_parameterization_matrix(self) -> pl.DataFrame:
        """
        Generate the parameterization matrix from the generators configurations.

        This is assembled as a dataframe containing the following columns:
         - descriptor: Benchmark descriptor objects
         - target: Built-in parameterization axis for the host system configuration
         - <params>: A variable number of columns depending on the benchmark parameterization.

        :return: The benchmark matrix as a polars dataframe
        """
        # First pass, collect the parameterization axes
        parameters = defaultdict(list)
        for dataset_config in self.config.configurations:
            self.logger.debug("Found benchmark run config %s", dataset_config)
            # Note that g_uuids are used for backward-compatibility only
            self.platform_map[dataset_config.g_uuid] = dataset_config.instance
            for k, p in dataset_config.parameters.items():
                parameters[k].append(p)

        all_parameters = set(parameters.keys())
        dataset_axes = {k: [] for k in all_parameters}
        for name in RESERVED_PARAMETER_NAMES:
            if name in dataset_axes:
                self.logger.error("Invalid parameterization axis: '%s', reserved name.", name)
                raise RuntimeError("Configuration error")
        # Instance identifier column
        instance_axis = []
        # Dataset descriptor object
        descriptors = []

        # Collect all configurations and produce the complete session dataset
        # parameterization table.
        # The invariant is that all configurations must have the same set of
        # parameterization axes, although may not have all the possible combinations
        # of values.
        for dataset_config in self.config.configurations:
            descriptor_axes = set(dataset_config.parameters.keys())
            if descriptor_axes.symmetric_difference(all_parameters):
                self.logger.error(
                    "Dataset parameterization for '%s' is incomplete: "
                    "found parameter axes %s, expected %s", dataset_config.name, descriptor_axes, all_parameters)
                raise RuntimeError("Configuration error")
            # Add one row to the table skeleton
            descriptor = Benchmark(self, dataset_config)
            for key, value in dataset_config.parameters.items():
                dataset_axes[key].append(value)
            instance_axis.append(dataset_config.g_uuid)
            descriptors.append(descriptor)

        dataset_axes.update({"instance": instance_axis, "descriptor": descriptors})
        # Actually create the table
        parameterization_table = pl.DataFrame(dataset_axes)
        self.logger.debug("Parameterization table:\n%s", parameterization_table)

        return parameterization_table

    def _ensure_dir_tree(self):
        """
        Build the session directory tree.
        """
        self.get_metadata_root_path().mkdir(exist_ok=True)
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
    def parameter_keys(self) -> list[str]:
        """
        The set of parameter keys that index the rows of the benchmark matrix.
        """
        return [col for col in self.parameterization_matrix.columns if col not in RESERVED_PARAMETER_NAMES]

    def get_public_tasks(self) -> list[Type[AnalysisTask]]:
        """
        Return the public tasks available for this session.

        :return: A list of :class:`AnalysisTask` classes.
        """
        namespaces = set()
        for bench_config in self.config.configurations:
            for exec_task_config in bench_config.generators:
                exec_task = TaskRegistry.resolve_exec_task(exec_task_config.handler)
                namespaces.add(exec_task.task_namespace)
        # Now group all public tasks from the namespaces
        tasks = []
        for ns in namespaces:
            tasks += TaskRegistry.public_tasks[ns].values()
        return [t for t in tasks if issubclass(t, AnalysisTask)]

    def delete(self):
        """
        Delete this session and all associated files.
        """
        self.logger.info("Remove session %s (%s)", self.name, self.uuid)
        shutil.rmtree(self.session_root_path)

    def bundle(self, path: Path | None = None, include_raw_data: bool = True) -> Path:
        """
        Produce a compressed archive with all the session output.
        """
        bundle_path = path if path else self.session_root_path.parent
        if bundle_path.exists() and bundle_path.is_dir():
            bundle_file = bundle_path / self.session_root_path.with_suffix(".tar.gz").name
        else:
            if bundle_path.name.endswith(".tar.gz"):
                bundle_file = bundle_path
            else:
                bundle_file = bundle_path.with_suffix(".tar.gz")
        self.logger.info("Generate %s bundle", self.session_root_path)
        if bundle_file.exists():
            self.logger.info("Replacing old bundle %s", bundle_file)
            bundle_file.unlink()
        if include_raw_data:
            archive_src = self.session_root_path.parent
        else:
            archive_src = self.get_plot_root_path()

        result = subprocess.run(["tar", "-z", "-c", "-C", archive_src, "-f", bundle_file, self.session_root_path.name])
        if result.returncode:
            self.logger.fatal("Failed to produce bundle")
            raise RuntimeError("Failed to bundle session")
        self.logger.info("Archive created at %s", bundle_file)
        return bundle_file

    def push(self, host: str, bundle_file: Path):
        """
        Push a bundled session to the given host.

        :param host: scp-like host address
        :bundle_file: source bundle to push
        """
        self.logger.info("Push %s to %s", bundle_file, host)
        result = subprocess.run(["scp", "-q", bundle_file, host])
        if result.returncode:
            self.logger.fatal("Failed to push session to %s", host)
            raise RuntimeError("Failed to push session")
        self.logger.info("Session bundle pushed")

    def get_instance_configuration(self, g_uuid: UUID | str) -> InstanceConfig:
        """
        Helper to retreive an instance configuration for the given g_uuid.
        """
        # Descriptors with the same instance share the same instance configuration
        # just grab the first one
        match_instance = self.parameterization_matrix.filter(instance=str(g_uuid))
        descriptor_config = match_instance["descriptor"][0].config
        return descriptor_config.instance

    def machine_configuration_name(self, g_uuid: UUID | str) -> str:
        """
        Retrieve the human-readable form of the machine configuration identified by a given UUID.
        This is useful for producing readable output.
        """
        instance_config = self.get_instance_configuration(g_uuid)
        return instance_config.name

    def get_metadata_root_path(self) -> Path:
        """
        :return: The metadata directory, used to store benchplot task metadata.
        """
        return self.session_root_path / ".metadata"

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

    def get_analysis_root_path(self) -> Path:
        """
        :return: The root path for generated files during analysis
        """
        return self.get_plot_root_path()

    def get_asset_root_path(self) -> Path:
        """
        :return: The root path for binary assets e.g. kernel images
        """
        return self.session_root_path / "assets"

    def all_benchmarks(self) -> list[Benchmark]:
        """
        Helper method to iterate over benchmark contexts.

        :return: A list containing all benchmark contexts from the
        benchmark matrix.
        """
        return self.parameterization_matrix["descriptor"]

    def clean_all(self):
        """
        Clean all output files, including benchmark data.
        """
        meta_root = self.get_metadata_root_path()
        data_root = self.get_data_root_path()
        plot_root = self.get_plot_root_path()
        asset_root = self.get_asset_root_path()
        if meta_root.exists():
            shutil.rmtree(meta_root)
        if data_root.exists():
            shutil.rmtree(data_root)
        if plot_root.exists():
            shutil.rmtree(plot_root)
        if asset_root.exists():
            shutil.rmtree(asset_root)
        self._ensure_dir_tree()

    def clean_analysis(self):
        """
        Clean all output analysis files.
        This retains the benchmark run data and assets.
        """
        plot_root = self.get_plot_root_path()
        if plot_root.exists():
            shutil.rmtree(plot_root)
        self._ensure_dir_tree()

    def generate(self, clean: bool = False):
        """
        Run the session execution generators.

        This will run the tasks configured as "generators" in the configuration to
        produce shell scripts in the session directories.
        """
        if clean:
            self.clean_all()

        # Generate top-level runner scripts to run benchmark groups on different systems
        # These are generated according to :attr:`PipelineBenchmarkConfig.system`
        # specification.
        data_root = self.get_data_root_path()
        for (target, ), section in self.parameterization_matrix.group_by("target"):
            section = section.select(
                pl.col("descriptor").map_elements(lambda bench: bench.get_run_script_path().relative_to(data_root),
                                                  return_dtype=pl.Object).alias("run_script"))
            ctx = TemplateContextBase(self.logger)
            ctx.set_template("run-target.sh.jinja")
            ctx.extend_context({
                "target": target,
                "run_scripts": section["run_script"],
                "bundle_results": self.config.bundle_results
            })
            run_target = re.sub(r"[\s/]", "-", target)
            script_path = self.get_data_root_path() / f"run-target-{run_target}.sh"
            with open(script_path, "w+") as script_file:
                ctx.render(script_file)
            script_path.chmod(0o755)

        # Schedule generators according to the collected descriptors
        for descriptor in self.parameterization_matrix["descriptor"]:
            descriptor.schedule_exec_tasks(self.scheduler, ExecTaskConfig())

        self.logger.info("Session %s generate execution plan", self.name)
        self.scheduler.run()
        self.logger.info("Session %s execution plan ready", self.name)
        self.scheduler.report_failures(self.logger)

    def run(self, driver_config, clean: bool = True):
        """
        Run the session on host machines.

        TODO NOT IMPLEMENTED
        Need to define DriverConfig.
        Implement Runner infrastructure

        This uses different executor strategies depending on the host where we
        are going to run.
        """
        raise NotImplementedError("TODO")

        # Cleanup previous run, to avoid any confusion
        if clean:
            self.clean_all()

        instance_manager = InstanceManager(self)
        self.scheduler.register_resource(instance_manager)

        # Resolve the session driver strategy
        # driver_class = TaskRegistry.resolve_task(driver_config.handler, kind=SessionDriver)

        # Package the session to transfer to the remote hosts
        # session_root = self.session_root_path

        # Schedule the session driver task
        # self.scheduler.add_task(driver)
        self.logger.info("Session %s begin execution", self.name)
        self.scheduler.run()
        self.logger.info("Session %s execution completed", self.name)
        self.scheduler.report_failures(self.logger)

    def analyse(self, analysis_config: AnalysisConfig | None, set_default: bool = False):
        """
        Run the session analysis tasks requested.
        The analysis pipeline is slighly different from the execution pipeline.
        This is because the same data can be used for different analysis targets and
        the user should be able to specify the set of targets we need to run.
        Here the analysis configuration should specify a set of public task names that
        we collect and schedule

        :param analysis_config: The analysis configuration
        """
        if set_default:
            self.logger.info("Update analysis configuration for session %s", self.name)
            self.config.analysis_config = analysis_config
            with open(self.session_root_path / SESSION_RUN_FILE, "w") as runfile:
                runfile.write(self.config.emit_json())
            if analysis_config is None:
                # We are removing the default config, nothing else to do
                return

        if analysis_config is None:
            # Load default analysis configuration from the session
            analysis_config = self.config.analysis_config

        for task_spec in analysis_config.tasks:
            resolved = TaskRegistry.resolve_task(task_spec.handler)
            if not resolved:
                self.logger.error("Invalid task name specification %s", task_spec.handler)
                raise ValueError("Invalid task name")
            for task_klass in resolved:
                if not issubclass(task_klass, AnalysisTask):
                    self.logger.warning("Analysis step only supports scheduling of AnalysisTasks, skipping %s",
                                        task_klass)
                if task_klass.task_config_class and isinstance(task_spec.task_options, dict):
                    options = task_klass.task_config_class.schema().load(task_spec.task_options)
                else:
                    options = task_spec.task_options

                # If the task is a session-wide task, just schedule it.
                # If the task is per-dataset, schedule one instance for each dataset.
                if task_klass.is_session_task():
                    task = task_klass(self, analysis_config, task_config=options)
                elif task_klass.is_dataset_task():
                    task = DatasetAnalysisTaskGroup(self, task_klass, analysis_config, task_config=options)
                self.logger.debug("Schedule analysis task %s with opts %s", task, options)
                self.scheduler.add_task(task)
        self.logger.info("Session %s start analysis", self.name)
        self.scheduler.run()
        self.logger.info("Session %s analysis finished", self.name)
        self.scheduler.report_failures(self.logger)

    def find_exec_task(self, task_class: Type[SessionExecutionTask]) -> SessionExecutionTask:
        """
        Find a session-wide execution task and return a task instance.

        This can be used by analysis tasks to load generator dependencies.
        """
        task = self.all_benchmarks()[0].find_exec_task(task_class)
        if not task.is_session_task() or not task.is_exec_task():
            self.logger.error("The task %s is not a SessionExecutionTask", task_class)
            raise TypeError("Invalid task type")
        return task

    def find_all_exec_tasks(self, task_class: Type[ExecutionTask]) -> list[ExecutionTask]:
        """
        Find all execution tasks of a given type and return them as a list.

        This can be used by analysis tasks to load generator dependencies.
        """
        tasks = []
        for bench in self.all_benchmarks():
            task = bench.find_exec_task(task_class)
            if not task.is_dataset_task() or not task.is_exec_task():
                self.logger.error("The task %s is not an ExecutionTask", task_class)
                raise TypeError("Invalid task type")
            tasks.append(task)
        return tasks
