from dataclasses import dataclass
from pathlib import Path
from typing import Type

from .artefact import RemoteTarget
from .config import CommandHookConfig, Config, ExecTargetConfig
from .elf import AddressSpaceManager
from .error import TaskNotFound
from .scheduler import TaskScheduler
from .shellgen import ScriptContext, ScriptHook
from .task import ExecutionTask, Task, TaskRegistry
from .util import new_logger


type Session = "Session"
type BenchmarkRunConfig = "BenchmarkRunConfig"


@dataclass
class ExecTaskConfig(Config):
    """
    Configuration object for :class:`BenchmarkExecTask`
    """

    pass


class BenchmarkExecTask(Task):
    """
    Task that dynamically gathers dependencies for all the execution task required
    by a benchmark run instance.
    """

    task_namespace = "internal.exec"
    task_name = "benchmark-root"

    def __init__(self, benchmark, task_config: ExecTaskConfig):
        #: Associated benchmark context
        self.benchmark = benchmark
        #: Script template context for this benchmark context
        self.script = ScriptContext(benchmark)

        # Borg initialization occurs here
        super().__init__(task_config=task_config)

    def _fetch_task(self, config: ExecTargetConfig) -> Task:
        task_klass = TaskRegistry.resolve_exec_task(config.handler)
        if task_klass is None:
            self.logger.error(
                "Invalid task name specification exec.%s", task_klass.task_name
            )
            raise ValueError("Invalid task name")
        if issubclass(task_klass, ExecutionTask):
            return task_klass(
                self.benchmark, self.script, task_config=config.task_options
            )
        else:
            self.logger.error("Invalid task %s, must be ExecutionTask", task_klass)
            raise TypeError("Invalid task type")

    def _handle_command_hook(self, phase: str, hook_config: CommandHookConfig):
        """
        Verify whether the given command hook configuration is enabled for the
        current parameterization.
        If it is enabled, issue all commands to the corresponding script context phase.
        """
        for match_param, match_value in hook_config.matches.items():
            if match_param not in self.benchmark.config.parameters:
                self.logger.error("Invalid parameter '%s' in command hook", match_param)
                raise RuntimeError("Invalid configuration")
            if self.benchmark.config.parameters[match_param] != match_value:
                return

        match_str = " && ".join([f"{k}={v}" for k, v in hook_config.matches.items()])
        hook = ScriptHook(
            name=f"User hook when {match_str}", commands=hook_config.commands
        )
        self.script.add_hook(phase, hook)

    def _handle_config_command_hooks(self):
        """
        Add the extra commands from :attr:`BenchmarkRunConfig.command_hooks`
        See the :class:`BenchmarkRunConfig` class for a description of the format we expect.
        """
        for phase, hooks in self.benchmark.config.command_hooks.items():
            for hook_config in hooks:
                self._handle_command_hook(phase, hook_config)

    def _extract_files(self, instance, task):
        """
        Extract remote files for a given task.
        """
        for output_key, output in task.outputs():
            if not isinstance(output, RemoteTarget):
                continue
            for remote_path, host_path in zip(output.remote_paths(), output.paths()):
                self.logger.debug(
                    "Extract %s guest: %s => host: %s",
                    output_key,
                    remote_path,
                    host_path,
                )
                instance.extract_file(remote_path, host_path)

    @property
    def session(self):
        return self.benchmark.session

    @property
    def task_id(self):
        return f"{self.task_namespace}-{self.task_name}-{self.benchmark.uuid}"

    def resources(self):
        """
        Request resources for this task prior to running.
        """
        yield from ()

    def dependencies(self):
        # Note that dependencies are guaranteed to be scanned on schedule resolution,
        # so before resource requests are resolved. We can determine here how many deps
        # require an instance
        for exec_target in self.benchmark.config.generators:
            yield self._fetch_task(exec_target)

    def run(self):
        """
        Generate the per-benchmark execution scripts.
        """
        self._handle_config_command_hooks()
        script_path = self.benchmark.get_run_script_path()
        self.script.register_global(
            "workload_done_file", self.session.workload_done_file
        )
        with open(script_path, "w+") as script_file:
            self.script.render(script_file)
        script_path.chmod(0o755)


class Benchmark:
    """
    Benchmark context

    :param session: The parent session
    :param config: The benchmark run configuration allocated to this handler.
    """

    def __init__(self, session: Session, config: BenchmarkRunConfig):
        self.session = session
        self.config = config
        self.logger = new_logger(f"{self.config.name}:{self.config.instance.name}")

        # Symbol mapping handler for this benchmark instance
        self.sym_resolver = AddressSpaceManager(self)

    def __str__(self):
        return f"{self.config.name}({self.uuid})"

    def __repr__(self):
        return str(self)

    @property
    def uuid(self):
        return self.config.uuid

    @property
    def g_uuid(self):
        return self.config.g_uuid

    @property
    def parameters(self) -> dict[str, any]:
        """
        Helper to inspect the user-definied parameterization for this benchmark instance.
        This is the index in the datagen matrix corresponding to this benchmark's row.
        """
        return self.config.parameters

    @property
    def metadata_columns(self) -> list[str]:
        """
        Helper to find the metadata columns for the current benchmark.
        The metadata columns are the standard dataset_id, dataset_gid and iteration, plus
        the parameterzation keys.
        """
        return ["dataset_id", "dataset_gid", "iteration"] + list(self.parameters.keys())

    @property
    def user_config(self):
        return self.session.user_config

    @property
    def analysis_config(self):
        return self.session.analysis_config

    @property
    def cheribsd_rootfs_path(self):
        rootfs_path = (
            self.user_config.rootfs_path / f"rootfs-{self.config.instance.cheri_target}"
        )
        if not rootfs_path.exists() or not rootfs_path.is_dir():
            raise ValueError(
                f"Invalid rootfs path {rootfs_path} for benchmark instance"
            )
        return rootfs_path

    def _ensure_dir_tree(self):
        """
        Ensure that per-dataset directories are ready
        """
        pass

    def get_run_script_path(self) -> Path:
        """
        :return: The path to the run script to import to the guest for this benchmark
        """
        return self.get_benchmark_data_path() / self.session.workload_run_script

    def get_benchmark_data_path(self) -> Path:
        """
        :return: The output directory for run data corresponding to this benchmark configuration.
        """
        return self.session.get_data_root_path() / f"{self.config.name}-{self.uuid}"

    def get_benchmark_asset_path(self) -> Path:
        """
        :return: Path for archiving binaries used in the run
        """
        return self.session.get_asset_root_path() / f"{self.config.name}-{self.uuid}"

    def get_instance_asset_path(self) -> Path:
        """
        :return: Path for archiving binaries used in the run but belonging to the instance.
            This avoids duplication.
        """
        return (
            self.session.get_asset_root_path()
            / f"{self.config.instance.name}-{self.g_uuid}"
        )

    def get_plot_path(self) -> Path:
        """
        Get base plot path for the current benchmark instance.
        Every plot should use this path as the base path to generate plots.
        """
        return self.session.get_plot_root_path() / self.config.name

    def get_analysis_path(self) -> Path:
        """
        :return: Base path for generated analysis files related to this benchmark.
        """
        return self.get_plot_path()

    def get_benchmark_iter_data_path(self, iteration: int) -> Path:
        """
        :return: The output directory for run data corresponding to this benchmark configuration and
            a specific benchmark iteration.
        """
        iter_path = self.get_benchmark_data_path() / str(iteration)
        return iter_path

    def build_exec_task(self, exec_config: ExecTaskConfig) -> BenchmarkExecTask:
        """
        Create an instance of the top-level benchmark execution task for this benchmark.

        The task is not scheduled and may be used to obtain dependency and output
        information for loading data from generators during analysis.

        :param exec_config: The execution task configuration.
        :return: The top-level execution task
        """
        exec_task = BenchmarkExecTask(self, task_config=exec_config)
        return exec_task

    def find_exec_task(
        self, task_class: Type[ExecutionTask], include_subclass=False
    ) -> ExecutionTask:
        """
        Find the given execution task configured for the current benchmark context.

        This is used during analysis to resolve generators as dependencies.
        :param task_class: The type of the task that needs to be resolved.
        :return: An instance of the given task type, bound to the current benchmark context.
        """
        main_task = self.build_exec_task(ExecTaskConfig())
        deps = list(main_task.dependencies())
        for task in deps:
            if (
                task_class.task_namespace == task.task_namespace
                and task_class.task_name == task.task_name
            ):
                if not isinstance(task, task_class):
                    self.logger.error(
                        "Found generator for %s but it is not an instance of %s",
                        task.task_id,
                        task_class,
                    )
                    raise RuntimeError("Invalid exec task instance")
                return task
            if include_subclass and isinstance(task, task_class):
                return task
        # XXX Recursively look into dependencies
        raise TaskNotFound(
            "Task %s can not be found among the session generators", task_class
        )

    def schedule_exec_tasks(
        self, scheduler: TaskScheduler, exec_config: ExecTaskConfig
    ):
        """
        Generate the top-level benchmark execution task for this benchmark run and schedule it.

        :param scheduler: The scheduler for this session
        :param exec_config: The execution task configuration. This is used to control partial
            execution runs and behaviour outside of the session configuration file.
        """
        # Ensure that all the data paths are initialized
        self.get_benchmark_data_path().mkdir(exist_ok=True)
        for i in range(self.config.iterations):
            self.get_benchmark_iter_data_path(i).mkdir(exist_ok=True)

        exec_task = self.build_exec_task(exec_config)
        self.logger.debug("Initialize top-level benchmark task %s", exec_task)
        scheduler.add_task(exec_task)
