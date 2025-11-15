import dataclasses as dc
import re
from collections import defaultdict
from io import StringIO
from textwrap import indent
from threading import Event
from typing import Callable, Generator, Hashable, Iterable, Self, Type

import polars as pl

from .borg import Borg
from .config import Config
from .error import MissingDependency, TaskNotFound
from .util import new_logger

type Session = "Session"
type Benchmark = "Benchmark"
type TaskFwd = "Task"
type TaskType = Type[TaskFwd]
type TaskTypeSequence = Generator[TaskType]
type Target = "Target"
type ScriptContext = "ScriptContext"
type ResourceRequest = "ResourceRequest"


@dc.dataclass
class TargetRef:
    """
    Internal helper for targets
    """

    #: The unique key for the target
    name: str
    #: The attribute name for the target in the Task subclass.
    attr: str


class TaskRegistry(type):
    """
    The task registry maintains a global list of tasks.
    This is used to resolve tasks from names in configuration files and CLI.

    :attribute public_tasks: Map task_namespace -> { task_name -> Task } for publicly name-able tasks
    :attribute all_tasks: Map task_namespace -> { task_name -> Task } for all tasks
    """

    public_tasks = defaultdict(dict)
    all_tasks = defaultdict(dict)

    def __new__(mcls, name: str, bases: tuple[Type], kdict: dict):
        """
        Add two registry attributes to each task class.
        The output_registry maintains a set of output generators for the class.
        The deps_registry maintins a set of dependency generators for the class.
        """
        kdict["_output_registry"] = {}
        kdict["_deps_registry"] = []
        return super().__new__(mcls, name, bases, kdict)

    def __init__(self, name: str, bases: tuple[Type], kdict: dict):
        super().__init__(name, bases, kdict)
        if self.task_name is None or self.task_namespace is None:
            # Skip, this is an abstract task
            return
        if self.task_config_class:
            assert dc.is_dataclass(self.task_config_class), (
                "Task configuration must be a dataclass"
            )
            assert issubclass(self.task_config_class, Config), (
                "Task configuration must inherit Config"
            )
        ns = TaskRegistry.all_tasks[self.task_namespace]
        if self.task_name in ns:
            raise ValueError(
                f"Multiple tasks with the same name: {self}, {ns[self.task_name]}"
            )
        ns[self.task_name] = self
        if self.public:
            ns = TaskRegistry.public_tasks[self.task_namespace]
            ns[self.task_name] = self
        for base in bases:
            if hasattr(base, "_deps_registry"):
                self._deps_registry.extend(base._deps_registry)
            if hasattr(base, "_output_registry"):
                self._output_registry.update(base._output_registry)

    def __str__(self):
        return f"<Task {self.__name__}: spec={self.task_namespace}.{self.task_name}>"

    def register_dependency(cls, name):
        cls._deps_registry.append(name)

    def register_output(cls, ref: TargetRef):
        cls._output_registry[ref.name] = ref.attr

    @classmethod
    def resolve_exec_task(cls, task_spec: str) -> TaskType | None:
        """
        Find the exec task named by the given task specifier.

        :param task_spec:
            Can be one of the following:

            1. The fully qualified name of the task (e.g. <task_namespace>.<task_name>)
            2. The namespace of an exec task (e.g. <task_spec>.exec exists)
        :return: The matching exec task or None
        """
        parts = task_spec.split(".")
        ns = cls.public_tasks[".".join(parts[:-1])]
        # If it is a full name, we are done
        task = ns.get(parts[-1])
        if task:
            return task
        # Do we have an exec task?
        exec_ns = TaskRegistry.public_tasks[".".join(parts)]
        task = exec_ns.get("exec")
        return task

    @classmethod
    def resolve_task(cls, task_spec: str) -> list[TaskType]:
        """
        Find the task named by the given task specifier.

        :param task_spec:
            Can be one of the following:

            1. The fully qualified name of the task (e.g. <task_namespace>.<task_name>)
            2. A wildcard task (e.g. <task_namespace>.*)
        :return: A list of matching tasks
        """
        parts = task_spec.split(".")
        ns = cls.public_tasks[".".join(parts[:-1])]
        if parts[-1] == "*":
            return list(ns.values())
        task = ns.get(parts[-1])
        if task:
            return [task]
        return []

    @classmethod
    def iter_public(cls) -> TaskTypeSequence:
        for ns, tasks in cls.public_tasks.items():
            for key, t in tasks.items():
                yield t


# Check that a task_namespace and task_name are valid
TASK_NAME_REGEX = re.compile(r"[\S.-]+")


class Task(Borg, metaclass=TaskRegistry):
    """
    Abstract base class for scheduled operations.

    This can be a task to drive data generation or perform an analysis step.
    Tasks in the pipeline have determined inputs and outputs that are derived
    from the session that creates the tasks.

    Tasks start as individual entities. When scheduled, if multiple tasks have
    the same ID, one task instance will be elected as the main task, the others
    will be attached to it as drones.
    The drones will share the task state with the main task instance to
    maintain consistency when producing stateful task outputs.
    This is essentially a Borg pattern where aliasing is determined dynamically
    using the task_id.

    Tasks are differentiated in two categories:
      1. Data generation tasks (also referred as "execution tasks")
      2. Data analysis tasks
    The former are scheduled during the session "run" phase, while the latter execute
    during the "analysis" phase.

    There are two further categories of tasks that are ortogonal to the distinction
    between data generation and analysis:
      1. Session-wide tasks exist as a single instance for each session.
        These are useful to collect data that is independent from execution contexts.
        An example may be scraping a source code repository to extract some information,
        this is independent from the variations of the built artefacts being run.
      2. Per-dataset context tasks (also referred as "benchmark tasks") are replicated for each
        data generation context. This essentially means that there is a per-dataset task
        for each parameterization in the session datagen matrix.
    """

    #: Mark the task as a top-level target, which can be named in configuration files and from CLI commands.
    public = False
    #: Human-readable task namespace, used for task identification
    task_namespace = None
    #: Human-readable task identifier, used for task identification
    task_name = None
    #: If set, use the given Config class to unpack the task configuration
    task_config_class: Type[Config] = None

    def __init__(self, task_config: Config = None):
        assert self.task_name is not None, (
            f"Attempted to use task with uninitialized name {self.__class__.__name__}"
        )
        assert self.task_namespace is None or TASK_NAME_REGEX.match(
            self.task_namespace
        ), f"Invalid task namespace '{self.task_namespace}'"
        assert TASK_NAME_REGEX.match(self.task_name), (
            f"Invalid task name '{self.task_name}'"
        )

        #: task-specific configuration options, if any
        self.config = task_config
        #: task logger
        self.logger = new_logger(f"{self.task_namespace}.{self.task_name}")
        #: notify when task is completed
        self._completed = Event()
        #: if the task run() fails, the scheduler will set this to the exception raised
        #: by the task.
        #: This field can not be read concurrently unless the :attr:`_completed` event
        #: is set, notifying that the task relinquishes write access to the failure data.
        self.failure = None
        #: set of tasks we resolved that we depend upon, this is filled by the scheduler.
        #: Read-only during the execution of :meth:`Task.execute()`.
        self.resolved_dependencies = set()
        #: This currently caches the results from output_map().
        #: Ideally, however, this should be filled either by the scheduler or
        #: from cached task metadata.
        #: The scheduler fill should occur after all dependencies have completed,
        #: but before run(), so it is possible to access dependencies in the
        #: outputs generator. Analysis tasks should be able to resolve exec
        #: tasks outputs from metadata. This would remove the necessity for
        #: instantiating tasks to reference outputs and removes limitations for
        #: dynamic output descriptor generation.
        self.collected_outputs = {}

        # Note: the whole __dict__ may be replaced by Borg, so no property
        # initialization beyond this point
        super().__init__()

    def __str__(self):
        return str(self.task_id)

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.task_id}>"

    def __hash__(self):
        return hash(self.task_id)

    def __eq__(self, other: "Task"):
        return self.task_id == other.task_id

    @classmethod
    def describe(cls):
        """
        Describe the task in and human-readable way.

        By default this takes the current task docstring content.
        """
        desc = StringIO()
        desc.write(f"# {cls.task_namespace}.{cls.task_name} ({cls.__name__}):\n")
        desc.write(cls.__doc__ + "\n")

        if cls.task_config_class:
            conf_class = cls.task_config_class
            desc.write(f"    Task configuration: {conf_class.__name__}\n")
            desc.write(conf_class.__doc__ + "\n")
            desc.write(indent(conf_class.describe(), " " * 4))
        return desc.getvalue()

    @classmethod
    def sample_config(cls) -> Config | None:
        """
        Generate a sample configuration for this task.

        This is intended to help setup benchmarks.
        """
        if cls.task_config_class:
            return cls.task_config_class()
        else:
            return None

    @property
    def task_id(self) -> Hashable:
        """
        Return the unique task identifier.
        """
        raise NotImplementedError("Subclass should override")

    @property
    def borg_state_id(self) -> Hashable:
        return self.task_id

    @property
    def completed(self) -> bool:
        """
        Check whether the task has completed. This only notifies whether
        the task is done, to check whether an error occurred, the :attr:`Task.failed`
        should be used.
        """
        return self._completed.is_set()

    @property
    def output_map(self) -> dict[str, Target]:
        """
        Return the output descriptors for the task.
        See note on :attr:`Task.collected_outputs`.
        The only invariant that should be enforced here is that the output map is only ever accessed
        after all dependencies tasks have completed.
        XXX this is not strictly true, it can be accessed after the dependencies generator
        has been consumed, so that dependencies are known and their side-effects are known.
        The actual data however will be ready only after the tasks are done.
        """
        if not self.collected_outputs:
            self.collected_outputs = dict(self.outputs())
        return self.collected_outputs

    def wait(self, timeout=None):
        """
        Wait for the task to complete.
        """
        self._completed.wait(timeout)

    def notify_done(self):
        """
        Notify that the task is done.
        This is used internally to set the completed event.
        """
        self._completed.set()

    def notify_failed(self, err: Exception):
        """
        Notify that the task is done but failed.
        """
        self.failure = err
        self._completed.set()

    def resources(self) -> Iterable[ResourceRequest]:
        """
        Produce a set of resources that are consumed by this task.
        Once the resources are available, they will be reserved and
        the resouce object will be availabe via the resource request get()
        method within Task.run().
        """
        yield from []

    def dependencies(self) -> Iterable[TaskFwd]:
        """
        Produce the set of :class:`Task` objects that this task depends upon.

        :return: sequence of dependencies
        """
        for attr in self._deps_registry:
            deps = getattr(self, attr)
            if not isinstance(deps, Iterable):
                deps = [deps]
            yield from filter(lambda d: d is not None, deps)

    def outputs(self) -> Iterable[tuple[str, Target]]:
        """
        Produce the set of :class:`Target` objects that describe the outputs
        that are produced by this task.
        Each target must be associated to a unique name.
        The name is considered to be the public interface for other tasks to
        fetch targets and outputs of this task.
        The name/target pairs may be used to construct a dictionary to access
        the return values by key.

        Note that this method generates output descriptors, not output data.
        The descriptors may be dynamic but must be determined independently of
        :meth:`Task.run`. This allows to use cached outputs in the future, and
        avoid calling :meth:`Task.run` altogether. The :meth:`Task.run` is
        responsible for producing the output data and setting it to the
        output_map, if needed.
        """
        registered = {
            key: getattr(self, attr) for key, attr in self._output_registry.items()
        }
        yield from registered.items()

    def execute(self) -> Self:
        """
        Main entry point for the executor thread pool.

        This ensures that dependencies are satisfied and acquires resources
        before running the task body :meth:`Task.run`.

        :return: The completed task itself.
        """
        try:
            self.logger.debug("Executing task %s", self)
            for dep in self.resolved_dependencies:
                assert dep.completed, f"Invalid execution, pending dependency {dep}"
                if dep.failure:
                    self.logger.debug("Cascade failure from dependency %s", dep)
                    raise RuntimeError("Cascade failure")
            self.run()
            self.notify_done()
            self.logger.debug("Done task %s", self)
        except Exception as err:
            self.notify_failed(err)
            raise err

    def run(self):
        pass


class dependency:
    """
    Decorator for :class:`Task` dependencies.
    Provides a shorthand to declare dependencies of a task.
    Note that this behaves as a property field.
    This should be used in place of the :meth:`Task.dependencies()` and
    :attr:`Task.resolved_dependencies` if there are no special requirements.

    Note that dependencies are Borgs, so there is a shared state for all tasks
    with the same ID, regardless of how many times the dependency property
    is accessed, therefore there is no need to cache here a single instance.
    """

    def __init__(self, fn: Callable | None = None, optional: bool = False):
        self._fn = fn
        self._optional = optional
        self._dependency_name = None

    def __set_name__(self, owner, name):
        """
        Register this descriptor as a dependency generator of this :class:`Task`
        """
        assert issubclass(owner, Task), (
            "@dependency decorator may be used only within Task classes"
        )
        owner.register_dependency(name)
        self._dependency_name = name

    def __get__(self, instance, owner=None):
        assert instance is not None
        assert self._fn is not None
        try:
            result = self._fn(instance)
            # Normalize any generator or iterable to a list, keep single value unchanged
            if isinstance(result, Iterable) and not isinstance(result, list):
                return list(result)
        except TaskNotFound:
            result = None

        if not self._optional and not result:
            instance.logger.error(
                "Missing required dependency %s", self._dependency_name
            )
            raise MissingDependency("Failed to resolve dependency")
        return result

    def __call__(self, fn):
        """
        Invoked when the decorator is called with arguments.
        Just return the descriptor with the correct arguments.
        """
        return dependency(fn=fn, optional=self._optional)


class output:
    """
    Decorator for :class:`Task` output artefacts.
    Provides a shorthand to declare outputs of a task.
    Note that this behaves as a property field.
    This should be used in place of the :meth:`Task.outputs()` and
    :attr:`Task.output_map` if there are no special requirements.

    This decorator acts as a property decorator.
    There is only one instance for each task output target, which is held here,
    further references to it may be obtained from the :class:`Task` class for
    convenience.
    """

    def __init__(self, fn: Callable | None = None, name: str | None = None):
        """
        :param name: Override the name of the output, the name of the decorated
        function is used otherwise.
        """
        self._fn = fn
        self._key = name

    def __set_name__(self, owner, name):
        """
        Register this descriptor as an output of this :class:`Task`
        """
        assert issubclass(owner, Task), (
            "@output decorator may be used only within Task classes"
        )
        assert self._fn is not None
        if self._key is None:
            self._key = name
        owner.register_output(TargetRef(self._key, name))

    def __get__(self, instance, owner=None):
        assert instance is not None
        assert self._fn is not None
        result = self._fn(instance)

        from .artefact import Target

        # Normalize generators and iterables to a list
        if (
            not isinstance(result, Target)
            and isinstance(result, Iterable)
            and not isinstance(result, list)
        ):
            result = list(result)
        return result

    def __call__(self, fn):
        """
        Invoked when the decorator is called with arguments.
        Just return the descriptor with the correct arguments.
        """
        return output(fn=fn, name=self._key)


class SessionTask(Task):
    """
    Base class for all tasks that are unique within a session.

    These tasks do not reference a specific benchmark ID or benchmark group ID.
    """

    def __init__(self, session: Session, task_config: Config = None):
        self._session = session

        # Borg initialization occurs here
        super().__init__(task_config=task_config)

        #: Task logger is a child of the session logger
        self.logger = new_logger(f"{self.task_name}", parent=session.logger)

    @property
    def session(self):
        return self._session

    @property
    def task_id(self):
        return f"{self.task_namespace}.{self.task_name}-{self.session.uuid}"


class DatasetTask(Task):
    """
    Base class for all tasks that are unique for each parameterised dataset.

    These tasks reference a specific benchmark parameterisation.
    """

    def __init__(self, desc: Benchmark, task_config: Config = None):
        #: Associated benchmark descriptor
        self.benchmark = desc

        # Borg initialization occurs here
        super().__init__(task_config=task_config)

        #: Task logger is a child of the benchmark logger
        self.logger = new_logger(f"{self.task_name}", parent=self.benchmark.logger)

    @property
    def session(self):
        return self.benchmark.session

    @property
    def task_id(self) -> str:
        return f"{self.task_namespace}.{self.task_name}-{self.benchmark.uuid}"

    def add_dataset_metadata(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add metadata columns for the dataset to a given dataframe.

        This will create the following columns:
          - dataset_id set to the UUID dataset identifier
          - dataset_gid set to the UUID platform identifier
          - a variable number of parameter key/value entries from the datagen
            parameterization.
        """
        df = df.with_columns(
            pl.lit(self.benchmark.uuid).alias("dataset_id"),
            pl.lit(self.benchmark.g_uuid).alias("dataset_gid"),
            *(pl.lit(v).alias(k) for k, v in self.benchmark.parameters.items()),
        )
        return df


class ExecutionTask(DatasetTask):
    """
    Execution tasks are per-dataset tasks that are scheduled as part of the generation phase.

    These tasks define how the run scripts are generated for each benchmark parameterisation
    and are scheduled as dependencies of the root benchmark execution generator task
    (see :class:`BenchmarkExecTask`).

    The :meth:`ExecutionTask.run()` method is generally used for the following:
    1. Override the default runner script template.
    2. Extend the script context with additional configuration options.
    3. Add hooks to different benchmark run phases.

    Invariant: tasks with the same name are not issued more than once for each benchmark
    descriptor UUID, this is a requirement for the Borg task_id.
    """

    task_name = "exec"

    def __init__(
        self, benchmark: Benchmark, script: ScriptContext, task_config: Config = None
    ):
        super().__init__(benchmark, task_config=task_config)
        #: Script builder associated to the current benchmark context.
        self.script = script

    @property
    def uuid(self):
        return self.benchmark.uuid
