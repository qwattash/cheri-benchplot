import dataclasses as dc
import multiprocessing as mp
import re
import typing
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from queue import Queue
from threading import Condition, Event, Lock, Semaphore, Thread

import networkx as nx

from .config import Config
from .shellgen import ShellScriptBuilder
from .util import new_logger


class WorkerShutdown(Exception):
    """
    Special exception that triggers the worker thread loop shutdown.
    """
    pass


class TaskRegistry(type):
    """
    The task registry maintains a global list of tasks.
    This is used to resolve tasks from names in configuration files and CLI.

    :attribute public_tasks: Map task_namespace -> { task_name -> Task } for publicly name-able tasks
    :attribute all_tasks: Map task_namespace -> { task_name -> Task } for all tasks
    """
    public_tasks = defaultdict(dict)
    all_tasks = defaultdict(dict)

    def __init__(self, name: str, bases: typing.Tuple[typing.Type], kdict: dict):
        super().__init__(name, bases, kdict)
        if self.task_name is None:
            # Skip, this is an abstract task
            return
        ns = TaskRegistry.all_tasks[self.task_namespace]
        if self.task_name in ns:
            raise ValueError(f"Multiple tasks with the same name: {self}, {ns[self.task_name]}")
        ns[self.task_name] = self
        if self.public:
            ns = TaskRegistry.public_tasks[self.task_namespace]
            ns[self.task_name] = self


class Target:
    """
    Helper to represent output artifacts of a task.
    """
    def is_file(self):
        """
        When true, the target should expose an host absolute file path in the :attr:`Target.path` property
        """
        return False

    def needs_extraction(self):
        return False

    def target_id(self):
        return None


class DataFileTarget(Target):
    """
    A target output file that is generated on the guest and needs to be extracted.
    """
    @classmethod
    def from_task(cls, task: "Task", iteration: int = None, extension: str = None):
        """
        Create a task path using the task identifier and the parent session data root path
        """
        name = task.task_id
        if iteration:
            base_path = task.benchmark.get_benchmark_iter_data_path(iteration)
        else:
            base_path = task.benchmark.get_benchmark_data_path()
        path = base_path / name
        if extension:
            path = path.with_suffix(extension)
        return cls(task.benchmark, path)

    def __init__(self, benchmark, path):
        self.benchmark = benchmark
        self.path = path

    def to_remote_path(self):
        """
        Convert a path in the benchmark local data directory to a remote file path
        """
        base_path = self.benchmark.get_benchmark_data_path()
        base_guest_output = self.benchmark.config.remote_output_dir
        assert self.path.is_absolute(), f"Ensure target path is absolute {self.path}"
        assert str(self.path).startswith(
            str(base_path)), f"Ensure target path {self.path} is in benchmark {self.benchmark} output"
        return base_guest_output / self.path.relative_to(base_path)

    def is_file(self):
        return True

    def needs_extraction(self):
        return True

    def target_id(self):
        return self.path


class Task(metaclass=TaskRegistry):
    """
    Base class for dataset operations.
    This can be a task to run a benchmark or perform an analysis step.
    Tasks in the pipeline have determined inputs and outputs that are derived from the session that
    creates the tasks.
    """
    #: Mark the task as a top-level target
    public = False
    #: Human-readable task namespace, used for task identification
    task_namespace = "internal"
    #: Human-readable task identifier, used for task identification
    task_name = None
    #: If set, use the given Config class to unpack the task configuration
    task_config_class: typing.Type[Config] = None

    def __init__(self, benchmark: "Benchmark", task_config: Config = None):
        #: parent benchmark
        self.benchmark = benchmark
        #: task-specific configuration options, if any
        self.config = task_config
        #: task logger
        self.logger = new_logger(f"{self.task_name}", parent=self.benchmark.logger)
        #: notify when task is completed
        self._completed = Event()
        #: if the task run() fails, the scheduler will set this to the exception raised by the task
        self.failed = None
        #: set of tasks we resolved that we depend upon, this is filled by the scheduler
        self.resolved_dependencies = set()

        assert self.task_name is not None, "Attempted to use task with uninitialized name"

    def __str__(self):
        return f"{self.task_namespace}.{self.task_name}-{self.benchmark.uuid}"

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.task_id}>"

    def __hash__(self):
        return hash(self.task_id)

    def __eq__(self, other: "Task"):
        return self.task_id == other.task_id

    @property
    def session(self):
        return self.benchmark.session

    @property
    def uuid(self):
        return self.benchmark.uuid

    @property
    def g_uuid(self):
        return self.benchmark.g_uuid

    @property
    def task_id(self) -> typing.Hashable:
        """
        Return the unique task identifier.
        Note that this currently assumes that tasks with the same name are not issued
        more than once for each benchmark run UUID. If this is violated, we need to
        change the task ID generation.
        """
        return f"{self.task_namespace}.{self.task_name}-{self.benchmark.uuid}"

    @property
    def completed(self) -> bool:
        return self._completed.is_set()

    def wait(self):
        """
        Wait for the task to complete.
        """
        self._completed.wait()

    def notify_done(self):
        """
        Notify that the task is done.
        This is used internally to set the completed event.
        """
        self._completed.set()

    def notify_failed(self, ex: Exception):
        """
        Notify that the task is done but failed.
        """
        self.failed = ex
        self._completed.set()

    def resources(self) -> typing.Generator[str, None, None]:
        """
        Produce a set of resources that are consumed by this task.
        Once the resources are available, they will be reserved and
        the resouce object will be availabe via the resource request get()
        method within Task.run().
        """
        yield from []

    def dependencies(self) -> typing.Generator["Task", None, None]:
        """
        Produce the set of :class:`Task` objects that this task depends upon.

        :return: sequence of dependencies
        """
        yield from []

    def outputs(self) -> typing.Generator[Target, None, None]:
        """
        Produce the set of :class:`Target` objects that describe the outputs that are produced
        by this task.
        """
        yield from []

    def run(self):
        raise NotImplementedError("Task.run() must be overridden")


class ExecutionTask(Task):
    """
    Base class for tasks that are scheduled as dependencies of the main internal benchmark
    run task.
    These are intended to generate the benchmark run script and perform any instance or benchmark
    configuration before the actual benchmark instance runs.
    Note that these tasks are dynamically resolved by the benchmark context and attached as dependencies
    of the top-level execution tasks :class:`BenchmarkExecTask`.

    The run() method for ExecutionTasks is a good place to perform any of the following:
    1. Extract any static information from benchmark binary files.
    2. Configure the benchmark instance via platform_options.
    3. Add commands to the benchmark run script sections.
    """
    task_name = "exec"

    def __init__(self, benchmark: "Benchmark", script: ShellScriptBuilder, task_config: Config = None):
        super().__init__(benchmark, task_config=task_config)
        assert self.task_name == "exec", "ExecutionTask convention mandates the 'exec' task name"
        self.script = script


class DatasetIngestionTask(Task):
    """
    Base class for tasks that perform data ingestion.
    These are generally reusable and perform the load/pre_merge/merge/post_merge steps.
    """
    task_name = "load"

    def run(self):
        pass


class DatasetAnalysisTask(Task):
    """
    Analysis tasks that perform anythin from plotting to data checks and transformations.
    These generally are the top-level tasks that correspond to run targets exposed via CLI.
    """
    pass


class ResourceManager:
    """
    Base class to abstract resource limits on tasks.
    Resouce managers must be registered with the task scheduler before running or adding
    any tasks to the scheduler.
    Resource managers are identified by the resource name, so concrete managers must set
    a resource_name class property and collisions are not allowed.
    """
    resource_name: str = None

    @dc.dataclass
    class ResourceRequest:
        """
        Private object representing a request for this resource.
        This is sortable and comparable.
        """
        name: str
        pool: typing.Hashable | None
        acquire_args: dict[str, typing.Any]
        _resource: typing.Any = dc.field(init=False, default=None)

        def __lt__(self, other):
            return self.name < other.name

        def __eq__(self, other):
            return self.name == other.name and self.pool == other.pool and self.acquire_args == other.acquire_args

        def set_resource(self, r):
            """
            Internal interface to notify that a resource was given in response to the request
            """
            self._resource = r

        def clear_resource(self):
            """
            Internal interface to notify that a resource is no longer assigned to this request
            """
            self._resource = None

        def get(self):
            if self._resource is None:
                raise RuntimeError(f"Access resource {self.name} but it is not ready")
            return self._resource

    @classmethod
    def request(cls, pool: typing.Hashable = None, **kwargs) -> ResourceRequest:
        assert cls.resource_name is not None, f"{cls} is missing resource name?"
        return cls.ResourceRequest(cls.resource_name, pool, kwargs)

    def __init__(self, session: "PipelineSession", limit: int | None):
        assert self.resource_name is not None, f"{self.__class__} is missing resource name?"
        self.session = session
        self.logger = new_logger(f"rman-{self.resource_name}")
        self.limit = limit
        if limit is not None:
            self._limit_guard = Semaphore(self.limit)

    def __str__(self):
        return f"{self.__class__.__name__}[{self.resource_name}]"

    def _acquire(self, pool: typing.Hashable):
        """
        Reserve an item slot in the underlying resource pool
        """
        if self.is_unlimited:
            return
        self._limit_guard.acquire()

    def _release(self, pool: typing.Hashable):
        """
        Release an item slot to the underlying resource pool
        """
        if self.is_unlimited:
            return
        self._limit_guard.release()

    def _get_resource(self, req: ResourceRequest) -> typing.Any:
        """
        Produce a resource item after a slot is acquired.
        """
        raise NotImplementedError("Must override")

    def _put_resource(self, r: typing.Any, req: ResourceRequest):
        """
        Return a resource item to the manager after the client is done with it
        """
        raise NotImplementedError("Must override")

    @property
    def is_unlimited(self) -> bool:
        return self.limit is None

    @contextmanager
    def acquire(self, req: ResourceRequest) -> typing.ContextManager[typing.Any]:
        """
        Acquire a resource, optionally specifying an internal pool that
        may be used to improve allocation.
        Resource managers may also require extra arguments to perform the
        resource allocation, there are passed via the ResourceRequest helper
        object.
        """
        self.logger.debug("Waiting for %s", self.resource_name)
        self._acquire(req.pool)
        self.logger.debug("Acquired %s", self.resource_name)
        # Produce the resource
        try:
            r = self._get_resource(req)
            req.set_resource(r)
        except:
            self.logger.error("Failed to produce %s from pool %s", self.resource_name, req.pool)
            raise
        finally:
            self._release(req.pool)
        # Wait for somebody to do something with it
        try:
            yield r
        finally:
            req.clear_resource()
            try:
                self._put_resource(r, req)
            except:
                self.logger.error("Failed to return %s to pool %s", self.resource_name, req.pool)
                raise
            finally:
                self._release(req.pool)
        self.logger.debug("Released %s", self.resource_name)

    def sched_shutdown(self):
        """
        The scheduler is shutting down and all activity should
        be moped up. If we need to wait for something to complete cleanup we can
        block here.
        """
        pass


class TaskScheduler:
    """
    Schedule running tasks into workers, handling task dependencies.
    """
    def __init__(self, session: "PipelineSession"):
        """
        :param session: The parent session
        """
        #: parent session
        self.session = session
        #: task graph
        self._task_graph = nx.DiGraph()
        #: scheduler logger
        self.logger = new_logger("task-scheduler", self.session.logger)
        #: task runner threads
        self._worker_threads = []
        #: worker wakeup lock, used by the worker wakeup condition
        self._worker_lock = Lock()
        #: worker wakeup event
        self._worker_wakeup = Condition(self._worker_lock)
        #: worker shutdown request
        self._worker_shutdown = False
        #: task queue, this is not a synchronized queue as it shares the lock with _worker_wakeup.
        self._task_queue = []
        #: active tasks set, this contains the set of tasks the workers are currently working on.
        #: Protected by the _worker_wakeup lock.
        self._active_tasks = []
        # XXX should this share the lock with worker_wakeup?
        #: pending tasks count, protected by _pending_tasks_cv
        self._pending_tasks = 0
        #: task completion barrier
        self._pending_tasks_cv = Condition()
        #: number of worker threads to run
        self.num_workers = session.config.concurrent_workers or (mp.cpu_count() * 5)
        #: resource managers
        self._rman = {}

    def _handle_failure(self, failed_task: Task):
        """
        Handle a task failure.
        Depending on the configuration policy, we either cancel every task or only cancel the subgraph
        linked to the failed task.

        :param failed_task: The task that failed, note that this may already have been cleaned up
        by a previous call to _handle_failure()
        """
        if self.session.config.abort_on_failure:
            # Kill all tasks and signal workers stop
            self.logger.warning("Cancelling all pending tasks")
            with self._worker_wakeup:
                self._task_queue.clear()
                # XXX Find a nice way to propagate the early-stop request to current tasks.
                # for task in self._active_tasks:
                #     task.cancel()
                self._worker_shutdown = True
                self._worker_wakeup.notify_all()
            # Should this share the lock with worker_wakeup?
            with self._pending_tasks_cv:
                self._pending_tasks = 0
                self._pending_tasks_cv.notify_all()
        else:
            # Find out which tasks we have to kill and unschedule them
            pass

    def _check_worker_shutdown(self):
        """
        Check the worker shutdown signal condition.
        If the flag is set, we abort the execution of the worker loop via an exception.
        """
        with self._worker_lock:
            if self._worker_shutdown:
                raise WorkerShutdown()

    def _worker_loop(self, worker_index: int):
        """
        Main worker loop. This pulls tasks from the task_queue and handles them when their dependencies
        have completed. If the schedule is well-formed, we are guaranteed not to deadlock.

        :param worker_index: The sequential index of the worker in the worker list.
        """
        self.logger.debug("Start worker[%d]", worker_index)
        while True:
            self.logger.debug("worker[%d] waiting for work...", worker_index)
            with self._worker_wakeup:
                self._worker_wakeup.wait_for(lambda: self._worker_shutdown or len(self._task_queue) > 0)
                # Check shutdown signal
                if self._worker_shutdown:
                    break
                task = self._task_queue.pop(0)
            self.logger.debug("worker[%d] received task: %s", worker_index, task)
            try:
                # Wait for all dependencies to be done
                for dep in self._task_graph.successors(task):
                    dep.wait()
                    assert dep.completed, f"Dependency wait() returned but task is not completed {dep}"
                    # If a dependency failed, we need to do some scheduling changes, the exception triggers it
                    if dep.failed:
                        self.logger.error("Dependency %s failed, bail out from %s", dep, task)
                        raise RuntimeError("Task dependency failed")
                    # Before continuing, we should check whether somebody notified worker shutdown
                    with self._worker_wakeup:
                        if self._worker_shutdown:
                            raise WorkerShutdown()
                # Grab all resources
                with ExitStack() as resource_stack:
                    resources = {}
                    for req in sorted(task.resources()):
                        resources[req.name] = resource_stack.enter_context(self._rman[req.name].acquire(req))
                    # Now we can run the task
                    task.run()
                # After cleanup of resources we notify done and unlock the others
                task.notify_done()
            except WorkerShutdown:
                self.logger.debug("Caught worker shutdown signal, exit worker loop")
                break
            except Exception as ex:
                self.logger.exception("Error in worker[%d]", worker_index)
                # Notify other workers that may be waiting on this dependency that they should bail out.
                task.notify_failed(ex)
                # Now we need to cancel some tasks
                self._handle_failure(task)
            with self._pending_tasks_cv:
                self._pending_tasks -= 1
                self._pending_tasks_cv.notify_all()
        self.logger.debug("Shutdown worker[%d]", worker_index)

    def add_task(self, task: Task):
        self.logger.debug("Schedule task %s", task.task_id)
        if task in self._task_graph:
            # Assume that we can not have tasks with duplicate IDs
            # If a task is already scheduled just skip this, duplicate
            # dependencies are allowed.
            return
        self._task_graph.add_node(task)
        for dep in task.dependencies():
            task.resolved_dependencies.add(dep)
            self.add_task(dep)
            self._task_graph.add_edge(task, dep)

    def resolve_schedule(self):
        """
        Produce a schedule for the tasks. Note that we use lexicographic sort by g_uuid in
        order to help the instance manager maximise reuse of instances when this is enabled.
        """
        try:
            if self.session.config.reuse_instances:
                sched = nx.lexicographical_topological_sort(self._task_graph, key=lambda t: t.g_uuid)
            else:
                sched = nx.topological_sort(self._task_graph)
            run_sched = list(reversed(list(sched)))
            self.logger.debug("Resolved benchmark schedule %s", run_sched)
            return run_sched
        except nx.NetworkXUnfeasible:
            for cycle in nx.simple_cycles(self._task_graph):
                cycle_str = [str(c) for c in cycle]
                self.logger.error("Impossible task schedule: cyclic dependency %s", cycle_str)
            raise RuntimeError("Impossible to create a task schedule")

    def register_resource(self, rman: ResourceManager):
        if rman.resource_name in self._rman:
            self.logger.error("Duplicate resource manager registration %s: given %s found %s", rman.resource_name, rman,
                              self._rman[rman.resource_name])
            raise ValueError("Resource manager with the same name is already registered")
        self._rman[rman.resource_name] = rman

    def run(self):
        """
        Spawn worker threads and run the tasks in the schedule.
        Once all tasks are done, the workers are cleaned up.
        """
        self._worker_shutdown = False
        schedule = self.resolve_schedule()
        try:
            # No need to synchronize this as the workers have not started yet
            self._task_queue.extend(schedule)
            self._pending_tasks = len(self._task_queue)
            self.logger.debug("Queued tasks: pending=%d", self._pending_tasks)
            # only start working if we actually have work
            if self._pending_tasks == 0:
                return
            for i in range(self.num_workers):
                worker = Thread(target=self._worker_loop, args=[i])
                self._worker_threads.append(worker)
                worker.start()
            # notify that we have work
            with self._worker_wakeup:
                self._worker_wakeup.notify_all()
            # wait for all tasks to complete
            with self._pending_tasks_cv:
                self._pending_tasks_cv.wait_for(lambda: self._pending_tasks == 0)
        finally:
            # shutdown all workers
            self.logger.debug("Shutting down workers")
            with self._worker_wakeup:
                self._worker_shutdown = True
                self._worker_wakeup.notify_all()
            for i, worker in enumerate(self._worker_threads):
                self.logger.debug("Wait for worker[%d] shutdown", i)
                worker.join()
            self._worker_threads.clear()
            for name, rman in self._rman.items():
                self.logger.debug("Shutting down resource manager %s", name)
                rman.sched_shutdown()
            self._rman.clear()
