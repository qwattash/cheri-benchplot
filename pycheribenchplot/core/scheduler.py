import concurrent.futures as cf
import dataclasses as dc
import multiprocessing as mp
from collections import deque
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import contextmanager
from threading import Semaphore
from typing import Any, ContextManager, Hashable

import networkx as nx

from .task import DatasetTask, Task
from .util import new_logger

type Session = "Session"


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
        pool: Hashable | None
        acquire_args: dict[str, Any]
        _resource: Any = dc.field(init=False, default=None)

        def __lt__(self, other):
            return self.name < other.name

        def __eq__(self, other):
            return (
                self.name == other.name
                and self.pool == other.pool
                and self.acquire_args == other.acquire_args
            )

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
    def request(cls, pool: Hashable = None, **kwargs) -> ResourceRequest:
        assert cls.resource_name is not None, f"{cls} is missing resource name?"
        return cls.ResourceRequest(cls.resource_name, pool, kwargs)

    def __init__(self, session: Session, limit: int | None):
        assert self.resource_name is not None, (
            f"{self.__class__} is missing resource name?"
        )
        self.session = session
        self.logger = new_logger(f"rman-{self.resource_name}")
        self.limit = limit
        if not self.is_unlimited:
            self._limit_guard = Semaphore(self.limit)
        self.logger.debug(
            "Initialized resource manager %s with limit %s",
            self.resource_name,
            self.limit or "<unlimited>",
        )

    def __str__(self):
        return f"{self.__class__.__name__}[{self.resource_name}]"

    def _acquire(self, pool: Hashable):
        """
        Reserve an item slot in the underlying resource pool
        """
        if self.is_unlimited:
            return
        self._limit_guard.acquire()

    def _release(self, pool: Hashable):
        """
        Release an item slot to the underlying resource pool
        """
        if self.is_unlimited:
            return
        self._limit_guard.release()

    def _get_resource(self, req: ResourceRequest) -> Any:
        """
        Produce a resource item after a slot is acquired.
        """
        raise NotImplementedError("Must override")

    def _put_resource(self, r: Any, req: ResourceRequest):
        """
        Return a resource item to the manager after the client is done with it
        """
        raise NotImplementedError("Must override")

    @property
    def is_unlimited(self) -> bool:
        return self.limit is None or self.limit == 0

    @contextmanager
    def acquire(self, req: ResourceRequest) -> ContextManager[Any]:
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
            self.logger.error(
                "Failed to produce %s from pool %s", self.resource_name, req.pool
            )
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
                self.logger.error(
                    "Failed to return %s to pool %s", self.resource_name, req.pool
                )
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
    Schedule tasks into workers, handling task dependencies.

    The scheduler manages a thread pool.
    A task queue is used to accumulate tasks to schedule. When all tasks
    have been registered, the scheduler resolves dependencies and submits
    them in order to the thread pool.
    """

    def __init__(self, session: Session):
        """
        :param session: The session for which we are scheduling tasks.
        """
        n_workers = session.config.concurrent_workers or mp.cpu_count()

        #: Parent session
        self.session = session
        #: Scheduler logger
        self.logger = new_logger("sched", self.session.logger)
        #: Task graph, nodes are Task.task_id, each node has the attribute task
        #: containing the task instance to run.
        self._task_graph = nx.DiGraph()
        #: Worker pool executor
        self._executor = ThreadPoolExecutor(max_workers=n_workers)
        #: Resource managers registered
        self._rman = {}
        #: Active futures set
        self.active_set: dict[Future, Task] = {}
        #: Failed tasks, this does not include tasks that were cancelled as a
        #: result of a failure.
        self.failed_tasks = []

    def add_task(self, task: Task):
        self.logger.debug("Schedule task %s", task.task_id)
        if task.task_id in self._task_graph:
            # In our model we can not have tasks with duplicate IDs
            # If a task is already scheduled just skip this, duplicate
            # dependencies are allowed.
            return
        self._task_graph.add_node(task.task_id, task=task)
        for dep in task.dependencies():
            task.resolved_dependencies.add(dep)
            self.add_task(dep)
            self._task_graph.add_edge(task.task_id, dep.task_id)

    def register_resource(self, rman: ResourceManager):
        if rman.resource_name in self._rman:
            self.logger.error(
                "Duplicate resource manager %s: given %s found %s",
                rman.resource_name,
                rman,
                self._rman[rman.resource_name],
            )
            raise ValueError("Duplicate resource manager")
        self._rman[rman.resource_name] = rman

    def resolve_schedule(self):
        """
        Produce a schedule for the tasks.
        """
        try:
            sched = list(nx.topological_sort(self._task_graph))
            tasks = [self._task_graph.nodes[t]["task"] for t in reversed(sched)]
            self.logger.debug(
                "Resolved benchmark schedule:\n%s", "\n".join(map(str, tasks))
            )
            return tasks
        except nx.NetworkXUnfeasible:
            for cycle in nx.simple_cycles(self._task_graph):
                cycle_str = [str(self._task_graph.nodes[c]["task"]) for c in cycle]
                self.logger.error(
                    "Impossible task schedule: cyclic dependency %s", cycle_str
                )
            raise RuntimeError("Impossible to create a task schedule")

    def report_failures(self, logger):
        """
        Report task failures to the given logger
        """
        for failed in self.failed_tasks:
            task_details = []
            if isinstance(failed, DatasetTask):
                iconf = failed.benchmark.config.instance
                task_details.append(
                    f"on {iconf.platform}-{iconf.cheri_target}-{iconf.kernel}"
                )
                task_details.append(f"params: {failed.benchmark.parameters}")
            logger.error(
                "Task %s.%s failed %s",
                failed.task_namespace,
                failed.task_name,
                " ".join(task_details),
            )

    def run(self):
        """
        Schedule and submit tasks to the workers.

        The schedule ensures that tasks will not deadlock waiting for a dependency.
        """
        schedule = deque(self.resolve_schedule())
        with self._executor as pool:
            # Drain tasks from the schedule incrementally, as the dependencies
            # are completed.
            while self.active_set or schedule:
                while schedule:
                    peek_next = schedule[0]
                    self.logger.debug("Peek next task %s", peek_next)
                    deps_done = (
                        dep.completed for dep in peek_next.resolved_dependencies
                    )
                    deps_failed = (
                        dep.failure for dep in peek_next.resolved_dependencies
                    )
                    if any(deps_failed):
                        self.logger.error("Detected failed dependency")
                        break
                    elif all(deps_done):
                        next_task = schedule.popleft()
                        self.logger.debug("Submit task %s", next_task)
                        fut = pool.submit(lambda t: t.execute(), next_task)
                        self.active_set[fut] = next_task
                    else:
                        # If the next task is not ready, topological sort guarantees
                        # that following tasks are also not ready.
                        break
                assert len(self.active_set), "active set drained but nothing scheduled"
                # Wait for the next completion
                done, _ = cf.wait(
                    self.active_set.keys(), return_when=cf.FIRST_COMPLETED
                )
                for fut in done:
                    done_task = self.active_set.pop(fut)
                    try:
                        res = fut.result()
                        self.logger.debug(
                            "Task %s completed with result: %s", done_task, res
                        )
                    except CancelledError:
                        self.logger.debug("Task cancelled %s", done_task)
                    except Exception as err:
                        self.logger.error("Task %s failed: %s", done_task, err)
                        self.failed_tasks.append(done_task)
                        # Drain everything
                        schedule.clear()
                        pool.shutdown(wait=False, cancel_futures=True)
        self.logger.debug("Task schedule completed")
