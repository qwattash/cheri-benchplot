import random
import time
from concurrent.futures import ThreadPoolExecutor, wait

import pytest

from pycheribenchplot.core.scheduler import ResourceManager, TaskScheduler
from pycheribenchplot.core.task import Task, dependency, output

from .util.session import *


@pytest.fixture
def sched(task_factory):
    session = task_factory.build()
    sched = TaskScheduler(session)
    return sched


class SimpleTestTask(Task):
    task_namespace = "test.scheduler"
    task_name = "simple"

    @property
    def task_id(self):
        return "test.scheduler.simple"

    def run(self):
        time.sleep(random.uniform(0.05, 0.1))
        return


def test_task_schedule_simple(sched, task_factory):
    task = task_factory.build_task(SimpleTestTask)

    sched.add_task(task)
    schedule = sched.resolve_schedule()
    assert len(schedule) == 1

    sched.run()
    assert len(sched.failed_tasks) == 0
    assert task.completed


class FailingTestTask(Task):
    task_namespace = "test.scheduler"
    task_name = "failing"

    @property
    def task_id(self):
        return "test.scheduler.failing"

    def run(self):
        raise Exception("test failure")


def test_task_schedule_fail(sched, task_factory):
    task = task_factory.build_task(FailingTestTask)

    sched.add_task(task)
    sched.run()
    assert len(sched.failed_tasks) == 1
    assert task.completed
    assert sched.failed_tasks[0] == task


class Cyclic0TestTask(Task):
    task_namespace = "test.scheduler"
    task_name = "cycle0"

    @dependency
    def other(self):
        return Cyclic1TestTask()

    @property
    def task_id(self):
        return "test.scheduler.cycle0"

    def run(self):
        pass


class Cyclic1TestTask(Task):
    task_namespace = "test.scheduler"
    task_name = "cycle1"

    @dependency
    def other(self):
        return Cyclic0TestTask()

    @property
    def task_id(self):
        return "test.scheduler.cycle1"

    def run(self):
        pass


def test_task_schedule_cycle(sched, task_factory):
    task0 = task_factory.build_task(Cyclic0TestTask)
    task1 = task_factory.build_task(Cyclic1TestTask)

    sched.add_task(task0)
    sched.add_task(task1)
    with pytest.raises(Exception):
        sched.resolve_schedule()

    with pytest.raises(Exception):
        sched.run()


class DelayedTestTask(Task):
    task_namespace = "test.scheduler"
    task_name = "delayed"

    @property
    def task_id(self):
        return "test.scheduler.delayed"

    def run(self):
        time.sleep(random.uniform(0.5, 0.2))
        return


class SimpleDepTestTask(Task):
    task_namespace = "test.scheduler"
    task_name = "simple-dep"

    @dependency
    def other(self):
        return DelayedTestTask()

    @property
    def task_id(self):
        return "test.scheduler.simple-dep"

    def run(self):
        assert self.other.completed
        time.sleep(0.01)
        return


def test_task_with_dependency(sched, task_factory):
    task = task_factory.build_task(SimpleDepTestTask)

    sched.add_task(task)
    order = sched.resolve_schedule()
    assert len(order) == 2
    dep = task_factory.build_task(DelayedTestTask)
    assert order == [dep, task]

    # Run the tasks in order with a custom executor
    sched.run()
    assert task.completed
    assert dep.completed


class WithFailDepTestTask(Task):
    task_namespace = "test.scheduler"
    task_name = "with-fail-dep"

    @dependency
    def other(self):
        yield FailingTestTask()
        # This will be scheduled immediately afterwards because
        # it is a dependency, so this task will remain unscheduled
        yield DelayedTestTask()

    @property
    def task_id(self):
        return "test.scheduler.with-fail-dep"

    def run(self):
        assert False, "not reached"


def test_task_cancellation(task_factory):
    session = task_factory.build()
    session.config.concurrent_workers = 1
    sched = TaskScheduler(session)

    failing = task_factory.build_task(FailingTestTask)
    to_cancel = task_factory.build_task(WithFailDepTestTask)
    sched.add_task(failing)
    sched.add_task(to_cancel)
    sched.run()
    assert failing.completed
    assert failing.failure is not None
    assert to_cancel.completed == False
    assert to_cancel.failure is None


# def test_task_resource_acquisition()
