import time
from collections import defaultdict
from unittest.mock import ANY, PropertyMock, call

import pytest

from pycheribenchplot.core.task import ResourceManager, Task, TaskScheduler


class FakeInvalidTask(Task):
    # Missing the task_name

    def run(self):
        return


class FakeTaskA(Task):
    task_name = "fake-task-A"

    def run(self):
        time.sleep(1)


class FakeTaskB(Task):
    task_name = "fake-task-B"

    test_result_task_A_completed = False

    def dependencies(self):
        self.a_task = FakeTaskA(self.benchmark)
        yield self.a_task

    def run(self):
        FakeTaskB.test_result_task_A_completed = self.a_task.completed


class FakeTaskC(Task):
    task_name = "fake-task-C"

    def dependencies(self):
        yield FakeTaskB(self.benchmark)

    def run(self):
        return


class FakeTaskHorizontalDeps(Task):
    task_name = "fake-task-horizontal-deps"

    def dependencies(self):
        yield FakeTaskC(self.benchmark)
        yield FakeTaskB(self.benchmark)

    def run(self):
        return


class FakeTaskCycleA(Task):
    task_name = "fake-task-cycle-A"

    def dependencies(self):
        yield FakeTaskA(self.benchmark)
        yield FakeTaskCycleB(self.benchmark)

    def run(self):
        return


class FakeTaskCycleB(Task):
    task_name = "fake-task-cycle-B"

    def dependencies(self):
        yield FakeTaskCycleA(self.benchmark)

    def run(self):
        return


class FakeTaskCycle(Task):
    task_name = "fake-task-cycle"

    def dependencies(self):
        yield FakeTaskCycleA(self.benchmark)
        yield FakeTaskCycleB(self.benchmark)

    def run(self):
        return


class FakeTaskError(Task):
    task_name = "fake-task-error"

    def run(self):
        raise RuntimeError("Failed task test")


class FakeTaskDepError(Task):
    task_name = "fake-task-with-dep-error"

    def dependencies(self):
        self.error_task = FakeTaskError(self.benchmark)
        self.valid_task = FakeTaskA(self.benchmark)
        yield from [self.error_task, self.valid_task]

    def run(self):
        return


class DummyResource(ResourceManager):
    resource_name = "dummy-resource"

    def _get_resource(self, req):
        return "dummy"

    def _put_resource(self, r, req):
        return


class FakeTaskWithResource(Task):
    task_name = "fake-task-with-resource"

    def dependencies(self):
        self.a = FakeTaskA(self.benchmark)
        yield self.a

    def resources(self):
        self.r = DummyResource.request()
        yield self.r

    def run(self):
        self.deps_a_completed = self.a.completed
        self.resource_r_value = self.r.get()


def test_invalid_task_check(fake_simple_benchmark):
    with pytest.raises(AssertionError):
        task = FakeInvalidTask(fake_simple_benchmark)


def test_task_schedule_simple(fake_session_with_params):
    sched = TaskScheduler(fake_session_with_params)
    bench = fake_session_with_params.benchmark_matrix.iloc[0, 0]
    sched.add_task(FakeTaskB(bench))

    schedule = [t.task_id for t in sched.resolve_schedule()]
    assert len(schedule) == 2
    assert schedule[0] == f"internal.fake-task-A-{bench.uuid}"
    assert schedule[1] == f"internal.fake-task-B-{bench.uuid}"


def test_task_schedule_horizontal_deps(fake_session_with_params):
    sched = TaskScheduler(fake_session_with_params)
    bench = fake_session_with_params.benchmark_matrix.iloc[0, 0]
    sched.add_task(FakeTaskHorizontalDeps(bench))

    schedule = [t.task_id for t in sched.resolve_schedule()]
    assert len(schedule) == 4
    assert schedule[0] == f"internal.fake-task-A-{bench.uuid}"
    assert schedule[1] == f"internal.fake-task-B-{bench.uuid}"
    assert schedule[2] == f"internal.fake-task-C-{bench.uuid}"
    assert schedule[3] == f"internal.fake-task-horizontal-deps-{bench.uuid}"


def test_task_schedule_cyclic_deps(fake_session_with_params):
    sched = TaskScheduler(fake_session_with_params)
    bench = fake_session_with_params.benchmark_matrix.iloc[0, 0]
    sched.add_task(FakeTaskCycle(bench))

    with pytest.raises(RuntimeError):
        sched.resolve_schedule()


@pytest.mark.timeout(5)
def test_task_run(fake_session_with_params, mocker):
    fake_session_with_params.config.concurrent_workers = 2
    sched = TaskScheduler(fake_session_with_params)
    bench = fake_session_with_params.benchmark_matrix.iloc[0, 0]
    task = FakeTaskB(bench)

    sched.add_task(task)
    sched.run()
    assert len(task.resolved_dependencies) == 1
    for t in task.resolved_dependencies:
        assert t.completed
        assert not t.failed
    assert task.completed


@pytest.mark.timeout(5)
def test_task_run_with_error(single_benchmark_config, fake_session_factory, mocker):
    # Setup the fake session
    single_benchmark_config["concurrent_workers"] = 2
    session = fake_session_factory(single_benchmark_config)
    # Required test objects setup
    sched = TaskScheduler(session)
    bench = session.benchmark_matrix.iloc[0, 0]
    task = FakeTaskError(bench)

    # Run the scheduler and check
    sched.add_task(task)
    sched.run()
    assert len(task.resolved_dependencies) == 0
    assert task.completed
    assert task.failed


@pytest.mark.timeout(5)
def test_task_run_with_dependency_error(single_benchmark_config, fake_session_factory, mocker):
    # Setup the fake session
    single_benchmark_config["concurrent_workers"] = 2
    session = fake_session_factory(single_benchmark_config)
    # Required test objects setup

    sched = TaskScheduler(session)
    bench = session.benchmark_matrix.iloc[0, 0]
    task = FakeTaskDepError(bench)

    # Run the scheduler and check
    sched.add_task(task)
    sched.run()
    assert len(task.resolved_dependencies) == 2
    assert task.error_task.completed
    assert task.error_task.failed
    assert task.valid_task.completed
    assert not task.valid_task.failed
    assert task.completed
    assert task.failed


def test_task_registry(mocker):
    # Mock the task registry dictionaries
    public_tasks = defaultdict(dict)
    all_tasks = defaultdict(dict)
    mock_public_tasks = mocker.patch("pycheribenchplot.core.task.TaskRegistry.public_tasks", new_callable=PropertyMock)
    mock_public_tasks.return_value = public_tasks
    mock_all_tasks = mocker.patch("pycheribenchplot.core.task.TaskRegistry.all_tasks", new_callable=PropertyMock)
    mock_all_tasks.return_value = all_tasks

    class TaskA(Task):
        public = True
        task_name = "fake-task-A"

    class TaskB(Task):
        public = False
        task_name = "fake-task-B"

    class AbstractTask(Task):
        pass

    assert public_tasks == {"internal": {"fake-task-A": TaskA}}
    assert all_tasks == {"internal": {"fake-task-A": TaskA, "fake-task-B": TaskB}}


def test_task_registry_duplicate(mocker):
    # Mock the task registry dictionaries
    mock_public_tasks = mocker.patch("pycheribenchplot.core.task.TaskRegistry.public_tasks",
                                     new_callable=PropertyMock())
    mock_all_tasks = mocker.patch("pycheribenchplot.core.task.TaskRegistry.all_tasks", new_callable=PropertyMock())

    class ExistingTask(Task):
        task_name = "fake-dup-task"

    mock_all_tasks.return_value = {"fake-dup-task": ExistingTask}

    class TaskDup(Task):
        task_name = "fake-dup-task"


@pytest.mark.timeout(5)
def test_task_scheduler_rman_simple(mocker, single_benchmark_config, fake_session_factory):
    """
    Test a simple case of a task requiring a dummy resource
    """
    single_benchmark_config["concurrent_workers"] = 2
    session = fake_session_factory(single_benchmark_config)
    bench = session.benchmark_matrix.iloc[0, 0]
    sched = TaskScheduler(session)
    rman = DummyResource(session, limit=1)
    sched.register_resource(rman)
    test_task = FakeTaskWithResource(bench)
    spy_task_run = mocker.patch.object(test_task, "run", wraps=test_task.run)

    sched.add_task(test_task)
    sched.run()

    spy_task_run.assert_called_once()
    assert test_task.deps_a_completed
    assert test_task.resource_r_value == "dummy"


def test_task_scheduler_rman_name_collision(fake_session):
    """
    Check that we can not register multiple resource managers with the same name
    """
    sched = TaskScheduler(fake_session)
    rman1 = DummyResource(fake_session, limit=None)
    rman2 = DummyResource(fake_session, limit=None)

    sched.register_resource(rman1)
    with pytest.raises(ValueError, match=r"already registered"):
        sched.register_resource(rman2)
