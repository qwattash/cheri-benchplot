import time
from unittest.mock import ANY, call

import pytest

from pycheribenchplot.core.task import (ResourceManager, Target, Task, TaskScheduler)


class FakeInvalidTask(Task):
    # Missing the task_name
    task_id = "fake-invalid-task"

    def run(self):
        return


class FakeTaskA(Task):
    task_name = "fake-task-A"
    task_id = "fake-task-A"

    def run(self):
        time.sleep(1)


class FakeTaskB(Task):
    task_name = "fake-task-B"
    task_id = "fake-task-B"

    test_result_task_A_completed = False

    def dependencies(self):
        self.a_task = FakeTaskA()
        yield self.a_task

    def run(self):
        FakeTaskB.test_result_task_A_completed = self.a_task.completed


class FakeTaskC(Task):
    task_name = "fake-task-C"
    task_id = "fake-task-C"

    def dependencies(self):
        yield FakeTaskB()

    def run(self):
        return


class FakeTaskHorizontalDeps(Task):
    task_name = "fake-task-horizontal-deps"
    task_id = "fake-task-horizontal-deps"

    def dependencies(self):
        yield FakeTaskC()
        yield FakeTaskB()

    def run(self):
        return


class FakeTaskCycleA(Task):
    task_name = "fake-task-cycle-A"
    task_id = "fake-task-cycle-A"

    def dependencies(self):
        yield FakeTaskA()
        yield FakeTaskCycleB()

    def run(self):
        return


class FakeTaskCycleB(Task):
    task_name = "fake-task-cycle-B"
    task_id = "fake-task-cycle-B"

    def dependencies(self):
        yield FakeTaskCycleA()

    def run(self):
        return


class FakeTaskCycle(Task):
    task_name = "fake-task-cycle"
    task_id = "fake-task-cycle"

    def dependencies(self):
        yield FakeTaskCycleA()
        yield FakeTaskCycleB()

    def run(self):
        return


class FakeTaskError(Task):
    task_name = "fake-task-error"
    task_id = "fake-task-error"

    def run(self):
        raise RuntimeError("Failed task test")


class FakeTaskDepError(Task):
    task_name = "fake-task-with-dep-error"
    task_id = "fake-task-with-dep-error"

    def dependencies(self):
        self.error_task = FakeTaskError()
        self.valid_task = FakeTaskA()
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
    task_id = "fake-task-with-resource"

    def dependencies(self):
        self.a = FakeTaskA()
        yield self.a

    def resources(self):
        self.r = DummyResource.request()
        yield self.r

    def run(self):
        self.deps_a_completed = self.a.completed
        self.resource_r_value = self.r.get()


class FakeStatefulTarget(Target):
    def __init__(self, value):
        self.value = value


class FakeTaskWithOutput(Task):
    task_name = "fake-task-with-output"
    task_id = "fake-task-with-output"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value = 0xdeadbeef

    def run(self):
        self.value = 0x12345

    def outputs(self):
        """
        Simulate output content dependency on the run() result
        """
        yield "test-output", FakeStatefulTarget(self.value)


def test_invalid_task_check():
    with pytest.raises(AssertionError):
        task = FakeInvalidTask()


def test_task_schedule_simple(fake_session_with_params):
    sched = TaskScheduler(fake_session_with_params)
    bench = fake_session_with_params.benchmark_matrix.iloc[0, 0]
    sched.add_task(FakeTaskB(bench))

    schedule = [t.task_id for t in sched.resolve_schedule()]
    assert len(schedule) == 2
    assert schedule[0] == f"fake-task-A"
    assert schedule[1] == f"fake-task-B"


def test_task_schedule_horizontal_deps(fake_session_with_params):
    sched = TaskScheduler(fake_session_with_params)
    bench = fake_session_with_params.benchmark_matrix.iloc[0, 0]
    sched.add_task(FakeTaskHorizontalDeps(bench))

    schedule = [t.task_id for t in sched.resolve_schedule()]
    assert len(schedule) == 4
    assert schedule[0] == f"fake-task-A"
    assert schedule[1] == f"fake-task-B"
    assert schedule[2] == f"fake-task-C"
    assert schedule[3] == f"fake-task-horizontal-deps"


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


def test_task_registry(mock_task_registry):
    # Mock the task registry dictionaries
    public_tasks, all_tasks = mock_task_registry
    public_tasks.clear()
    all_tasks.clear()

    class TaskA(Task):
        public = True
        task_namespace = "test"
        task_name = "fake-task-A"

    class TaskB(Task):
        public = False
        task_namespace = "test"
        task_name = "fake-task-B"

    class NoNamespaceTask(Task):
        task_name = "foobar"

    class NoNameTask(Task):
        task_namespace = "foobar"

    class AbstractTask(Task):
        pass

    assert public_tasks == {"test": {"fake-task-A": TaskA}}
    assert all_tasks == {"test": {"fake-task-A": TaskA, "fake-task-B": TaskB}}


def test_task_registry_duplicate(mock_task_registry):
    public_tasks, all_tasks = mock_task_registry

    class ExistingTask(Task):
        task_namespace = "test"
        task_name = "fake-dup-task"

    public_tasks.clear()
    all_tasks.clear()
    all_tasks["test"].update({"fake-dup-task": ExistingTask})

    with pytest.raises(ValueError):  # , match=r"[Dd]uplicate"):

        class TaskDup(Task):
            task_namespace = "test"
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


def test_clone_tasks_state(fake_session):
    """
    Clone tasks are tasks that have the same ID but are scheduled multiple times.
    We assume that these are interchangeable so only one will be executed, however
    the state and the results must propagate to its clone(s).
    This is necessary because tasks that depend on the clone will want to access
    its state and outputs. This is relevant only for output targets that are stateful,
    for instance targets holding a dataframe reference.
    """
    sched = TaskScheduler(fake_session)
    task = FakeTaskWithOutput()
    clone = FakeTaskWithOutput()

    sched.add_task(task)
    sched.add_task(clone)
    sched.run()

    assert not sched.failed_tasks
    assert len(sched.completed_tasks) == 1
    assert task.completed
    assert task.output_map["test-output"].value == 0x12345
    assert clone.completed
    assert clone.output_map["test-output"].value == 0x12345
    # Borg invariant
    assert task.__dict__ is clone.__dict__
