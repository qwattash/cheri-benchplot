from collections import defaultdict

import pytest

from pycheribenchplot.core.task import ExecutionTask, TaskRegistry


@pytest.fixture
def mock_task_registry(mocker):
    """
    Mock the task registry lists so that the tasks that are declared within the test
    are guaranteed to never conflict with other tests.
    """
    public_tasks = defaultdict(dict)
    all_tasks = defaultdict(dict)
    mock_public_tasks = mocker.patch.object(TaskRegistry, "public_tasks", new=public_tasks)
    mock_all_tasks = mocker.patch.object(TaskRegistry, "all_tasks", new=all_tasks)

    # Always define at least this as it is used by the common session fixtures
    class FakeExecTask(ExecutionTask):
        public = True
        task_namespace = "test-benchmark"

    return (public_tasks, all_tasks)
