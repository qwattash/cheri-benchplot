from collections import defaultdict

import pytest

from pycheribenchplot.core.task import ExecutionTask


@pytest.fixture
def mock_task_registry(mocker):
    """
    Mock the task registry lists so that the tasks that are declared within the test
    are guaranteed to never conflict with other tests.
    """
    public_tasks = defaultdict(dict)
    all_tasks = defaultdict(dict)
    mock_public_tasks = mocker.patch("pycheribenchplot.core.task.TaskRegistry.public_tasks",
                                     new_callable=mocker.PropertyMock)
    mock_public_tasks.return_value = public_tasks
    mock_all_tasks = mocker.patch("pycheribenchplot.core.task.TaskRegistry.all_tasks", new_callable=mocker.PropertyMock)
    mock_all_tasks.return_value = all_tasks

    # Always define at least this as it is used by the common session fixtures
    class FakeExecTask(ExecutionTask):
        public = True
        task_name = "test-benchmark"

    return (public_tasks, all_tasks)
