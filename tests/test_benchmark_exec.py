import copy

import pytest

from conftest import FakeExecTask
from pycheribenchplot.core.benchmark import BenchmarkExecTask, ExecTaskConfig
from pycheribenchplot.core.task import ExecutionTask


class FakeAuxExecTask(ExecutionTask):
    public = True
    task_namespace = "test-aux"


@pytest.fixture
def benchmark_config_with_aux(single_benchmark_config):
    conf = copy.deepcopy(single_benchmark_config)
    bench_conf = conf["configurations"][0]
    bench_conf["aux_tasks"] = [{"handler": "test-aux"}]
    return conf


def test_benchmark_task_deps(single_benchmark_config, fake_benchmark_factory):
    """
    Check that the benchmark exec task generates the expected dependencies
    """
    bench = fake_benchmark_factory(single_benchmark_config["configurations"][0])
    task = BenchmarkExecTask(bench, task_config=ExecTaskConfig())

    deps = list(task.dependencies())
    # Expect only the main test-benchmark.exec task and the instance-boot task
    assert len(deps) == 1
    assert [d for d in deps if isinstance(d, FakeExecTask)]


def test_benchmark_task_aux_deps(benchmark_config_with_aux, fake_benchmark_factory):
    """
    Check that the benchmark exec task generates the expected auxiliary dependencies
    """
    bench = fake_benchmark_factory(benchmark_config_with_aux["configurations"][0])
    task = BenchmarkExecTask(bench, task_config=ExecTaskConfig())

    deps = list(task.dependencies())
    # Expect only the main test-benchmark.exec task and the instance-boot task
    assert len(deps) == 2
    assert [d for d in deps if isinstance(d, FakeExecTask)]
    assert [d for d in deps if isinstance(d, FakeAuxExecTask)]
