import csv
import re

import polars as pl
import pytest

from pycheribenchplot.core.analysis import AnalysisTask
from pycheribenchplot.core.artefact import ValueTarget, Target, PLDataFrameLoadTask
from pycheribenchplot.core.config import AnalysisConfig
from pycheribenchplot.core.task import ExecutionTask


class DummyExecTask(ExecutionTask):
    public = True
    task_namespace = "test.analysis-load"

    def run(self):
        pass

    def outputs(self):
        yield "test-target", Target(self, "fake", ext="csv", loader=PLDataFrameLoadTask)


class DummyAnalysisRoot(AnalysisTask):
    """
    This is needed for testing as the load tasks are never meant to be public,
    so we need a top-level task with the correct __init__ signature that can be
    instantiated by the session analysis entry point.
    """
    public = True
    task_namespace = "test.analysis-load"
    task_name = "fake-analysis-main"

    def dependencies(self):
        exec_task = self.session.all_benchmarks()[0].find_exec_task(DummyExecTask)
        yield exec_task.output_map["test-target"].get_loader()

    def run(self):
        return


@pytest.fixture
def csv_file_content(fake_analysis_session):
    # Ensure that the loader will find the csv file where expected
    task = DummyExecTask(fake_analysis_session.parameterization_matrix["descriptor"][0], None)
    output = task.output_map["test-target"]
    path = list(output)[0]
    path.parent.mkdir(exist_ok=True)
    with open(path, "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["number", "name", "surname"])
        writer.writerow([0, "Dennis", "Ritchie"])
        writer.writerow([1, "Bjarne", "Stroustroup"])
        writer.writerow([2, "Graydon", "Hoare"])
        writer.writerow([3, "Claude", "Shannon"])
    yield path
    path.unlink()


@pytest.fixture
def session_config_with_param(single_benchmark_config):
    """
    Modify the single_benchmark_config to add a parameter key
    """
    conf = single_benchmark_config["configurations"][0]
    conf["parameters"] = {"param_key": "param-value"}
    conf["generators"] = [{"handler": "test.analysis-load.exec"}]
    return single_benchmark_config


@pytest.fixture
def fake_analysis_session(session_config_with_param, fake_session_factory):
    session = fake_session_factory(session_config_with_param)
    return session


def test_simple_load_task(csv_file_content, fake_analysis_session):
    """
    Test loading data from a simple CSV file.
    This exercises the common dataframe load task with data validation.
    """
    aconf = AnalysisConfig.schema().load({
        "tasks": [{"handler": "test.analysis-load.fake-analysis-main"}]
    })
    fake_analysis_session.analyse(aconf)

    assert not fake_analysis_session.scheduler.failed_tasks
    # Now check that we produced the dataframe output we are looking for
    tasks = fake_analysis_session.scheduler.completed_tasks
    load_task = None
    for task_id, task in tasks.items():
        print(task_id)
        if re.match(r"internal.pl-target-load-.*-for-test.analysis-load.*-output-fake", task_id):
            load_task = task
            break
    assert load_task is not None, "Load task did not run"

    target = load_task.output_map["df"]
    df = target.get()
    assert isinstance(target, ValueTarget)
    expect_columns = {"dataset_id", "dataset_gid", "iteration", "param_key", "number", "name", "surname"}
    assert set(df.columns) == expect_columns
    assert len(df) == 4
