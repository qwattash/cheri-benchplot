from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pycheribenchplot.core.analysis import AnalysisTask, DatasetAnalysisTask
from pycheribenchplot.core.artefact import *
from pycheribenchplot.core.config import AnalysisConfig
from pycheribenchplot.core.task import DataGenTask, SessionDataGenTask


@pytest.fixture
def fake_s_datagen_task(mock_task_registry, fake_session):
    class FakeTask(SessionDataGenTask):
        task_namespace = "test"
        task_name = "fake-task"

    return FakeTask(fake_session)


@pytest.fixture
def fake_datagen_task(mock_task_registry, fake_simple_benchmark):
    class FakeTask(DataGenTask):
        task_namespace = "test"
        task_name = "fake-task"

    fake_simple_benchmark.session.config.remote_session_path = Path("/test/remote")
    return FakeTask(fake_simple_benchmark, None)


@pytest.fixture
def fake_s_analysis_task(mock_task_registry, fake_session):
    class FakeTask(AnalysisTask):
        task_namespace = "test"
        task_name = "fake-task"

    return FakeTask(fake_session, AnalysisConfig())


@pytest.fixture
def fake_analysis_task(mock_task_registry, fake_simple_benchmark):
    class FakeTask(DatasetAnalysisTask):
        task_namespace = "test"
        task_name = "fake-task"

    return FakeTask(fake_simple_benchmark, AnalysisConfig())


@pytest.fixture
def sample_content():
    """
    Produce some sample data to store in a test file that conforms to FakeModel
    """
    df = pl.DataFrame({"index": [1, 2, 3], "value": [10, 20, 30]})
    return df


def test_target_session_exec_task(fake_s_datagen_task):
    target = Target(fake_s_datagen_task, "OUTID")
    task_uuid = fake_s_datagen_task.session.uuid
    expect_path = fake_s_datagen_task.session.get_data_root_path() / f"OUTID-test-fake-task-{task_uuid}.txt"

    assert target.get_base_path() == f"OUTID-test-fake-task-{task_uuid}"
    assert target.get_root_path() == fake_s_datagen_task.session.get_data_root_path()
    paths = list(target)
    assert len(paths) == 1
    assert paths[0] == expect_path


def test_target_exec_task(fake_datagen_task):
    target = Target(fake_datagen_task, "OUTID")
    task_uuid = fake_datagen_task.benchmark.uuid
    expect_path = fake_datagen_task.benchmark.get_benchmark_data_path() / f"OUTID-test-fake-task-{task_uuid}.txt"

    assert target.get_base_path() == f"OUTID-test-fake-task-{task_uuid}"
    assert target.get_root_path() == fake_datagen_task.benchmark.get_benchmark_data_path()
    paths = list(target)
    assert len(paths) == 1
    assert paths[0] == expect_path


def test_target_session_analysis_task(fake_s_analysis_task):
    target = Target(fake_s_analysis_task, "OUTID")
    task_uuid = fake_s_analysis_task.session.uuid
    expect_path = fake_s_analysis_task.session.get_analysis_root_path() / f"OUTID-test-fake-task-{task_uuid}.txt"

    assert target.get_base_path() == f"OUTID-test-fake-task-{task_uuid}"
    assert target.get_root_path() == fake_s_analysis_task.session.get_analysis_root_path()
    paths = list(target)
    assert len(paths) == 1
    assert paths[0] == expect_path


def test_target_analysis_task(fake_analysis_task):
    target = Target(fake_analysis_task, "OUTID")
    task_uuid = fake_analysis_task.benchmark.uuid
    expect_path = fake_analysis_task.benchmark.get_analysis_path() / f"OUTID-test-fake-task-{task_uuid}.txt"

    assert target.get_base_path() == f"OUTID-test-fake-task-{task_uuid}"
    assert target.get_root_path() == fake_analysis_task.benchmark.get_analysis_path()
    paths = list(target)
    assert len(paths) == 1
    assert paths[0] == expect_path


@pytest.mark.skip("Need to implement dataframe validators")
def test_dataframe_target(fake_s_analysis_task):
    test_df = pl.DataFrame({"index": [1, 2, 3], "value": [10, 20, 30]})
    # target = DataFrameTarget(fake_s_analysis_task, FakeModel)
    # task_uuid = fake_s_analysis_task.session.uuid

    # assert target.get_base_path() == f"fakemodel-test-fake-task-{task_uuid}"

    # target.assign(test_df)

    # df = target.get()
    # assert_frame_equal(df, test_df)

    # # Test Borg behaviour
    # target2 = DataFrameTarget(fake_s_analysis_task, FakeModel)
    # df = target2.get()
    # assert_frame_equal(df, test_df)


@pytest.mark.skip("Need to implement dataframe validators")
def test_dataframe_target_with_ext(fake_s_analysis_task):
    # target = DataFrameTarget(fake_s_analysis_task, FakeModel, ext="csv")
    # task_uuid = fake_s_analysis_task.session.uuid

    # assert target.get_base_path() == f"fakemodel-test-fake-task-{task_uuid}"
    # paths = list(target)
    # assert len(paths) == 1
    # assert paths[0].name == f"fakemodel-test-fake-task-{task_uuid}.csv"
    pass


@pytest.mark.skip("Need to implement dataframe validators")
def test_dataframe_target_without_model(fake_s_analysis_task):
    # test_df = pl.DataFrame({"index": [1, 2, 3], "value": [10, 20, 30]})
    # target = DataFrameTarget(fake_s_analysis_task, None)
    # task_uuid = fake_s_analysis_task.session.uuid

    # assert target.get_base_path() == f"frame-test-fake-task-{task_uuid}"

    # target.assign(test_df)
    # df = target.get()
    # assert_frame_equal(df, test_df)
    pass


@pytest.mark.skip("Need to implement dataframe validators")
def test_dataframe_target_invalid(fake_s_analysis_task):
    # invalid_df = pl.DataFrame({"invalid": [1, 2, 3], "value": ["foo", "bar", "baz"]})
    # target = DataFrameTarget(fake_s_analysis_task, FakeModel)

    # with pytest.raises(SchemaError):
    #     target.assign(invalid_df)
    pass


def test_session_remote_file_target(fake_datagen_task):
    target = RemoteTarget(fake_datagen_task, "OUTID", ext="EXT")
    task_uuid = fake_datagen_task.benchmark.uuid

    paths = list(target.remote_paths())
    assert len(paths) == 1
    assert paths[0].parent == Path(f"/test/remote/run/selftest0-{fake_datagen_task.uuid}")
    assert paths[0].name == f"OUTID-test-fake-task-{task_uuid}.EXT"


def test_iteration_target(fake_datagen_task):
    fake_datagen_task.benchmark.config.iterations = 3
    target = BenchmarkIterationTarget(fake_datagen_task, "OUTID", ext="EXT")
    task_uuid = fake_datagen_task.benchmark.uuid

    paths = list(target)
    assert len(paths) == 3
    for i, p in enumerate(paths):
        assert p.parent == fake_datagen_task.benchmark.get_benchmark_data_path() / str(i)
        assert p.name == f"OUTID-test-fake-task-{task_uuid}.EXT"


def test_target_with_prefix(fake_datagen_task):
    target = Target(fake_datagen_task, "OUTID", prefix="myprefix", ext="EXT")
    task_uuid = fake_datagen_task.benchmark.uuid

    paths = list(target)
    assert len(paths) == 1
    assert paths[0].parent == fake_datagen_task.benchmark.get_benchmark_data_path() / "myprefix"
    assert paths[0].name == f"OUTID-test-fake-task-{task_uuid}.EXT"


def test_file_target_loader(fake_s_datagen_task, sample_content):
    task_uuid = fake_s_datagen_task.session.uuid
    # Expect the content to be here
    expect_path = fake_s_datagen_task.session.get_data_root_path() / f"OUTID-test-fake-task-{task_uuid}.csv"
    sample_content.write_csv(expect_path)

    target = Target(fake_s_datagen_task, "OUTID", loader=PLDataFrameSessionLoadTask, ext="csv")

    loader = target.get_loader()
    assert loader.is_session_task()
    assert loader.is_dataset_task() == False
    assert loader.is_exec_task() == False

    loader.run()
    assert_frame_equal(loader.df.get(), sample_content)


def test_benchmark_file_target_loader(fake_datagen_task, sample_content):
    task_uuid = fake_datagen_task.benchmark.uuid
    # Expect the content to be here
    fake_datagen_task.benchmark.get_benchmark_data_path().mkdir(exist_ok=True)
    sample_content.write_csv(fake_datagen_task.benchmark.get_benchmark_data_path() /
                             f"OUTID-test-fake-task-{task_uuid}.csv")
    # Expect that the loaded data will have the dataset_id, dataset_gid and iteration index levels
    expect_content = sample_content.with_columns(
        pl.lit(fake_datagen_task.benchmark.uuid).alias("dataset_id"),
        pl.lit(fake_datagen_task.benchmark.g_uuid).alias("dataset_gid"),
        pl.lit(-1).alias("iteration"))

    target = Target(fake_datagen_task, "OUTID", loader=PLDataFrameLoadTask, ext="csv")

    loader = target.get_loader()
    assert loader.is_dataset_task()
    assert loader.is_session_task() == False
    assert loader.is_exec_task() == False

    loader.run()
    assert_frame_equal(loader.df.get(), expect_content)
