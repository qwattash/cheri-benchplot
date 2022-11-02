from unittest import mock

import pandas as pd
import pytest

from pycheribenchplot.core.config import (AnalysisConfig, DatasetConfig, DatasetName)
from pycheribenchplot.core.elf.symbolizer import SymInfo
from pycheribenchplot.pmc.analysis import PMCStacksPlot
from pycheribenchplot.pmc.dataset import ProfclockStackSamples


@pytest.fixture
def stacksample_dset_factory(fake_simple_benchmark):
    """
    Make a constructor of dummy stack samples datasets.
    """
    def factory():
        config = DatasetConfig(handler=DatasetName.PMC_PROFCLOCK_STACKSAMPLE, run_options={})
        return ProfclockStackSamples(fake_simple_benchmark, config)

    return factory


@pytest.fixture
def fake_loaded_stacksamples(stacksample_dset_factory, fake_simple_benchmark):
    """
    Simulate loaded stack samples
    """
    fake_ds_1 = stacksample_dset_factory()
    fake_ds_2 = stacksample_dset_factory()

    fake_df_1 = pd.DataFrame({
        "dataset_id": ["bench0", "bench0", "bench0"],
        "dataset_gid": ["machine0", "machine0", "machine0"],
        "iteration": [0, 0, 0],
        "seq": [0, 1, 2],
        "cpu": [0, 0, 0],
        "pid": [1, 1, 1],
        "tid": [1, 1, 1],
        "mode": ["s", "s", "s"],
        "stacktrace": [[0x10000001, 0x10000002, 0x10000003], [0x20000001, 0x20000002, 0x20000003],
                       [0x30000001, 0x30000002, 0x30000003]]
    })

    fake_df_2 = pd.DataFrame({
        "dataset_id": ["bench0", "bench0", "bench0"],
        "dataset_gid": ["machine0", "machine0", "machine0"],
        "iteration": [1, 1, 1],
        "seq": [0, 1, 2],
        "cpu": [0, 0, 0],
        "pid": [1, 1, 1],
        "tid": [1, 1, 1],
        "mode": ["s", "s", "s"],
        "stacktrace": [[0x1A000001, 0x1A000002, 0x1A000003], [0x2A000001, 0x2A000002, 0x2A000003],
                       [0x3A000001, 0x3A000002, 0x3A000003]]
    })

    # Fake PID mapping
    fake_simple_benchmark.pidmap.df = pd.DataFrame({
        "dataset_id": ["bench0", "bench0"],
        "dataset_gid": ["machine0", "machine0"],
        "iteration": [0, 1],
        "pid": [1, 1],
        "tid": [1, 1],
        "command": ["fake_command", "fake_command"],
        "thread_name": ["fake_thread", "fake_thread"],
        "thread_flags": [0, 0],
        "proc_flags": [0, 0]
    }).set_index(["dataset_id", "dataset_gid", "iteration"])
    # Fixup uuids to match
    fake_simple_benchmark.config.uuid = "bench0"
    fake_simple_benchmark.config.g_uuid = "machine0"
    fake_simple_benchmark.session.baseline_g_uuid = "machine0"

    # Populate fake datasets
    fake_ds_1._append_df(fake_df_1)
    fake_ds_1.max_stacktrace_depth = 3
    fake_ds_2._append_df(fake_df_2)
    fake_ds_2.max_stacktrace_depth = 3

    return (fake_ds_1, fake_ds_2)


@pytest.fixture
def fake_merged_stacksamples(mocker, fake_loaded_stacksamples, fake_simple_benchmark, fake_benchmark_factory):
    """
    Simulate merged stack samples from multiple benchmark variants
    """
    fake_ds_1, _ = fake_loaded_stacksamples

    # Split by iteration to make setup more readable
    df_i0 = pd.DataFrame({
        "dataset_id": ["bench0", "bench0", "bench0"],
        "dataset_gid": ["machine0", "machine0", "machine0"],
        "iteration": [0, 0, 0],
        "seq": [0, 1, 2],
        "cpu": [0, 0, 0],
        "pid": [1, 1, 1],
        "tid": [1, 1, 1],
        "process": ["fake_proc", "fake_proc", "fake_proc"],
        "stacktrace_0": map(SymInfo.unknown, [0x10000003, 0x20000003, 0x30000003]),
        "stacktrace_1": map(SymInfo.unknown, [0x10000002, 0x20000002, 0x30000002]),
        "stacktrace_2": map(SymInfo.unknown, [0x10000001, 0x20000001, 0x30000001]),
    })
    df_i1 = pd.DataFrame({
        "dataset_id": ["bench0", "bench0", "bench0"],
        "dataset_gid": ["machine0", "machine0", "machine0"],
        "iteration": [1, 1, 1],
        "seq": [0, 1, 2],
        "cpu": [0, 0, 0],
        "pid": [1, 1, 1],
        "tid": [1, 1, 1],
        "process": ["fake_proc", "fake_proc", "fake_proc"],
        "stacktrace_0": map(SymInfo.unknown, [0x1A000003, 0x2A000003, 0x3A000003]),
        "stacktrace_1": map(SymInfo.unknown, [0x1A000002, 0x2A000002, 0x3A000002]),
        "stacktrace_2": map(SymInfo.unknown, [0x1A000001, 0x2A000001, 0x3A000001]),
    })
    df = pd.concat([df_i0, df_i1])
    df["nsamples"] = 1
    # Duplicate the data for another benchmark variants and alter the nsamples column
    tmp_df = pd.concat([df_i0, df_i1])
    tmp_df["dataset_id"] = "bench1"
    tmp_df["dataset_gid"] = "machine1"
    tmp_df["nsamples"] = 10
    # Merge everything
    merged_df = pd.concat([df, tmp_df])
    merged_df.set_index(fake_ds_1.df.index.names, inplace=True)
    merged_df.columns.name = "metric"
    fake_ds_1.merged_df = merged_df

    # Simulate a new benchmark on machine1 in the merged benchmarks dict
    merged_benchmark = fake_benchmark_factory()
    merged_benchmark.config.uuid = "bench1"
    merged_benchmark.config.g_uuid = "machine1"
    merged_benchmark.config.instance.baseline = False
    fake_simple_benchmark.get_benchmark_groups = mocker.MagicMock(return_value={
        "machine0": [fake_simple_benchmark],
        "machine1": [merged_benchmark]
    })

    return fake_ds_1


def test_stacktrace_samples_merge(fake_loaded_stacksamples):
    """
    Test stacksample merging
    """
    fake_ds_1, fake_ds_2 = fake_loaded_stacksamples
    fake_ds_1.pre_merge()
    fake_ds_2.pre_merge()
    fake_ds_1.init_merge()
    fake_ds_1.merge(fake_ds_2)

    # Check annotated merged stack traces
    check_df = fake_ds_1.merged_df
    assert fake_ds_1.stacktrace_columns() == ["stacktrace_0", "stacktrace_1", "stacktrace_2"]
    assert "stacktrace_0" in check_df.columns
    assert "stacktrace_1" in check_df.columns
    assert "stacktrace_2" in check_df.columns
    # Expect 6 distinct samples
    assert len(check_df) == 6
    assert (check_df["nsamples"] == 1).all()
    # Check stacktrace syminfo
    xs = check_df.xs(0, level="iteration")
    assert (xs["stacktrace_0"].map(lambda v: v.addr) == [0x10000003, 0x20000003, 0x30000003]).all()
    assert (xs["stacktrace_1"].map(lambda v: v.addr) == [0x10000002, 0x20000002, 0x30000002]).all()
    assert (xs["stacktrace_2"].map(lambda v: v.addr) == [0x10000001, 0x20000001, 0x30000001]).all()
    xs = check_df.xs(1, level="iteration")
    assert (xs["stacktrace_0"].map(lambda v: v.addr) == [0x1A000003, 0x2A000003, 0x3A000003]).all()
    assert (xs["stacktrace_1"].map(lambda v: v.addr) == [0x1A000002, 0x2A000002, 0x3A000002]).all()
    assert (xs["stacktrace_2"].map(lambda v: v.addr) == [0x1A000001, 0x2A000001, 0x3A000001]).all()
    # Don't expect any symbol to be valid as we have no symbol sources loaded
    assert check_df[fake_ds_1.stacktrace_columns()].applymap(lambda si: si.is_unknown).all().all()


def test_stacktrace_samples_aggregate(fake_merged_stacksamples):
    fake_ds = fake_merged_stacksamples

    fake_ds.aggregate()
    fake_ds.post_aggregate()
    check_df = fake_ds.agg_df
    assert len(check_df) == 12
    xs = check_df.xs("bench0", level="dataset_id")
    assert (xs[("nsamples", "median", "sample")] == 1).all()
    assert (xs[("nsamples", "median", "delta_baseline")] == 0).all()

    xs = check_df.xs("bench1", level="dataset_id")
    assert (xs[("nsamples", "median", "sample")] == 10).all()
    assert (xs[("nsamples", "median", "delta_baseline")] == 9).all()


def test_stacktrace_samples_analysis(mocker, fake_merged_stacksamples, fake_simple_benchmark):
    fake_ds = fake_merged_stacksamples
    fake_ds.aggregate()
    fake_ds.post_aggregate()

    analysis = PMCStacksPlot(fake_simple_benchmark, AnalysisConfig())
    mock_emit = mocker.patch.object(analysis, "_emit_folded_stacks")
    analysis.get_dataset = mocker.MagicMock(return_value=fake_ds)
    analysis.process_datasets()

    expect_folded_stacks = [
        "??`0x10000003;??`0x10000002;??`0x10000001",
        "??`0x1a000003;??`0x1a000002;??`0x1a000001",
        "??`0x20000003;??`0x20000002;??`0x20000001",
        "??`0x2a000003;??`0x2a000002;??`0x2a000001",
        "??`0x30000003;??`0x30000002;??`0x30000001",
        "??`0x3a000003;??`0x3a000002;??`0x3a000001",
    ]
    mock_emit.assert_called_once()
    check_df = mock_emit.call_args.args[2]
    assert len(check_df) == 6
    assert (check_df["folded_stacks"] == expect_folded_stacks).all()
    assert (check_df[("nsamples", "median", "sample", "machine0")] == 1).all()
    assert (check_df[("nsamples", "median", "sample", "machine1")] == 10).all()
    assert len(check_df.columns) == 3
