from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from pycheribenchplot.core.benchmark import (BenchmarkBase, BenchmarkDataSetConfig)
from pycheribenchplot.core.dataset import *


class FakeDataset(DataSetContainer):
    fields = [Field.data_field("m0"), Field.data_field("m1"), Field.index_field("custom_index"), Field("ignore_me")]


@pytest.fixture
def fake_merged_df():
    df = pd.DataFrame()
    df["m0"] = [1, 2, 3, 10, 20, 30]
    df["m1"] = [4, 5, 6, 10, 14, 30]
    df["ignore_me"] = range(100, 106)
    df.columns.name = "metric"
    df["dataset_id"] = ["uuid-0"] * 3 + ["uuid-1"] * 3
    df["custom_index"] = [0, 1, 2] * 2
    df = df.set_index(["dataset_id", "custom_index"])
    return df


@pytest.fixture
def fake_dataset(mocker, fake_merged_df):
    fake_benchmark = mocker.MagicMock(BenchmarkBase)
    fake_benchmark.logger = mocker.Mock()
    fake_benchmark.uuid = "uuid-0"
    fake_config = mocker.Mock(BenchmarkDataSetConfig)
    ds = FakeDataset(fake_benchmark, "test", fake_config)
    ds.merged_df = fake_merged_df
    return ds


def test_aggregation(fake_dataset):
    agg = fake_dataset._compute_aggregations(fake_dataset.merged_df.groupby("dataset_id"))
    assert len(agg) == 2

    m0 = agg["m0"]
    assert (m0["median"] == [2, 20]).all()
    assert (m0["mean"] == [2, 20]).all()
    assert (m0["q25"] == [1.5, 15]).all()
    assert (m0["q75"] == [2.5, 25]).all()
    assert (m0["std"] == [1, 10]).all()

    m1 = agg["m1"]
    assert (m1["median"] == [5, 14]).all()
    assert (m1["mean"] == [5, 18]).all()
    assert (m1["q25"] == [4.5, 12]).all()
    assert (m1["q75"] == [5.5, 22]).all()
    assert np.allclose(m1["std"], [1, 10.583005])


def test_delta_without_aggregation_fail(fake_dataset):
    tmp_df = fake_dataset._add_delta_columns(fake_dataset.merged_df)
    with pytest.raises(AssertionError):
        fake_dataset._compute_delta_by_dataset(tmp_df)


def test_delta_without_aggregation(fake_dataset):
    tmp_df = fake_dataset.merged_df.copy()
    tmp_df = fake_dataset._add_aggregate_columns(tmp_df)
    tmp_df = fake_dataset._add_delta_columns(tmp_df)
    delta = fake_dataset._compute_delta_by_dataset(tmp_df)

    assert (delta[("m0", "-", "delta_baseline")] == [0, 0, 0, 9, 18, 27]).all()
    assert (delta[("m1", "-", "delta_baseline")] == [0, 0, 0, 6, 9, 24]).all()


def test_median_delta(fake_dataset):
    agg = fake_dataset._compute_aggregations(fake_dataset.merged_df.groupby("dataset_id"))
    with_delta = fake_dataset._add_delta_columns(agg)

    assert with_delta.columns.names == ["metric", "aggregate", "delta"]
    for col in with_delta.columns:
        assert col[2] == "sample", "Missing sample column"
        assert (with_delta[col] == agg[col[:-1]]).all(), "Values not copied correctly"

    delta = fake_dataset._compute_delta_by_dataset(with_delta)

    assert (delta[("m0", "median", "delta_baseline")].squeeze() == [0, 18]).all()
    assert (delta[("m0", "q25", "delta_baseline")].squeeze() == [-1, 12.5]).all()
    assert (delta[("m0", "q75", "delta_baseline")].squeeze() == [1, 23.5]).all()
    assert (delta[("m1", "median", "delta_baseline")].squeeze() == [0, 9]).all()
    assert (delta[("m1", "q25", "delta_baseline")].squeeze() == [-1, 6.5]).all()
    assert (delta[("m1", "q75", "delta_baseline")].squeeze() == [1, 17.5]).all()

    assert (delta[("m0", "median", "norm_delta_baseline")].squeeze() == [0, 9]).all()
    assert (delta[("m0", "q25", "norm_delta_baseline")].squeeze() == [-0.5, 6.25]).all()
    assert (delta[("m0", "q75", "norm_delta_baseline")].squeeze() == [0.5, 11.75]).all()
    assert (delta[("m1", "median", "norm_delta_baseline")].squeeze() == [0, 9 / 5]).all()
    assert (delta[("m1", "q25", "norm_delta_baseline")].squeeze() == [-0.2, 1.3]).all()
    assert (delta[("m1", "q75", "norm_delta_baseline")].squeeze() == [0.2, 3.5]).all()
