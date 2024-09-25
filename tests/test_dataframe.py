#
# Test dataframe manipulation context and utilities.
#
import polars as pl
import pytest

from pycheribenchplot.core.analysis import AnalysisTask
from pycheribenchplot.core.schema import Schema


@pytest.fixture
def src_df():
    return pl.DataFrame({
        "param_0": [10, 20, 10, 20],
        "param_1": ["a", "a", "b", "b"],
        "metric_0": [90, 100, 105, 110],
        "METRIC/1": [50, 55, 60, 65]
    })


@pytest.fixture
def dummy_task(fake_session_factory):
    task = AnalysisTask(fake_session)


@pytest.fixture
def schema(test_task, src_df):
    return Schema(dummy_task, src_df)


def test_schema_init(schema):
    assert False


def test_schema_aux_param(schema):
    assert False
