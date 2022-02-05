import pytest
import pandas as pd
import numpy as np

from pycheribenchplot.core.dataset import *

@pytest.fixture
def fake_simple_df():
    """
    Create fake dataframe with N index levels and 2 columns.
    The dataframe has an aligned index.
    """
    df = pd.DataFrame()
    N = 4  # levels
    K = 2  # keys per level
    for i in range(N):
        lvl = i + 1
        df[f"l{i}"] = np.tile(np.repeat(range(K), K**(N - lvl)), K**i)
    df.set_index([f"l{i}" for i in range(N)], inplace=True)
    df["c0"] = range(len(df))
    df["c1"] = np.random.randint(10, 100, len(df))
    # Sanity check
    assert len(df.index.names) == N
    assert df.shape == (K**N, 2)
    return df

def drop_slice(df, drop_key):
    drop_idx = df.index.values[df.index.get_locs(drop_key)]
    return df.drop(drop_idx)

def test_check_align_index(fake_simple_df):
    df = fake_simple_df
    assert check_multi_index_aligned(df, level="l0")
    assert check_multi_index_aligned(df, level=["l0", "l1"])
    assert check_multi_index_aligned(df, level=["l0", "l1", "l2"])
    assert check_multi_index_aligned(df, level=["l0", "l1", "l2", "l3"])

def test_check_align_index_unaligned(fake_simple_df):
    df = fake_simple_df
    # remove alignment on the first index level
    dropkey = (0,) * df.index.nlevels
    test_df = df.drop([dropkey])
    assert not check_multi_index_aligned(test_df, level="l0")
    assert not check_multi_index_aligned(test_df, level=["l0", "l1"])
    assert not check_multi_index_aligned(test_df, level=["l0", "l1", "l2"])
    assert check_multi_index_aligned(test_df, level=["l0", "l1", "l2", "l3"])

def test_check_align_index_missing_chunk(fake_simple_df):
    df = fake_simple_df
    drop_key = (0, 1, 1, slice(None))
    test_df = drop_slice(df, drop_key)

    assert not check_multi_index_aligned(test_df, level="l0")
    assert not check_multi_index_aligned(test_df, level=["l0", "l1"])
    assert check_multi_index_aligned(test_df, level=["l0", "l1", "l2"])
    assert check_multi_index_aligned(test_df, level=["l0", "l1", "l2", "l3"])

@pytest.mark.parametrize("drop_key", [
    (0,) * 4,  # First
    (0, 1, 1, 0),  # Mid
    (1,) * 4,  # Last
    (1, 0, slice(None), 0),  # Mid block
])
def test_align_index_last(fake_simple_df, drop_key):
    df = fake_simple_df
    align_levels = ["l0", "l1", "l2"]
    test_df = drop_slice(df, drop_key)

    assert not check_multi_index_aligned(test_df, level=align_levels)
    result_df = align_multi_index_levels(test_df, align_levels=align_levels)
    assert check_multi_index_aligned(result_df, level=align_levels)

@pytest.mark.parametrize("level", ["l0", "l1", "l3"])
def test_pivot_index_level(fake_simple_df, level):
    df = fake_simple_df

    result_df = pivot_multi_index_level(df, level)
    assert result_df.columns.nlevels == 2
    assert len(result_df.columns) == 4
    expect_cols = [("c0", 0), ("c0", 1), ("c1", 0), ("c1", 1)]
    assert len(result_df.columns.difference(expect_cols)) == 0
    assert result_df.columns.names[-1] == level

@pytest.mark.parametrize("level", ["l0", "l1", "l3"])
def test_pivot_index_level_rename(fake_simple_df, level):
    df = fake_simple_df
    rename = {0: "xxx", 1: "yyy"}

    result_df = pivot_multi_index_level(df, level, rename_map=rename)
    assert result_df.columns.nlevels == 2
    assert len(result_df.columns) == 4
    expect_cols = [("c0", "xxx"), ("c0", "yyy"), ("c1", "xxx"), ("c1", "yyy")]
    assert len(result_df.columns.difference(expect_cols)) == 0
    assert result_df.columns.names[-1] == level

@pytest.mark.parametrize("level", ["l0", "l1", "l3"])
def test_pivot_index_level_unaligned_nan(fake_simple_df, level):
    df = fake_simple_df
    test_df = drop_slice(df, (0,1,1,0))

    result_df = pivot_multi_index_level(test_df, level)
    assert check_multi_index_aligned(result_df, result_df.index.names[0])
    assert result_df.columns.nlevels == 2
    assert len(result_df.columns) == 4
    expect_cols = [("c0", 0), ("c0", 1), ("c1", 0), ("c1", 1)]
    assert len(result_df.columns.difference(expect_cols)) == 0
    assert result_df.columns.names[-1] == level
    # check that the NaN are in the right place
    assert result_df.isna().sum().sum() == 2

def test_reorder_columns(fake_simple_df):
    df = fake_simple_df

    result_df = reorder_columns(df, ["c1", "c0"])
    assert (df.iloc[:, 0] == result_df.iloc[:, 1]).all()
    assert (df.iloc[:, 1] == result_df.iloc[:, 0]).all()

