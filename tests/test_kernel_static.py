from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from pycheribenchplot.core.benchmark import BenchmarkDataSetConfig
from pycheribenchplot.core.dataset import *
from pycheribenchplot.kernel_static.dataset import *


def test_kstruct_size_dataset_derived(mocker, fake_simple_benchmark):
    """
    Test post-merge derived columns
    """
    asset_path = Path("tests/assets/test_dwarf_nested_structs.csv")
    config = BenchmarkDataSetConfig(type=DatasetName.KERNEL_STRUCT_STATS)
    # Setup mocks
    fake_get_output_path = mocker.patch.object(KernelStructSizeDataset, "output_file")
    fake_get_output_path.return_value = asset_path

    ds = KernelStructSizeDataset(fake_simple_benchmark, "__test__", config)
    ds.load()
    ds.pre_merge()
    ds.init_merge()
    ds.post_merge()

    def s(name):
        # helper for column naming
        return (name, "sample")

    df = ds.merged_df

    baz = df.xs("baz", level="name").iloc[0]
    assert baz[s("member_count")] == 3
    assert baz[s("nested_member_count")] == 3
    assert baz[s("ptr_count")] == 1
    assert baz[s("nested_ptr_count")] == 1
    assert baz[s("total_pad")] == 3
    assert baz[s("nested_total_pad")] == 3
    assert baz[s("size")] == 16
    assert baz[s("nested_packed_size")] == 13

    bar = df.xs("bar", level="name").iloc[0]
    assert bar[s("member_count")] == 3
    assert bar[s("nested_member_count")] == 6
    assert bar[s("ptr_count")] == 1
    assert bar[s("nested_ptr_count")] == 2
    assert bar[s("total_pad")] == 7
    assert bar[s("nested_total_pad")] == 10  # baz + 7
    assert bar[s("size")] == 16 + 16  # baz + 16
    assert bar[s("nested_packed_size")] == 22  # baz + 9

    foo = df.xs("foo", level="name").iloc[0]
    assert foo[s("member_count")] == 4
    assert foo[s("nested_member_count")] == 10
    assert foo[s("ptr_count")] == 1
    assert foo[s("nested_ptr_count")] == 3
    assert foo[s("total_pad")] == 4
    assert foo[s("nested_total_pad")] == 14  # bar + 4
    assert foo[s("size")] == 32 + 16  # bar + 16
    assert foo[s("nested_packed_size")] == 34  # bar + 12


@pytest.mark.asyncio
async def test_kstruct_size_dataset_gen(mocker, tmp_path, fake_simple_benchmark):
    """
    Test kernel struct stats dataset generator
    """
    asset_path = Path("tests/assets/test_dwarf_simple")
    config = BenchmarkDataSetConfig(type=DatasetName.KERNEL_STRUCT_STATS)
    # Setup mocks
    fake_get_output_path = mocker.patch.object(fake_simple_benchmark, "get_output_path")
    fake_get_output_path.return_value = tmp_path
    fake_kernel_path = mocker.patch.object(KernelStructSizeDataset, "full_kernel_path")
    fake_kernel_path.return_value = asset_path

    ds = KernelStructSizeDataset(fake_simple_benchmark, "__test__", config)
    await ds.after_extract_results()

    expected_least_columns = set(ds.input_all_columns())

    fake_get_output_path.assert_called()
    # Check that the output file was generated as expected
    out_path = tmp_path / f"__test__-{fake_simple_benchmark.uuid}.csv"
    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert expected_least_columns.issubset(set(df.columns))
    foo = df[df["name"] == "foo"]
    assert len(foo) == 3
    assert (foo["size"] == 24).all()
    assert (foo["src_line"] == 9).all()
    assert (foo["src_file"].map(lambda p: Path(p).name) == "test_dwarf_simple.c").all()

    bar = df[df["name"] == "bar"]
    assert len(bar) == 2
    assert (bar["size"] == 8).all()
    assert (bar["src_line"] == 4).all()
    assert (bar["src_file"].map(lambda p: Path(p).name) == "test_dwarf_simple.c").all()


def test_kstruct_member_pahole(fake_simple_benchmark):
    """
    Test dataset PAHOLE table generator using a simple test with only 1 structure and 2
    merged datasets.
    """
    config = BenchmarkDataSetConfig(type=DatasetName.KERNEL_STRUCT_MEMBER_STATS)
    ds = KernelStructMemberDataset(fake_simple_benchmark, "__test__", config)

    # Fill the fake merged_df dataframe
    df = pd.DataFrame({
        "name": ["my_struct"] * 2,
        "src_file": ["my/src/file.c"] * 2,
        "src_line": [1] * 2,
        "member_name": ["foo", "bar"],
        "member_size": [8, 16],
    })
    a_df = df.copy()
    a_df["dataset_id"] = "a"
    a_df["member_offset"] = [0, 8]
    a_df["member_pad"] = [0, 0]

    b_df = df.copy()
    b_df["dataset_id"] = "b"
    b_df["member_offset"] = [0, 16]
    b_df["member_pad"] = [8, 0]
    df = pd.concat([a_df, b_df], ignore_index=True, axis=0)
    df.set_index(["dataset_id", "name", "src_file", "src_line", "member_name"], inplace=True)
    ds.merged_df = df

    # Run the method under test
    result_df = ds.gen_pahole_table()

    # Check the resulting pahole structure
    assert len(result_df) == 6
    assert ("a", "my_struct", "my/src/file.c", 1, "foo.pad") in result_df.index
    assert ("b", "my_struct", "my/src/file.c", 1, "foo.pad") in result_df.index
    a = result_df.xs("a", level="dataset_id").sort_values("member_offset")
    assert a["member_size"].iloc[0] == 8
    assert np.isnan(a["member_size"].iloc[1])
    assert a["member_size"].iloc[2] == 16
    assert (a.index.get_level_values("member_name") == ["foo", "foo.pad", "bar"]).all()
    b = result_df.xs("b", level="dataset_id").sort_values("member_offset")
    assert (b["member_size"] == [8, 8, 16]).all()
    assert (b.index.get_level_values("member_name") == ["foo", "foo.pad", "bar"]).all()
