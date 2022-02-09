from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from pycheribenchplot.core.benchmark import BenchmarkDataSetConfig
from pycheribenchplot.core.dataset import DatasetName
from pycheribenchplot.kernel_static.dataset import *

# @patch("builtins.open", new_callable=mock_open)
# @patch("pycheribenchplot.core.benchmark.BenchmarkBase")
# @pytest.mark.skip
# def test_kstruct_size_dataset_load(fake_bench, fake_open, kstruct_size_data, fake_dataset_config):
#     # Make this realistic
#     fake_dataset_config.type = DatasetName.KERNEL_CSETBOUNDS_STATS
#     # Setup mocks
#     fake_path = Path("fake/bench/output")
#     fake_bench.get_output_path.return_value = fake_path
#     fake_open.read.return_value = kstruct_size_data

#     ds = KernelStructSizeDataset(fake_bench, "__fake__", fake_dataset_config)
#     ds.load()

#     fake_bench.get_output_path.assert_called()
#     fake_open.assert_called_once_with(fake_path, "r")

EXPECT_KSTRUCT_SIZE_COLUMNS = {
    "name", "size", "from_path", "src_file", "src_line", "member_name", "member_size", "member_offset", "member_pad",
    "member_type_name", "member_type_kind", "member_type_size", "member_type_pad", "member_type_ref_offset",
    "member_type_src_file", "member_type_src_line"
}


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

    ds = KernelStructSizeDataset(fake_simple_benchmark, "__fake__", config)
    await ds.after_extract_results()

    fake_get_output_path.assert_called()
    # Check that the output file was generated as expected
    out_path = tmp_path / f"__fake__-{fake_simple_benchmark.uuid}.csv"
    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert set(df.columns) == EXPECT_KSTRUCT_SIZE_COLUMNS
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
