from pathlib import Path

import pandas as pd
import pytest

from pycheribenchplot.compile_db import BuildConfig
from pycheribenchplot.core.elf import DWARFManager
from pycheribenchplot.kernel_static.cheribsd_subobject import \
    CheriBSDExtractImpreciseSubobject


@pytest.fixture
def extract_task(fake_simple_benchmark):
    return CheriBSDExtractImpreciseSubobject(fake_simple_benchmark, None, BuildConfig())


@pytest.fixture
def dwarf_manager(fake_simple_benchmark):
    return DWARFManager(fake_simple_benchmark)


def test_find_unrepresentable(dwarf_manager, extract_task):
    dw = dwarf_manager.register_object("tmp", "tests/assets/riscv_purecap_test_unrepresentable_subobject")
    info = dw.load_type_info()
    df = extract_task._find_imprecise_subobjects(info)

    assert len(df) == 2

    record = df.xs("test_large_subobject", level="type_name")
    assert len(record) == 1
    record = record.reset_index().iloc[0]
    assert Path(record["file"]).name == "test_unrepresentable_subobject.c"
    # Note this refers to the struct definition
    assert record["line"] == 2
    assert record["member_offset"] == 4
    assert record["member_aligned_base"] == 0
    assert record["member_aligned_top"] == 8192

    record = df.xs("test_mixed", level="type_name")
    assert len(record) == 1
    record = record.reset_index().iloc[0]
    assert Path(record["file"]).name == "test_unrepresentable_subobject.c"
    # Note this refers to the struct definition
    assert record["line"] == 14
    assert record["member_offset"] == 33024
    assert record["member_aligned_base"] == 32768
    assert record["member_aligned_top"] == 2977792
