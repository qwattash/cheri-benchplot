from pathlib import Path

import pandas as pd
import pytest
from test_dwarf import check_member

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
    df = extract_task._find_imprecise_subobjects(dw, info)

    assert len(df) == 4

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

    record = df.xs("test_complex", level="type_name").reset_index()
    assert len(record) == 2
    assert (record["file"].map(lambda p: Path(p).name) == "test_unrepresentable_subobject.c").all()
    # Note this refers to the struct definition
    assert (record["line"] == 19).all()
    assert (record["member_offset"] == [4, 8192]).all()
    assert (record["member_aligned_base"] == [0, 8192]).all()
    assert (record["member_aligned_top"] == [8192, 16384]).all()


def test_extract_layout(dwarf_manager, extract_task):
    dw = dwarf_manager.register_object("tmp", "tests/assets/riscv_purecap_test_unrepresentable_subobject")
    info = dw.load_type_info()
    imprecise = extract_task._find_imprecise_subobjects(dw, info)

    layout = extract_task._extract_layout(dw, info, imprecise)
    assert layout.index.is_unique

    large = layout.xs("test_large_subobject", level="base_name")
    assert len(large) == 2
    skew_offset = check_member(large, "skew_offset")
    assert skew_offset[["alias_group_id"]].isna().all()
    assert skew_offset[["alias_aligned_base"]].isna().all()
    assert skew_offset[["alias_aligned_top"]].isna().all()
    assert skew_offset["alias_groups"] == [0]
    buf = check_member(large, "large_buffer")
    assert buf["alias_group_id"] == 0
    assert buf["alias_aligned_base"] == 0
    assert buf["alias_aligned_top"] == 8192
    assert buf["alias_groups"] == None

    nested = layout.xs("test_complex", level="base_name")
    assert len(nested) == 4
    assert (nested["size"] == 16420).all()
    before = check_member(nested, "before")
    assert before[["alias_group_id"]].isna().all()
    assert before[["alias_aligned_base"]].isna().all()
    assert before[["alias_aligned_top"]].isna().all()
    assert before["alias_groups"] == [0]
    buf = check_member(nested, "inner.buf_before")
    assert buf["alias_group_id"] == 0
    assert buf["alias_aligned_base"] == 0
    assert buf["alias_aligned_top"] == 8192
    assert buf["alias_groups"] == None
    buf = check_member(nested, "inner.buf_after")
    assert buf["alias_group_id"] == 1
    assert buf["alias_aligned_base"] == 8192
    assert buf["alias_aligned_top"] == 16384
    assert buf["alias_groups"] == None
    after = check_member(nested, "after")
    assert after[["alias_group_id"]].isna().all()
    assert after[["alias_aligned_base"]].isna().all()
    assert after[["alias_aligned_top"]].isna().all()
    assert after["alias_groups"] == [1]
