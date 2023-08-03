import math
from pathlib import Path

import numpy as np
import pytest

from pycheribenchplot.core.elf.dwarf import (DWARFManager, DWARFStructLayoutModel)
from pycheribenchplot.ext import pydwarf


@pytest.fixture
def dwarf_manager(fake_simple_benchmark):
    return DWARFManager(fake_simple_benchmark)


def check_member_tflags(df_row, *args):
    """
    Helper to check structure member type flags
    """
    flags = df_row["member_flags"]
    if not args:
        assert flags == pydwarf.TypeInfoFlags.kNone, f"Invalid flags kNone != {flags}"
    check = [getattr(pydwarf.TypeInfoFlags, k).value for k in args]

    all_flags = 0
    for name, check_flag in zip(args, check):
        assert (flags & check_flag) != 0, f"Flag {name} is not set"
        all_flags |= check_flag
    complement = flags & ~all_flags
    assert complement == 0, f"Extra flags set {complement}"


def check_member(df, name):
    """
    Grab the row corresponding to a member with the given name.
    The row must be unique and it is returned as a series.
    """
    df = df.reset_index()
    result = df[df["member_name"] == name]
    assert len(result) > 0, f"No member matching member_name == {name}"
    assert len(result) == 1, f"Too many rows matching member_name == {name}"
    return result.iloc[0]


def test_register_obj(dwarf_manager):
    dw = dwarf_manager.register_object("k", "tests/assets/riscv_purecap_test_dwarf_simple")
    assert dw is not None
    assert dw.path == "tests/assets/riscv_purecap_test_dwarf_simple"


def test_extract_struct_layout(dwarf_manager, fake_session):
    """
    Check that we detect simple structure layouts correctly.
    """
    dw = dwarf_manager.register_object("k", "tests/assets/riscv_purecap_test_dwarf_simple")

    info = dw.load_type_info()
    df = dw.parse_struct_layout(info)

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    # A couple of extra entries are expected because of auxargs
    assert len(df) == 10

    foo = df.xs("foo", level="base_name").reset_index()
    assert (foo["size"] == 48).all()
    assert (foo["file"].map(lambda p: Path(p).name) == "test_dwarf_simple.c").all()
    assert (foo["line"] == 9).all()

    bar = df.xs("bar", level="base_name").reset_index()
    assert (bar["size"] == 8).all()
    assert (foo["file"].map(lambda p: Path(p).name) == "test_dwarf_simple.c").all()
    assert (bar["line"] == 4).all()


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_purecap_test_dwarf_struct_members", 16),
                                                 ("tests/assets/riscv_hybrid_test_dwarf_struct_members", 8)])
def test_extract_layout_members(fake_session, dwarf_manager, asset_file, ptr_size):
    dw = dwarf_manager.register_object("k", asset_file)

    info = dw.load_type_info()
    df = dw.parse_struct_layout(info)

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique

    foo = df.xs("foo", level="base_name")
    assert len(foo) == 13
    a = check_member(foo, "a")
    assert a["member_type_name"] == "int"
    assert a["member_size"] == 4
    b = check_member(foo, "b")
    assert b["member_type_name"] == "char *"
    assert b["member_size"] == ptr_size
    cx = check_member(foo, "c.x")
    assert cx["member_type_name"] == "int"
    assert cx["member_size"] == 4
    cy = check_member(foo, "c.y")
    assert cy["member_type_name"] == "int"
    assert cy["member_size"] == 4
    d = check_member(foo, "d")
    assert d["member_type_name"] == "char const *"
    assert d["member_size"] == ptr_size
    e = check_member(foo, "e")
    assert e["member_type_name"] == "char * const"
    assert e["member_size"] == ptr_size
    f = check_member(foo, "f")
    assert f["member_type_name"] == "void volatile const *"
    assert f["member_size"] == ptr_size
    g = check_member(foo, "g")
    assert g["member_type_name"] == "int * *"
    assert g["member_size"] == ptr_size
    h = check_member(foo, "h")
    assert h["member_type_name"] == "int * [10]"
    assert h["member_size"] == 10 * ptr_size
    i = check_member(foo, "i")
    assert i["member_type_name"] == "void(int, int) *"
    assert i["member_size"] == ptr_size
    j = check_member(foo, "j")
    assert j["member_type_name"] == "int [10] *"
    assert j["member_size"] == ptr_size
    k = check_member(foo, "k.z")
    assert k["member_type_name"] == "long"
    assert k["member_size"] == 8
    l = check_member(foo, "l")
    assert l["member_type_name"] == "struct forward *"
    assert l["member_size"] == ptr_size

    bar = df.xs("bar", level="base_name")
    assert len(bar) == 2
    x = check_member(bar, "x")
    y = check_member(bar, "y")
    assert x["member_type_name"] == "int"
    assert y["member_type_name"] == "int"

    baz = df.xs("baz", level="base_name")
    assert len(baz) == 1
    z = check_member(baz, "z")
    assert z["member_type_name"] == "long"


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_purecap_test_dwarf_struct_members_anon", 16),
                                                 ("tests/assets/riscv_hybrid_test_dwarf_struct_members_anon", 8)])
def test_extract_layout_members_anon(fake_session, dwarf_manager, asset_file, ptr_size):
    """
    Check the extraction of layouts with anonymous structures and unions.
    """
    dw = dwarf_manager.register_object("k", asset_file)

    info = dw.load_type_info()
    df = dw.parse_struct_layout(info)

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique

    foo = df.xs("foo", level="base_name")
    assert len(foo) == 5
    a = check_member(foo, "a.z")
    assert a["member_type_name"] == "long"
    assert a["member_size"] == 8

    s_a = check_member(foo, f"<anon>.{ptr_size}.s_a")
    assert s_a["member_type_name"] == "int"
    assert s_a["member_size"] == 4
    assert s_a["member_offset"] == ptr_size
    s_b = check_member(foo, f"<anon>.{ptr_size}.s_b")
    assert s_b["member_type_name"] == "char *"
    assert s_b["member_size"] == ptr_size
    assert s_b["member_offset"] == 2 * ptr_size

    un_a = check_member(foo, f"<anon>.{3 * ptr_size}.un_a")
    assert un_a["member_type_name"] == "int"
    assert un_a["member_size"] == 4
    assert un_a["member_offset"] == 3 * ptr_size
    un_b = check_member(foo, f"<anon>.{3 * ptr_size}.un_b")
    assert un_b["member_type_name"] == "char *"
    assert un_b["member_size"] == ptr_size
    assert un_b["member_offset"] == 3 * ptr_size

    bar = df.xs("bar", level="base_name")
    assert len(bar) == 2
    a = check_member(bar, "nested.a")
    assert a["member_type_name"] == "long"
    assert a["member_size"] == 8
    assert a["member_offset"] == 0
    b = check_member(bar, "nested.b")
    assert b["member_type_name"] == "char *"
    assert b["member_size"] == ptr_size
    assert b["member_offset"] == ptr_size


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_purecap_test_dwarf_nested", 16),
                                                 ("tests/assets/riscv_hybrid_test_dwarf_nested", 8)])
def test_extract_layout_members_nesting(fake_session, dwarf_manager, asset_file, ptr_size):
    """
    Check the extraction of layouts with deeply nested structures and union combinations.
    """
    dw = dwarf_manager.register_object("k", asset_file)

    info = dw.load_type_info()
    df = dw.parse_struct_layout(info)

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique

    foo = df.xs("foo", level="base_name")
    assert len(foo) == 7
    a = check_member(foo, "a")
    assert a["member_type_name"] == "int"
    assert a["member_size"] == 4
    assert a["member_offset"] == 0

    bar_x = check_member(foo, "b.b_bar.x")
    assert bar_x["member_type_name"] == "int"
    assert bar_x["member_size"] == 4
    assert bar_x["member_offset"] == 8

    bar_ux = check_member(foo, "b.b_bar.u.x")
    assert bar_ux["member_type_name"] == "int"
    assert bar_ux["member_size"] == 4
    assert bar_ux["member_offset"] == 16

    bar_uy = check_member(foo, "b.b_bar.u.y")
    assert bar_uy["member_type_name"] == "long"
    assert bar_uy["member_size"] == 8
    assert bar_uy["member_offset"] == 16

    baz_v = check_member(foo, "b.b_baz.v")
    assert baz_v["member_type_name"] == "int"
    assert baz_v["member_size"] == 4
    assert baz_v["member_offset"] == 8

    baz_ux = check_member(foo, "b.b_baz.u.x")
    assert baz_ux["member_type_name"] == "int"
    assert baz_ux["member_size"] == 4
    assert baz_ux["member_offset"] == 16

    baz_uy = check_member(foo, "b.b_baz.u.y")
    assert baz_uy["member_type_name"] == "long"
    assert baz_uy["member_size"] == 8
    assert baz_uy["member_offset"] == 16


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_purecap_test_dwarf_empty_struct", 16),
                                                 ("tests/assets/riscv_hybrid_test_dwarf_empty_struct", 8)])
def test_extract_empty_struct(fake_session, dwarf_manager, asset_file, ptr_size):
    dw = dwarf_manager.register_object("k", asset_file)

    info = dw.load_type_info()
    df = dw.parse_struct_layout(info)

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique

    bar = df.xs("bar", level="base_name")
    assert len(bar) == 1
    assert (bar.index.get_level_values("member_name") == "<empty>").all()
    assert bar["member_size"].isna().all()
    assert bar["member_type_name"].isna().all()

    foo = df.xs("foo", level="base_name")
    assert len(foo) == 2
    a = check_member(foo, "a")
    assert a["member_type_name"] == "int"
    assert a["member_size"] == 4
    assert a["member_offset"] == 0
    b = check_member(foo, "b.<empty>")
    assert b[["member_type_name"]].isna().all()
    assert b[["member_size"]].isna().all()
    assert b["member_offset"] == 4


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_purecap_test_dwarf_zero_length_array", 16),
                                                 ("tests/assets/riscv_hybrid_test_dwarf_zero_length_array", 8)])
def test_zero_length_array(fake_session, dwarf_manager, asset_file, ptr_size):
    """
    Check that zero-length array behave correctly
    """
    dw = dwarf_manager.register_object("k", asset_file)

    info = dw.load_type_info()
    df = dw.parse_struct_layout(info)

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique

    def align_up(v):
        return ((v + (ptr_size - 1)) & ~(ptr_size - 1))

    foo = df.xs("foo", level="base_name")
    assert len(foo) == 9
    why_l = check_member(foo, "why_l_")
    assert why_l["member_type_name"] == "char [0]"
    assert why_l["member_size"] == 0
    assert why_l["member_offset"] == 0
    why = check_member(foo, "why")
    assert why["member_type_name"] == "char const *"
    assert why["member_size"] == ptr_size
    assert why["member_offset"] == 0
    why_r = check_member(foo, "why_r_")
    assert why_r["member_type_name"] == "char [0]"
    assert why_r["member_size"] == 0
    assert why_r["member_offset"] == ptr_size

    nargs_l = check_member(foo, "nargs_l_")
    assert nargs_l["member_type_name"] == "char [0]"
    assert nargs_l["member_size"] == 0
    assert nargs_l["member_offset"] == ptr_size
    nargs = check_member(foo, "nargs")
    assert nargs["member_type_name"] == "int"
    assert nargs["member_size"] == 4
    assert nargs["member_offset"] == ptr_size
    nargs_r = check_member(foo, "nargs_r_")
    assert nargs_r["member_type_name"] == "char [12]"
    assert nargs_r["member_size"] == 12
    assert nargs_r["member_offset"] == 4 + ptr_size

    args_l = check_member(foo, "args_l_")
    assert args_l["member_type_name"] == "char [0]"
    assert args_l["member_size"] == 0
    assert args_l["member_offset"] == align_up(16 + ptr_size)
    args = check_member(foo, "args")
    assert args["member_type_name"] == "void * *"
    assert args["member_size"] == ptr_size
    assert args["member_offset"] == align_up(16 + ptr_size)
    args_r = check_member(foo, "args_r_")
    assert args_r["member_type_name"] == "char [0]"
    assert args_r["member_size"] == 0
    assert args_r["member_offset"] == align_up(16 + 2 * ptr_size)


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_purecap_test_dwarf_bitfield_struct", 16),
                                                 ("tests/assets/riscv_hybrid_test_dwarf_bitfield_struct", 8)])
def test_extract_bitfield_struct(fake_session, dwarf_manager, asset_file, ptr_size):
    """
    Check that bitfields are counted correctly.

    Note that the parse_struct_layout() data model does not care about bit-level resolution.
    Instead, bitfileds that belong to the same word/memeber will show up as aliased members
    of an union.
    """
    dw = dwarf_manager.register_object("k", asset_file)

    info = dw.load_type_info()
    df = dw.parse_struct_layout(info)

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique

    foo = df.xs("foo", level="base_name")
    assert len(foo) == 5

    before = check_member(foo, "before")
    assert before["member_offset"] == 0
    assert before["member_size"] == 1
    bitfield_a = check_member(foo, "bitfield_a")
    assert bitfield_a["member_offset"] == 0
    assert bitfield_a["member_size"] == 4
    bitfield_b = check_member(foo, "bitfield_b")
    assert bitfield_b["member_offset"] == 4
    assert bitfield_b["member_size"] == 4
    after = check_member(foo, "after")
    assert after["member_offset"] == 7
    assert after["member_size"] == 1
    x = check_member(foo, "x")
    assert x["member_offset"] == 8
    assert x["member_size"] == 8

    bar = df.xs("bar", level="base_name")
    assert len(bar) == 4
    before = check_member(bar, "before")
    assert before["member_offset"] == 0
    assert before["member_size"] == 4
    bitfield_a = check_member(bar, "bitfield_a")
    assert bitfield_a["member_offset"] == 4
    assert bitfield_a["member_size"] == 4
    bitfield_b = check_member(bar, "bitfield_b")
    assert bitfield_b["member_offset"] == 4
    assert bitfield_b["member_size"] == 4
    x = check_member(bar, "x")
    assert x["member_offset"] == 8
    assert x["member_size"] == 8


@pytest.mark.user_config
@pytest.mark.slow
def test_extract_kernel_struct(fake_session, dwarf_manager, benchplot_user_config):
    kernel_path = benchplot_user_config.sdk_path / "rootfs-riscv64-purecap/boot/kernel/kernel.full"
    dw = dwarf_manager.register_object("k", kernel_path)

    info = dw.load_type_info()
    df = dw.parse_struct_layout(info)

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique
