import math
from pathlib import Path

import numpy as np
import pytest

from pycheribenchplot.core.dwarf import (DWARFManager, DWARFStructLayoutModel, StructLayoutGraph)
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
    result = df.xs(name, level="member_name")
    assert len(result) > 0, f"No member matching member_name == {name}"
    assert len(result) == 1, f"Too many rows matching member_name == {name}"
    return result.reset_index().iloc[0]


def test_register_obj(dwarf_manager):
    dw = dwarf_manager.register_object("k", "tests/assets/riscv_purecap_test_dwarf_simple")
    assert dw is not None
    assert dw.path == "tests/assets/riscv_purecap_test_dwarf_simple"


def test_extract_struct_layout(dwarf_manager, fake_session):
    """
    Check that we detect simple structure layouts correctly.
    """
    dw = dwarf_manager.register_object("k", "tests/assets/riscv_purecap_test_dwarf_simple")

    df = dw.build_struct_layout_table()

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)

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
    """
    Test the type layout flattening interface.
    """
    dw = dwarf_manager.register_object("k", asset_file)

    df = dw.build_struct_layout_table()

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique

    check_member(df, "foo")
    assert df.reset_index()["member_name"].str.startswith("foo").sum() == 16
    a = check_member(df, "foo.a")
    assert a["member_type_name"] == "int"
    assert a["member_size"] == 4
    b = check_member(df, "foo.b")
    assert b["member_type_name"] == "char *"
    assert b["member_size"] == ptr_size
    c = check_member(df, "foo.c")
    assert c["member_type_name"] == "struct bar"
    assert c["member_size"] == 8
    cx = check_member(df, "foo.c.x")
    assert cx["member_type_name"] == "int"
    assert cx["member_size"] == 4
    cy = check_member(df, "foo.c.y")
    assert cy["member_type_name"] == "int"
    assert cy["member_size"] == 4
    d = check_member(df, "foo.d")
    assert d["member_type_name"] == "char const *"
    assert d["member_size"] == ptr_size
    e = check_member(df, "foo.e")
    assert e["member_type_name"] == "char * const"
    assert e["member_size"] == ptr_size
    f = check_member(df, "foo.f")
    assert f["member_type_name"] == "void volatile const *"
    assert f["member_size"] == ptr_size
    g = check_member(df, "foo.g")
    assert g["member_type_name"] == "int * *"
    assert g["member_size"] == ptr_size
    h = check_member(df, "foo.h")
    assert h["member_type_name"] == "int * [10]"
    assert h["member_size"] == 10 * ptr_size
    i = check_member(df, "foo.i")
    assert i["member_type_name"] == "void(int, int) *"
    assert i["member_size"] == ptr_size
    j = check_member(df, "foo.j")
    assert j["member_type_name"] == "int [10] *"
    assert j["member_size"] == ptr_size
    k = check_member(df, "foo.k")
    assert k["member_type_name"] == "baz_t"
    assert k["member_size"] == 8
    kz = check_member(df, "foo.k.z")
    assert kz["member_type_name"] == "long"
    assert kz["member_size"] == 8
    l = check_member(df, "foo.l")
    assert l["member_type_name"] == "struct forward *"
    assert l["member_size"] == ptr_size

    check_member(df, "bar")
    assert df.reset_index()["member_name"].str.startswith("bar").sum() == 3
    x = check_member(df, "bar.x")
    y = check_member(df, "bar.y")
    assert x["member_type_name"] == "int"
    assert y["member_type_name"] == "int"

    check_member(df, "baz")
    assert df.reset_index()["member_name"].str.startswith("baz").sum() == 2
    z = check_member(df, "baz.z")
    assert z["member_type_name"] == "long"


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_purecap_test_dwarf_struct_members", 16),
                                                 ("tests/assets/riscv_hybrid_test_dwarf_struct_members", 8)])
def test_extract_layout_members_graph(fake_session, dwarf_manager, asset_file, ptr_size):
    """
    Test the conversion of the typeinfo container to a graph
    """
    fake_session.user_config.src_path = Path.cwd()
    expect_file_path = "tests/assets/test_dwarf_struct_members.c"
    dw = dwarf_manager.register_object("k", asset_file)

    layouts_graph = dw.build_struct_layout_graph()
    g = layouts_graph.layouts

    # Count only nodes we have control over
    nodes = [n for n in g.nodes if n.file.find("test_dwarf_struct_members") != -1]
    assert len(nodes) == 21

    def check_node(nid):
        if nid not in nodes:
            for n in nodes:
                print("check n=", n, "MATCH:", nid == n)
        return nid in nodes

    def pick_size(size_purecap, size_hybrid):
        if ptr_size == 16:
            return size_purecap
        else:
            return size_hybrid

    NodeID = StructLayoutGraph.NodeID

    # Verify node information
    foo_size = pick_size(0x150, 0xa8)
    foo = NodeID(file=expect_file_path, line=15, member="foo", size=foo_size, member_size=None)
    assert check_node(foo)
    assert g.nodes[foo]["base_name"] == "foo"
    assert g.nodes[foo]["type_name"] == "struct foo"
    assert g.nodes[foo]["offset"] == 0
    assert g.in_degree(foo) == 0
    assert g.out_degree(foo) == 12

    foo_a = NodeID(file=expect_file_path, line=15, member="foo.a", size=foo_size, member_size=4)
    assert check_node(foo_a)
    assert g.nodes[foo_a]["base_name"] == "foo"
    assert g.nodes[foo_a]["type_name"] == "int"
    assert g.nodes[foo_a]["offset"] == 0
    assert g.in_degree(foo_a) == 1
    assert g.out_degree(foo_a) == 0

    foo_b = NodeID(file=expect_file_path, line=15, member="foo.b", size=foo_size, member_size=ptr_size)
    assert check_node(foo_b)
    assert g.nodes[foo_b]["base_name"] == "foo"
    assert g.nodes[foo_b]["type_name"] == "char *"
    assert g.nodes[foo_b]["offset"] == ptr_size
    assert g.in_degree(foo_b) == 1
    assert g.out_degree(foo_b) == 0

    foo_c = NodeID(file=expect_file_path, line=15, member="foo.c", size=foo_size, member_size=8)
    assert check_node(foo_c)
    assert g.nodes[foo_c]["base_name"] == "foo"
    assert g.nodes[foo_c]["type_name"] == "struct bar"
    assert g.nodes[foo_c]["offset"] == 2 * ptr_size
    assert g.in_degree(foo_c) == 1
    assert g.out_degree(foo_c) == 2

    foo_cx = NodeID(file=expect_file_path, line=15, member="foo.c.x", size=foo_size, member_size=4)
    assert check_node(foo_cx)
    assert g.nodes[foo_cx]["base_name"] == "foo"
    assert g.nodes[foo_cx]["type_name"] == "int"
    assert g.nodes[foo_cx]["offset"] == 2 * ptr_size
    assert g.in_degree(foo_cx) == 1
    assert g.out_degree(foo_cx) == 0

    foo_cy = NodeID(file=expect_file_path, line=15, member="foo.c.y", size=foo_size, member_size=4)
    assert check_node(foo_cy)
    assert g.nodes[foo_cy]["base_name"] == "foo"
    assert g.nodes[foo_cy]["type_name"] == "int"
    assert g.nodes[foo_cy]["offset"] == 2 * ptr_size + 4
    assert g.in_degree(foo_cy) == 1
    assert g.out_degree(foo_cy) == 0

    foo_d = NodeID(file=expect_file_path, line=15, member="foo.d", size=foo_size, member_size=ptr_size)
    assert check_node(foo_d)
    assert g.nodes[foo_d]["base_name"] == "foo"
    assert g.nodes[foo_d]["type_name"] == "char const *"
    assert g.nodes[foo_d]["offset"] == 3 * ptr_size
    assert g.in_degree(foo_d) == 1
    assert g.out_degree(foo_d) == 0

    foo_e = NodeID(file=expect_file_path, line=15, member="foo.e", size=foo_size, member_size=ptr_size)
    assert check_node(foo_e)
    assert g.nodes[foo_e]["base_name"] == "foo"
    assert g.nodes[foo_e]["type_name"] == "char * const"
    assert g.nodes[foo_e]["offset"] == 4 * ptr_size
    assert g.in_degree(foo_e) == 1
    assert g.out_degree(foo_e) == 0

    foo_f = NodeID(file=expect_file_path, line=15, member="foo.f", size=foo_size, member_size=ptr_size)
    assert check_node(foo_f)
    assert g.nodes[foo_f]["base_name"] == "foo"
    assert g.nodes[foo_f]["type_name"] == "void volatile const *"
    assert g.nodes[foo_f]["offset"] == 5 * ptr_size
    assert g.in_degree(foo_f) == 1
    assert g.out_degree(foo_f) == 0

    foo_g = NodeID(file=expect_file_path, line=15, member="foo.g", size=foo_size, member_size=ptr_size)
    assert check_node(foo_g)
    assert g.nodes[foo_g]["base_name"] == "foo"
    assert g.nodes[foo_g]["type_name"] == "int * *"
    assert g.nodes[foo_g]["offset"] == 6 * ptr_size
    assert g.in_degree(foo_g) == 1
    assert g.out_degree(foo_g) == 0

    foo_h = NodeID(file=expect_file_path, line=15, member="foo.h", size=foo_size, member_size=10 * ptr_size)
    assert check_node(foo_h)
    assert g.nodes[foo_h]["base_name"] == "foo"
    assert g.nodes[foo_h]["type_name"] == "int * [10]"
    assert g.nodes[foo_h]["offset"] == 7 * ptr_size
    assert g.in_degree(foo_h) == 1
    assert g.out_degree(foo_h) == 0

    foo_i = NodeID(file=expect_file_path, line=15, member="foo.i", size=foo_size, member_size=ptr_size)
    assert check_node(foo_i)
    assert g.nodes[foo_i]["base_name"] == "foo"
    assert g.nodes[foo_i]["type_name"] == "void(int, int) *"
    assert g.nodes[foo_i]["offset"] == 7 * ptr_size + 10 * ptr_size
    assert g.in_degree(foo_i) == 1
    assert g.out_degree(foo_i) == 0

    foo_j = NodeID(file=expect_file_path, line=15, member="foo.j", size=foo_size, member_size=ptr_size)
    assert check_node(foo_j)
    assert g.nodes[foo_j]["base_name"] == "foo"
    assert g.nodes[foo_j]["type_name"] == "int [10] *"
    assert g.nodes[foo_j]["offset"] == 8 * ptr_size + 10 * ptr_size
    assert g.in_degree(foo_j) == 1
    assert g.out_degree(foo_j) == 0

    foo_k = NodeID(file=expect_file_path, line=15, member="foo.k", size=foo_size, member_size=8)
    assert check_node(foo_k)
    assert g.nodes[foo_k]["base_name"] == "foo"
    assert g.nodes[foo_k]["type_name"] == "baz_t"
    assert g.nodes[foo_k]["offset"] == 9 * ptr_size + 10 * ptr_size
    assert g.in_degree(foo_k) == 1
    assert g.out_degree(foo_k) == 1

    foo_kz = NodeID(file=expect_file_path, line=15, member="foo.k.z", size=foo_size, member_size=8)
    assert check_node(foo_kz)
    assert g.nodes[foo_kz]["base_name"] == "foo"
    assert g.nodes[foo_kz]["type_name"] == "long"
    assert g.nodes[foo_kz]["offset"] == 9 * ptr_size + 10 * ptr_size
    assert g.in_degree(foo_kz) == 1
    assert g.out_degree(foo_kz) == 0

    foo_l = NodeID(file=expect_file_path, line=15, member="foo.l", size=foo_size, member_size=ptr_size)
    assert check_node(foo_l)
    assert g.nodes[foo_l]["base_name"] == "foo"
    assert g.nodes[foo_l]["type_name"] == "struct forward *"
    assert g.nodes[foo_l]["offset"] == 10 * ptr_size + 10 * ptr_size
    assert g.in_degree(foo_l) == 1
    assert g.out_degree(foo_l) == 0


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_purecap_test_dwarf_struct_members_anon", 16),
                                                 ("tests/assets/riscv_hybrid_test_dwarf_struct_members_anon", 8)])
def test_extract_layout_members_anon(fake_session, dwarf_manager, asset_file, ptr_size):
    """
    Check the extraction of layouts with anonymous structures and unions.
    """
    fake_session.user_config.src_path = Path.cwd()
    expect_file_path = "tests/assets/test_dwarf_struct_members_anon.c"
    dw = dwarf_manager.register_object("k", asset_file)

    df = dw.build_struct_layout_table()

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique

    check_member(df, "foo")
    assert df.reset_index()["member_name"].str.startswith("foo").sum() == 9
    a = check_member(df, "foo.a")
    assert a["member_type_name"] == "baz_t"
    assert a["member_size"] == 8
    az = check_member(df, "foo.a.z")
    assert az["member_type_name"] == "long"
    assert az["member_size"] == 8

    a = check_member(df, f"foo.<anon>.{ptr_size}")
    assert a["member_type_name"] == f"struct <anon>.{expect_file_path}.10"
    assert a["member_size"] == 2 * ptr_size
    assert a["member_offset"] == ptr_size
    s_a = check_member(df, f"foo.<anon>.{ptr_size}.s_a")
    assert s_a["member_type_name"] == "int"
    assert s_a["member_size"] == 4
    assert s_a["member_offset"] == ptr_size
    s_b = check_member(df, f"foo.<anon>.{ptr_size}.s_b")
    assert s_b["member_type_name"] == "char *"
    assert s_b["member_size"] == ptr_size
    assert s_b["member_offset"] == 2 * ptr_size

    un = check_member(df, f"foo.<anon>.{3 * ptr_size}")
    assert un["member_type_name"] == f"union <anon>.{expect_file_path}.14"
    assert un["member_size"] == ptr_size
    assert un["member_offset"] == 3 * ptr_size
    un_a = check_member(df, f"foo.<anon>.{3 * ptr_size}.un_a")
    assert un_a["member_type_name"] == "int"
    assert un_a["member_size"] == 4
    assert un_a["member_offset"] == 3 * ptr_size
    un_b = check_member(df, f"foo.<anon>.{3 * ptr_size}.un_b")
    assert un_b["member_type_name"] == "char *"
    assert un_b["member_size"] == ptr_size
    assert un_b["member_offset"] == 3 * ptr_size

    check_member(df, "bar")
    assert df.reset_index()["member_name"].str.startswith("bar").sum() == 4
    nested = check_member(df, "bar.nested")
    assert nested["member_type_name"] == f"struct <anon>.{expect_file_path}.21"
    assert nested["member_size"] == 2 * ptr_size
    assert nested["member_offset"] == 0
    a = check_member(df, "bar.nested.a")
    assert a["member_type_name"] == "long"
    assert a["member_size"] == 8
    assert a["member_offset"] == 0
    b = check_member(df, "bar.nested.b")
    assert b["member_type_name"] == "char *"
    assert b["member_size"] == ptr_size
    assert b["member_offset"] == ptr_size


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_purecap_test_dwarf_nested", 16),
                                                 ("tests/assets/riscv_hybrid_test_dwarf_nested", 8)])
def test_extract_layout_members_nesting(fake_session, dwarf_manager, asset_file, ptr_size):
    """
    Check the extraction of layouts with deeply nested structures and union combinations.
    """
    fake_session.user_config.src_path = Path.cwd()
    expect_file_path = "tests/assets/test_dwarf_nested.c"
    dw = dwarf_manager.register_object("k", asset_file)

    df = dw.build_struct_layout_table()

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique

    check_member(df, "foo")
    assert df.reset_index()["member_name"].str.startswith("foo").sum() == 13
    a = check_member(df, "foo.a")
    assert a["member_type_name"] == "int"
    assert a["member_size"] == 4
    assert a["member_offset"] == 0

    b = check_member(df, "foo.b")
    assert b["member_type_name"] == f"union <anon>.{expect_file_path}.19"
    assert b["member_size"] == 16
    assert b["member_offset"] == 8

    bar = check_member(df, "foo.b.b_bar")
    assert bar["member_type_name"] == f"struct bar"
    assert bar["member_size"] == 16
    assert bar["member_offset"] == 8

    bar_x = check_member(df, "foo.b.b_bar.x")
    assert bar_x["member_type_name"] == "int"
    assert bar_x["member_size"] == 4
    assert bar_x["member_offset"] == 8

    bar_u = check_member(df, "foo.b.b_bar.u")
    assert bar_u["member_type_name"] == "union nested_union"
    assert bar_u["member_size"] == 8
    assert bar_u["member_offset"] == 16

    bar_ux = check_member(df, "foo.b.b_bar.u.x")
    assert bar_ux["member_type_name"] == "int"
    assert bar_ux["member_size"] == 4
    assert bar_ux["member_offset"] == 16

    bar_uy = check_member(df, "foo.b.b_bar.u.y")
    assert bar_uy["member_type_name"] == "long"
    assert bar_uy["member_size"] == 8
    assert bar_uy["member_offset"] == 16

    baz = check_member(df, "foo.b.b_baz")
    assert baz["member_type_name"] == f"struct baz"
    assert baz["member_size"] == 16
    assert baz["member_offset"] == 8

    baz_v = check_member(df, "foo.b.b_baz.v")
    assert baz_v["member_type_name"] == "int"
    assert baz_v["member_size"] == 4
    assert baz_v["member_offset"] == 8

    baz_u = check_member(df, "foo.b.b_baz.u")
    assert baz_u["member_type_name"] == "union nested_union"
    assert baz_u["member_size"] == 8
    assert baz_u["member_offset"] == 16

    baz_ux = check_member(df, "foo.b.b_baz.u.x")
    assert baz_ux["member_type_name"] == "int"
    assert baz_ux["member_size"] == 4
    assert baz_ux["member_offset"] == 16

    baz_uy = check_member(df, "foo.b.b_baz.u.y")
    assert baz_uy["member_type_name"] == "long"
    assert baz_uy["member_size"] == 8
    assert baz_uy["member_offset"] == 16


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_purecap_test_dwarf_empty_struct", 16),
                                                 ("tests/assets/riscv_hybrid_test_dwarf_empty_struct", 8)])
def test_extract_empty_struct(fake_session, dwarf_manager, asset_file, ptr_size):
    dw = dwarf_manager.register_object("k", asset_file)

    df = dw.build_struct_layout_table()

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique

    bar = check_member(df, "bar")
    assert df.reset_index()["member_name"].str.startswith("bar").sum() == 1
    assert np.isnan(bar["member_size"])
    assert bar["member_type_name"] == "struct bar"

    foo = check_member(df, "foo")
    assert df.reset_index()["member_name"].str.startswith("foo").sum() == 3
    a = check_member(df, "foo.a")
    assert a["member_type_name"] == "int"
    assert a["member_size"] == 4
    assert a["member_offset"] == 0
    b = check_member(df, "foo.b")
    assert b["member_type_name"] == "struct bar"
    assert b["member_size"] == 0
    assert b["member_offset"] == 4


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_purecap_test_dwarf_zero_length_array", 16),
                                                 ("tests/assets/riscv_hybrid_test_dwarf_zero_length_array", 8)])
def test_zero_length_array(fake_session, dwarf_manager, asset_file, ptr_size):
    """
    Check that zero-length array behave correctly
    """
    dw = dwarf_manager.register_object("k", asset_file)

    df = dw.build_struct_layout_table()

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique

    def align_up(v):
        return ((v + (ptr_size - 1)) & ~(ptr_size - 1))

    foo = df.xs("foo", level="base_name")
    check_member(df, "foo")
    assert df.reset_index()["member_name"].str.startswith("foo").sum() == 10
    why_l = check_member(df, "foo.why_l_")
    assert why_l["member_type_name"] == "char [0]"
    assert why_l["member_size"] == 0
    assert why_l["member_offset"] == 0
    why = check_member(df, "foo.why")
    assert why["member_type_name"] == "char const *"
    assert why["member_size"] == ptr_size
    assert why["member_offset"] == 0
    why_r = check_member(df, "foo.why_r_")
    assert why_r["member_type_name"] == "char [0]"
    assert why_r["member_size"] == 0
    assert why_r["member_offset"] == ptr_size

    nargs_l = check_member(df, "foo.nargs_l_")
    assert nargs_l["member_type_name"] == "char [0]"
    assert nargs_l["member_size"] == 0
    assert nargs_l["member_offset"] == ptr_size
    nargs = check_member(df, "foo.nargs")
    assert nargs["member_type_name"] == "int"
    assert nargs["member_size"] == 4
    assert nargs["member_offset"] == ptr_size
    nargs_r = check_member(df, "foo.nargs_r_")
    assert nargs_r["member_type_name"] == "char [12]"
    assert nargs_r["member_size"] == 12
    assert nargs_r["member_offset"] == 4 + ptr_size

    args_l = check_member(df, "foo.args_l_")
    assert args_l["member_type_name"] == "char [0]"
    assert args_l["member_size"] == 0
    assert args_l["member_offset"] == align_up(16 + ptr_size)
    args = check_member(df, "foo.args")
    assert args["member_type_name"] == "void * *"
    assert args["member_size"] == ptr_size
    assert args["member_offset"] == align_up(16 + ptr_size)
    args_r = check_member(df, "foo.args_r_")
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

    df = dw.build_struct_layout_table()

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique

    check_member(df, "foo")
    assert df.reset_index()["member_name"].str.startswith("foo").sum() == 6

    before = check_member(df, "foo.before")
    assert before["member_offset"] == 0
    assert before["member_size"] == 1
    bitfield_a = check_member(df, "foo.bitfield_a")
    assert bitfield_a["member_offset"] == 1
    assert bitfield_a["member_size"] == 1
    bitfield_b = check_member(df, "foo.bitfield_b")
    assert bitfield_b["member_offset"] == 4
    assert bitfield_b["member_size"] == 3
    after = check_member(df, "foo.after")
    assert after["member_offset"] == 7
    assert after["member_size"] == 1
    x = check_member(df, "foo.x")
    assert x["member_offset"] == 8
    assert x["member_size"] == 8

    bar = df.xs("bar", level="base_name")
    check_member(df, "bar")
    assert df.reset_index()["member_name"].str.startswith("bar").sum() == 5
    before = check_member(df, "bar.before")
    assert before["member_offset"] == 0
    assert before["member_size"] == 4
    bitfield_a = check_member(df, "bar.bitfield_a")
    assert bitfield_a["member_offset"] == 4
    assert bitfield_a["member_size"] == 3 / 8  # 3 bits
    bitfield_b = check_member(df, "bar.bitfield_b")
    assert bitfield_b["member_offset"] == 4 + (3 / 8)  # 3 bits past the 4-byte boundary
    assert bitfield_b["member_size"] == 4 / 8  # 4 bits
    x = check_member(df, "bar.x")
    assert x["member_offset"] == 8
    assert x["member_size"] == 8

    check_member(df, "etherip_header")
    assert df.reset_index()["member_name"].str.startswith("etherip_header").sum() == 4
    eip_resvl = check_member(df, "etherip_header.eip_resvl")
    assert eip_resvl["member_offset"] == 0
    assert eip_resvl["member_size"] == 4 / 8  # 4 bits
    eip_ver = check_member(df, "etherip_header.eip_ver")
    assert eip_ver["member_offset"] == 0 + 4 / 8  # 4 bits past the start
    assert eip_ver["member_size"] == 4 / 8  # 4 bits
    eip_resvh = check_member(df, "etherip_header.eip_resvh")
    assert eip_resvh["member_offset"] == 1
    assert eip_resvh["member_size"] == 1


@pytest.mark.user_config
@pytest.mark.slow
def test_extract_kernel_struct(fake_session, dwarf_manager, benchplot_user_config):
    kernel_path = benchplot_user_config.sdk_path / "rootfs-riscv64-purecap/boot/kernel/kernel.full"
    dw = dwarf_manager.register_object("k", kernel_path)

    df = dw.build_struct_layout_table()

    expected_schema = DWARFStructLayoutModel.as_raw_model().to_schema(fake_session)
    expected_schema.validate(df)
    assert df.index.is_unique
