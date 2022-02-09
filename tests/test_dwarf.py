from pathlib import Path

import pytest

from pycheribenchplot.core.elf import DWARFHelper


class DummyBenchmark:
    def __init__(self):
        self.uuid = "dummy"


@pytest.fixture
def dummy_bench():
    return DummyBenchmark()


@pytest.fixture
def dwhelper(dummy_bench):
    return DWARFHelper(dummy_bench)


def test_register_obj(dwhelper):
    dwhelper.register_object("k", "tests/assets/test_dwarf_simple")
    dw = dwhelper.get_object("k")
    assert dw is not None
    assert dw.path == "tests/assets/test_dwarf_simple"


def test_extract_struct_info(dwhelper):
    dwhelper.register_object("k", "tests/assets/test_dwarf_simple")
    dw = dwhelper.get_object("k")

    dw.parse_dwarf()
    data = dw.get_dwarf_data()
    df = data.get_struct_info()
    assert len(df) == 5

    foo = df.xs("foo", level="name").reset_index()
    assert (foo["size"] == 24).all()
    assert (foo["from_path"].map(lambda p: p.name) == "test_dwarf_simple.c").all()
    assert (foo["src_file"].map(lambda p: p.name) == "test_dwarf_simple.c").all()
    assert (foo["src_line"] == 9).all()

    bar = df.xs("bar", level="name").reset_index()
    assert (bar["size"] == 8).all()
    assert (foo["from_path"].map(lambda p: p.name) == "test_dwarf_simple.c").all()
    assert (foo["src_file"].map(lambda p: p.name) == "test_dwarf_simple.c").all()
    assert (bar["src_line"] == 4).all()


def test_extract_struct_members(dwhelper):
    dwhelper.register_object("k", "tests/assets/test_dwarf_struct_members")
    dw = dwhelper.get_object("k")
    # This is not built for CHERI, we should do that instead?
    PTR_SIZE = 8

    dw.parse_dwarf()
    data = dw.get_dwarf_data()
    df = data.get_struct_info()
    assert df.index.is_unique
    assert len(df) == 14

    foo = df.xs("foo", level="name")
    assert len(foo) == 11
    a = foo.xs("a", level="member_name").iloc[0]
    assert a["member_type_name"] == "int"
    assert a["member_type_kind"].is_ptr == False
    assert a["member_type_size"] == 4
    b = foo.xs("b", level="member_name").iloc[0]
    assert b["member_type_name"] == "char *"
    assert b["member_type_kind"].is_ptr == True
    assert b["member_type_size"] == PTR_SIZE
    c = foo.xs("c", level="member_name").iloc[0]
    assert c["member_type_name"] == "struct bar"
    assert c["member_type_kind"].is_ptr == False
    assert c["member_type_size"] == 8
    d = foo.xs("d", level="member_name").iloc[0]
    assert d["member_type_name"] == "char const *"
    assert d["member_type_kind"].is_ptr == True
    assert d["member_type_size"] == PTR_SIZE
    e = foo.xs("e", level="member_name").iloc[0]
    assert e["member_type_name"] == "char * const"
    assert e["member_type_kind"].is_ptr == True
    assert e["member_type_size"] == PTR_SIZE
    f = foo.xs("f", level="member_name").iloc[0]
    assert f["member_type_name"] == "void const volatile *"
    assert f["member_type_kind"].is_ptr == True
    assert f["member_type_size"] == PTR_SIZE
    g = foo.xs("g", level="member_name").iloc[0]
    assert g["member_type_name"] == "int * *"
    assert g["member_type_kind"].is_ptr == True
    assert g["member_type_size"] == PTR_SIZE
    h = foo.xs("h", level="member_name").iloc[0]
    assert h["member_type_name"] == "int * [10]"
    assert h["member_type_kind"].is_ptr == True
    assert h["member_type_size"] == 10 * PTR_SIZE
    i = foo.xs("i", level="member_name").iloc[0]
    assert i["member_type_name"] == "void(int,int) *"
    assert i["member_type_kind"].is_ptr == True
    assert i["member_type_size"] == PTR_SIZE
    j = foo.xs("j", level="member_name").iloc[0]
    assert j["member_type_name"] == "int [10] *"
    assert j["member_type_kind"].is_ptr == True
    assert j["member_type_size"] == PTR_SIZE
    k = foo.xs("k", level="member_name").iloc[0]
    assert k["member_type_name"] == "baz_t"
    assert k["member_type_kind"].is_ptr == False
    assert j["member_type_size"] == 8

    bar = df.xs("bar", level="name")
    assert len(bar) == 2
    x = bar.xs("x", level="member_name").iloc[0]
    y = bar.xs("y", level="member_name").iloc[0]
    assert x["member_type_name"] == "int"
    assert y["member_type_name"] == "int"

    baz = df.xs("baz", level="name")
    assert len(baz) == 1
    z = baz.xs("z", level="member_name").iloc[0]
    assert z["member_type_name"] == "long int"


def test_extract_struct_members_anon(dwhelper):
    dwhelper.register_object("k", "tests/assets/test_dwarf_struct_members_anon")
    dw = dwhelper.get_object("k")

    dw.parse_dwarf()
    data = dw.get_dwarf_data()
    df = data.get_struct_info()
    assert df.index.is_unique
    assert len(df) == 7

    foo = df.xs("foo", level="name")
    assert len(foo) == 4
    a = foo.xs("a", level="member_name").iloc[0]
    assert a["member_type_name"] == "baz_t"
    assert a["member_type_kind"].is_ptr == False
    s_a = foo.xs("<anon>.8.s_a", level="member_name").iloc[0]
    # XXX check byte offset
    assert s_a["member_type_name"] == "int"
    assert s_a["member_type_kind"].is_ptr == False
    s_b = foo.xs("<anon>.8.s_b", level="member_name").iloc[0]
    # XXX check byte offset
    assert s_b["member_type_name"] == "char *"
    assert s_b["member_type_kind"].is_ptr == True
    u = foo.xs("<anon>.24", level="member_name").iloc[0]
    assert u["member_type_name"] == "union <anon>"
    assert u["member_type_kind"].is_ptr == False

    baz = df.xs("baz_t", level="name")
    assert len(baz) == 1
    z = baz.xs("z", level="member_name").iloc[0]
    assert z["member_type_name"] == "long int"

    bar = df.xs("bar", level="name")
    assert len(bar) == 2
    a = bar.xs("nested.a", level="member_name").iloc[0]
    assert a["member_type_name"] == "long int"
    assert a["member_type_kind"].is_ptr == False
    b = bar.xs("nested.b", level="member_name").iloc[0]
    assert b["member_type_name"] == "char *"
    assert b["member_type_kind"].is_ptr == True


def test_extract_struct_padding(dwhelper):
    dwhelper.register_object("k", "tests/assets/test_dwarf_struct_pad")
    dw = dwhelper.get_object("k")

    dw.parse_dwarf()
    data = dw.get_dwarf_data()
    df = data.get_struct_info()
    assert df.index.is_unique
    assert len(df) == 7

    foo = df.xs("foo", level="name")
    assert len(foo) == 5
    a = foo.xs("a", level="member_name").iloc[0]
    assert a["member_pad"] == 3
    b = foo.xs("b", level="member_name").iloc[0]
    assert b["member_pad"] == 0
    c = foo.xs("c", level="member_name").iloc[0]
    assert c["member_pad"] == 7
    d = foo.xs("d", level="member_name").iloc[0]
    assert d["member_pad"] == 4
    e = foo.xs("e", level="member_name").iloc[0]
    assert e["member_pad"] == 0

    bar = df.xs("bar", level="name")
    assert len(bar) == 2
    x = bar.xs("x", level="member_name").iloc[0]
    assert x["member_pad"] == 0
    y = bar.xs("y", level="member_name").iloc[0]
    assert y["member_pad"] == 4
