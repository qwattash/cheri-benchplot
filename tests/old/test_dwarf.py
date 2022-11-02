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


def check_member_tflags(df_row, *args):
    for col in df_row.index:
        if not col.startswith("member_type_is_"):
            continue
        flag_name = col.removeprefix("member_type_")
        if flag_name in args:
            assert df_row[col] == True, f"Flag {flag_name} is not set"
        else:
            assert df_row[col] == False, f"Flag {flag_name} is set"


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
    check_member_tflags(a)
    assert a["member_type_name"] == "int"
    assert a["member_type_size"] == 4
    b = foo.xs("b", level="member_name").iloc[0]
    check_member_tflags(b, "is_ptr")
    assert b["member_type_name"] == "char *"
    assert b["member_type_size"] == PTR_SIZE
    c = foo.xs("c", level="member_name").iloc[0]
    check_member_tflags(c, "is_struct")
    assert c["member_type_name"] == "struct bar"
    assert c["member_type_size"] == 8
    d = foo.xs("d", level="member_name").iloc[0]
    check_member_tflags(d, "is_ptr")
    assert d["member_type_name"] == "char const *"
    assert d["member_type_size"] == PTR_SIZE
    e = foo.xs("e", level="member_name").iloc[0]
    check_member_tflags(e, "is_ptr", "is_const")
    assert e["member_type_name"] == "char * const"
    assert e["member_type_size"] == PTR_SIZE
    f = foo.xs("f", level="member_name").iloc[0]
    check_member_tflags(f, "is_ptr")
    assert f["member_type_name"] == "void const volatile *"
    assert f["member_type_size"] == PTR_SIZE
    g = foo.xs("g", level="member_name").iloc[0]
    check_member_tflags(g, "is_ptr")
    assert g["member_type_name"] == "int * *"
    assert g["member_type_size"] == PTR_SIZE
    h = foo.xs("h", level="member_name").iloc[0]
    check_member_tflags(h, "is_ptr", "is_array")
    assert h["member_type_name"] == "int * [10]"
    assert h["member_type_size"] == 10 * PTR_SIZE
    assert h["member_type_array_items"] == 10
    i = foo.xs("i", level="member_name").iloc[0]
    check_member_tflags(i, "is_ptr")
    assert i["member_type_name"] == "void(int,int) *"
    assert i["member_type_size"] == PTR_SIZE
    j = foo.xs("j", level="member_name").iloc[0]
    check_member_tflags(j, "is_ptr")
    assert j["member_type_name"] == "int [10] *"
    assert j["member_type_size"] == PTR_SIZE
    k = foo.xs("k", level="member_name").iloc[0]
    check_member_tflags(k, "is_struct", "is_typedef")
    assert k["member_type_name"] == "baz_t"
    assert k["member_type_size"] == 8

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
    check_member_tflags(a, "is_struct", "is_typedef")
    assert a["member_type_name"] == "baz_t"
    s_a = foo.xs("<anon>.8.s_a", level="member_name").iloc[0]
    check_member_tflags(s_a)
    # XXX check byte offset
    assert s_a["member_type_name"] == "int"
    s_b = foo.xs("<anon>.8.s_b", level="member_name").iloc[0]
    check_member_tflags(s_b, "is_ptr")
    # XXX check byte offset
    assert s_b["member_type_name"] == "char *"
    u = foo.xs("<anon>.24", level="member_name").iloc[0]
    check_member_tflags(u, "is_union")
    assert u["member_type_name"] == "union <anon>"

    baz = df.xs("baz_t", level="name")
    assert len(baz) == 1
    z = baz.xs("z", level="member_name").iloc[0]
    assert z["member_type_name"] == "long int"

    bar = df.xs("bar", level="name")
    assert len(bar) == 2
    a = bar.xs("nested.a", level="member_name").iloc[0]
    check_member_tflags(a)
    assert a["member_type_name"] == "long int"
    b = bar.xs("nested.b", level="member_name").iloc[0]
    check_member_tflags(b, "is_ptr")
    assert b["member_type_name"] == "char *"


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
    assert d["member_pad"] == 0
    e = foo.xs("e", level="member_name").iloc[0]
    assert e["member_pad"] == 0

    bar = df.xs("bar", level="name")
    assert len(bar) == 2
    x = bar.xs("x", level="member_name").iloc[0]
    assert x["member_pad"] == 0
    y = bar.xs("y", level="member_name").iloc[0]
    assert y["member_pad"] == 4


def test_extract_struct_multi_typedef(dwhelper):
    dwhelper.register_object("k", "tests/assets/test_dwarf_struct_multi_typedef")
    dw = dwhelper.get_object("k")

    dw.parse_dwarf()
    data = dw.get_dwarf_data()
    df = data.get_struct_info()
    assert df.index.is_unique
    assert len(df) == 1

    foo = df.xs("foo", level="name")
    assert len(foo) == 1
    x = foo.xs("x", level="member_name").iloc[0]
    check_member_tflags(x, "is_typedef")
    assert x["member_type_name"] == "bar_int"
    assert x["member_type_base_name"] == "int"
    assert x["member_type_alias_name"] == "int"


def test_extract_empty_struct(dwhelper):
    dwhelper.register_object("k", "tests/assets/test_dwarf_empty_struct")
    dw = dwhelper.get_object("k")

    dw.parse_dwarf()
    data = dw.get_dwarf_data()
    df = data.get_struct_info()
    assert df.index.is_unique
    assert len(df) == 3

    bar = df.xs("bar", level="name")
    assert len(bar) == 1
    assert bar.index.get_level_values("member_name").isna().all()


def test_extract_bitfield_struct(dwhelper):
    dwhelper.register_object("k", "tests/assets/test_dwarf_bitfield_struct")
    dw = dwhelper.get_object("k")

    dw.parse_dwarf()
    data = dw.get_dwarf_data()
    df = data.get_struct_info()
    assert df.index.is_unique
    assert len(df) == 9

    foo = df.xs("foo", level="name")
    assert len(foo) == 5
    # Verify offset, size and padding for bitfields
    before = foo.xs("before", level="member_name").iloc[0]
    assert before["member_offset"] == 0
    assert before["member_bit_offset"] == 0
    assert before["member_size"] == 1
    assert before["member_bit_size"] == 0
    assert before["member_pad"] == 0
    assert before["member_bit_pad"] == 0
    bitfield_a = foo.xs("bitfield_a", level="member_name").iloc[0]
    assert bitfield_a["member_offset"] == 0
    assert bitfield_a["member_bit_offset"] == 8
    assert bitfield_a["member_size"] == 1
    assert bitfield_a["member_bit_size"] == 0
    assert bitfield_a["member_pad"] == 2
    assert bitfield_a["member_bit_pad"] == 0
    bitfield_b = foo.xs("bitfield_b", level="member_name").iloc[0]
    assert bitfield_b["member_offset"] == 4
    assert bitfield_b["member_bit_offset"] == 0
    assert bitfield_b["member_size"] == 3
    assert bitfield_b["member_bit_size"] == 0
    assert bitfield_b["member_pad"] == 0
    assert bitfield_b["member_bit_pad"] == 0
    after = foo.xs("after", level="member_name").iloc[0]
    assert after["member_offset"] == 7
    assert after["member_bit_offset"] == 0
    assert after["member_size"] == 1
    assert after["member_bit_size"] == 0
    assert after["member_pad"] == 0
    assert after["member_bit_pad"] == 0
    x = foo.xs("x", level="member_name").iloc[0]
    assert x["member_offset"] == 8
    assert x["member_bit_offset"] == 0
    assert x["member_size"] == 8
    assert x["member_bit_size"] == 0
    assert x["member_pad"] == 0
    assert x["member_bit_pad"] == 0

    bar = df.xs("bar", level="name")
    assert len(bar) == 4
    before = bar.xs("before", level="member_name").iloc[0]
    assert before["member_offset"] == 0
    assert before["member_bit_offset"] == 0
    assert before["member_size"] == 4
    assert before["member_bit_size"] == 0
    assert before["member_pad"] == 0
    assert before["member_bit_pad"] == 0
    bitfield_a = bar.xs("bitfield_a", level="member_name").iloc[0]
    assert bitfield_a["member_offset"] == 4
    assert bitfield_a["member_bit_offset"] == 0
    assert bitfield_a["member_size"] == 0
    assert bitfield_a["member_bit_size"] == 3
    assert bitfield_a["member_pad"] == 0
    assert bitfield_a["member_bit_pad"] == 0
    bitfield_b = bar.xs("bitfield_b", level="member_name").iloc[0]
    assert bitfield_b["member_offset"] == 4
    assert bitfield_b["member_bit_offset"] == 3
    assert bitfield_b["member_size"] == 0
    assert bitfield_b["member_bit_size"] == 4
    assert bitfield_b["member_pad"] == 3
    assert bitfield_b["member_bit_pad"] == 1
    x = bar.xs("x", level="member_name").iloc[0]
    assert x["member_offset"] == 8
    assert x["member_bit_offset"] == 0
    assert x["member_size"] == 8
    assert x["member_bit_size"] == 0
    assert x["member_pad"] == 0
    assert x["member_bit_pad"] == 0


def test_zero_length_array(dwhelper):
    dwhelper.register_object("k", "tests/assets/test_dwarf_zero_length_array")
    dw = dwhelper.get_object("k")

    dw.parse_dwarf()
    data = dw.get_dwarf_data()
    df = data.get_struct_info()
    assert df.index.is_unique
    assert len(df) == 9

    foo = df.xs("foo", level="name")
    assert len(foo) == 9
    why_l = foo.xs("why_l_", level="member_name").iloc[0]
    check_member_tflags(why_l, "is_array")
    assert why_l["member_type_array_items"] == 0
    assert why_l["member_size"] == 0
    assert why_l["member_bit_size"] == 0
    assert why_l["member_pad"] == 0
    assert why_l["member_bit_pad"] == 0
    why = foo.xs("why", level="member_name").iloc[0]
    check_member_tflags(why, "is_ptr")
    assert why["member_size"] == 8
    assert why["member_bit_size"] == 0
    assert why["member_pad"] == 0
    assert why["member_bit_pad"] == 0
    why_r = foo.xs("why_r_", level="member_name").iloc[0]
    check_member_tflags(why_r, "is_array")
    assert why_r["member_size"] == 0
    assert why_r["member_bit_size"] == 0
    assert why_r["member_pad"] == 0
    assert why_r["member_bit_pad"] == 0

    nargs_l = foo.xs("nargs_l_", level="member_name").iloc[0]
    check_member_tflags(nargs_l, "is_array")
    assert nargs_l["member_type_array_items"] == 0
    assert nargs_l["member_size"] == 0
    assert nargs_l["member_bit_size"] == 0
    assert nargs_l["member_pad"] == 0
    assert nargs_l["member_bit_pad"] == 0
    nargs = foo.xs("nargs", level="member_name").iloc[0]
    check_member_tflags(nargs)
    assert nargs["member_size"] == 4
    assert nargs["member_bit_size"] == 0
    assert nargs["member_pad"] == 0
    assert nargs["member_bit_pad"] == 0
    nargs_r = foo.xs("nargs_r_", level="member_name").iloc[0]
    check_member_tflags(nargs_r, "is_array")
    assert nargs_r["member_size"] == 12
    assert nargs_r["member_bit_size"] == 0
    assert nargs_r["member_pad"] == 0
    assert nargs_r["member_bit_pad"] == 0

    args_l = foo.xs("args_l_", level="member_name").iloc[0]
    check_member_tflags(args_l, "is_array")
    assert args_l["member_type_array_items"] == 0
    assert args_l["member_size"] == 0
    assert args_l["member_bit_size"] == 0
    assert args_l["member_pad"] == 0
    assert args_l["member_bit_pad"] == 0
    args = foo.xs("args", level="member_name").iloc[0]
    check_member_tflags(args, "is_ptr")
    assert args["member_size"] == 8
    assert args["member_bit_size"] == 0
    assert args["member_pad"] == 0
    assert args["member_bit_pad"] == 0
    args_r = foo.xs("args_r_", level="member_name").iloc[0]
    check_member_tflags(args_r, "is_array")
    assert args_r["member_size"] == 0
    assert args_r["member_bit_size"] == 0
    assert args_r["member_pad"] == 0
    assert args_r["member_bit_pad"] == 0


@pytest.mark.user_config
@pytest.mark.slow
def test_extract_kernel_struct(dwhelper, benchplot_user_config):
    kernel_path = benchplot_user_config.sdk_path / "rootfs-riscv64-purecap/boot/kernel/kernel.full"
    dwhelper.register_object("k", kernel_path)
    dw = dwhelper.get_object("k")

    dw.parse_dwarf()
    data = dw.get_dwarf_data()
    df = data.get_struct_info(dedup=True)
    assert df.index.is_unique, df[df.index.duplicated(keep=False)]


@pytest.mark.user_config
@pytest.mark.slow
def test_extract_nodebug_kernel_struct(dwhelper, benchplot_user_config):
    kernel_path = benchplot_user_config.sdk_path / "rootfs-riscv64-purecap/boot/kernel.CHERI-QEMU-NODEBUG/kernel.full"
    dwhelper.register_object("k", kernel_path)
    dw = dwhelper.get_object("k")

    dw.parse_dwarf()
    data = dw.get_dwarf_data()
    df = data.get_struct_info(dedup=True)
    assert df.index.is_unique, df[df.index.duplicated(keep=False)]
