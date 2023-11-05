
import os
import pytest
from pathlib import Path
from sqlalchemy import select

from pycheribenchplot.subobject.imprecise import *
from pycheribenchplot.subobject.model import *

@pytest.fixture(scope="session")
def find_scraper():
    os.environ["PATH"] = os.environ["PATH"] + ":tools/build/dwarf-scraper"


def find_member(name: str, mlist: list[StructMember]) -> StructMember:
    """
    Helper function to find a member of a structure by name.
    """
    matches = []
    for m in mlist:
        if m.name == name:
            matches.append(m)
    assert len(matches) < 2, f"Too many members named '{name}'"
    assert len(matches) == 1, f"No member '{name}' found"
    return matches[0]


def check_member(m: StructMember, nested=None, type_name="", line=None, size=None,
                 bit_size=0, offset=None, bit_offset=0, flags=StructMemberFlags.Unset,
                 array_items=0):
    """
    Helper to check a member row.
    """
    assert m.nested == nested
    assert m.type_name == type_name
    assert m.line == line
    assert m.size == size
    assert m.bit_size == bit_size
    assert m.offset == offset
    assert m.bit_offset == bit_offset
    assert m.flags == flags
    assert m.array_items == array_items


@pytest.mark.parametrize("asset_file", ["tests/assets/riscv_purecap_test_dwarf_simple"])
def test_raw_table_simple(find_scraper, fake_benchmark_factory, asset_file):
    benchmark = fake_benchmark_factory(randomize_uuid=True)
    config = ExtractImpreciseSubobjectConfig(dwarf_data_sources=[
        PathMatchSpec(path=Path(asset_file), matcher=None)])
    task = ExtractImpreciseSubobject(benchmark, None, task_config=config)

    task.run()

    # Open the database and check the raw table data
    with task.struct_layout_db.session() as session:
        result = session.scalars(select(StructType)).all()
        # At least foo and bar structures
        assert len(result) > 2, "Invalid result"

        foo = session.scalars(select(StructType).where(StructType.name == "foo")).one()
        assert Path(foo.file).name == "test_dwarf_simple.c"
        assert foo.line == 9
        assert foo.c_name == "foo"
        assert foo.size == 48
        assert foo.flags == StructTypeFlags.IsStruct

        bar = session.scalars(select(StructType).where(StructType.name == "bar")).one()
        assert Path(bar.file).name == "test_dwarf_simple.c"
        assert bar.line == 4
        assert bar.c_name == "bar"
        assert bar.size == 8
        assert bar.flags == StructTypeFlags.IsStruct

        # Now check the corresponding member types
        assert len(foo.members) == 3
        a = find_member("a", foo.members)
        assert a.nested == None
        assert a.type_name == "int"
        assert a.line == 10
        assert a.size == 4
        assert a.bit_size == 0
        assert a.offset == 0
        assert a.bit_offset == 0
        assert a.flags == StructMemberFlags.Unset
        assert a.array_items == 0

        b = find_member("b", foo.members)
        assert b.nested == None
        assert b.type_name == "char *"
        assert b.line == 11
        assert b.size == 0x10
        assert b.bit_size == 0
        assert b.offset == 0x10
        assert b.bit_offset == 0
        assert b.flags == StructMemberFlags.IsPtr
        assert b.array_items == 0

        c = find_member("c", foo.members)
        assert c.nested == bar.id
        assert c.type_name == "bar"
        assert c.line == 12
        assert c.size == 0x8
        assert c.bit_size == 0
        assert c.offset == 0x20
        assert c.bit_offset == 0
        assert c.flags == StructMemberFlags.IsStruct
        assert c.array_items == 0

        assert len(bar.members) == 2
        x = find_member("x", bar.members)
        assert x.nested == None
        assert x.type_name == "int"
        assert x.line == 5
        assert x.size == 4
        assert x.bit_size == 0
        assert x.offset == 0
        assert x.bit_offset == 0
        assert x.flags == StructMemberFlags.Unset
        assert x.array_items == 0

        y = find_member("y", bar.members)
        assert y.nested == None
        assert y.type_name == "int"
        assert y.line == 6
        assert y.size == 4
        assert y.bit_size == 0
        assert y.offset == 4
        assert y.bit_offset == 0
        assert y.flags == StructMemberFlags.Unset
        assert y.array_items == 0


@pytest.mark.parametrize("asset_file,ptr_size", [
    ("tests/assets/riscv_hybrid_test_dwarf_struct_members", 8),
    ("tests/assets/riscv_purecap_test_dwarf_struct_members", 16)])
def test_raw_table_members(find_scraper, fake_benchmark_factory, asset_file, ptr_size):
    benchmark = fake_benchmark_factory(randomize_uuid=True)
    config = ExtractImpreciseSubobjectConfig(dwarf_data_sources=[
        PathMatchSpec(path=Path(asset_file), matcher=None)])
    task = ExtractImpreciseSubobject(benchmark, None, task_config=config)

    task.run()

    # Open the database and check the raw table data
    with task.struct_layout_db.session() as session:
        foo = session.scalars(select(StructType).where(StructType.name == "foo")).one()
        bar = session.scalars(select(StructType).where(StructType.name == "bar")).one()
        baz = session.scalars(select(StructType).where(StructType.name == "baz")).one()

        assert len(foo.members) == 13
        a = find_member("a", foo.members)
        check_member(a, type_name="int", line=14, size=4, offset=0)
        b = find_member("b", foo.members)
        check_member(b, type_name="char *", line=15, size=ptr_size, offset=ptr_size,
                     flags=StructMemberFlags.IsPtr)
        c = find_member("c", foo.members)
        check_member(c, nested=bar.id, type_name="bar", line=16, size=8,
                     offset=(2 * ptr_size), flags=StructMemberFlags.IsStruct)
        d = find_member("d", foo.members)
        check_member(d, type_name="const char *", line=17, size=ptr_size,
                     offset=(3 * ptr_size), flags=StructMemberFlags.IsPtr)
        e = find_member("e", foo.members)
        check_member(e, type_name="char *const", line=18, size=ptr_size,
                     offset=(4 * ptr_size),
                     flags=StructMemberFlags.IsPtr)
        f = find_member("f", foo.members)
        check_member(f, type_name="const volatile void *", line=19, size=ptr_size,
                     offset=(5 * ptr_size), flags=StructMemberFlags.IsPtr)
        g = find_member("g", foo.members)
        check_member(g, type_name="int **", line=20, size=ptr_size,
                     offset=(6 * ptr_size), flags=StructMemberFlags.IsPtr)
        h = find_member("h", foo.members)
        check_member(h, type_name="int *[10]", line=21, size=(10 * ptr_size),
                     offset=(7 * ptr_size), flags=StructMemberFlags.IsArray,
                     array_items=10)
        i = find_member("i", foo.members)
        check_member(i, type_name="void (*)(int, int)", line=22, size=ptr_size,
                     offset=(17 * ptr_size), flags=StructMemberFlags.IsFnPtr)
        j = find_member("j", foo.members)
        check_member(j, type_name="int (*)[10]", line=23, size=ptr_size,
                     offset=(18 * ptr_size), flags=StructMemberFlags.IsPtr)
        k = find_member("k", foo.members)
        check_member(k, nested=baz.id, type_name="baz_t", line=24, size=8,
                     offset=(19 * ptr_size), flags=StructMemberFlags.IsStruct)
        l = find_member("l", foo.members)
        check_member(l, type_name="forward *", line=25, size=ptr_size,
                     offset=(20 * ptr_size), flags=StructMemberFlags.IsPtr)
        # The name for this breaks due to a bug in LLVM
        m = find_member("m", foo.members)
        check_member(m, type_name="char (*(*(*[3])(int))[4])(double)", line=26,
                     size=ptr_size, offset=(21 * ptr_size),
                     flags=StructMemberFlags.IsFnPtr)
