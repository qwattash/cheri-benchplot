import os
import re
from pathlib import Path

import pytest
from sqlalchemy import select

from pycheribenchplot.core.config import AnalysisConfig
from pycheribenchplot.subobject.analysis import *
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
    try:
        assert len(matches) < 2, f"Too many members named '{name}'"
        assert len(matches) == 1, f"No member '{name}' found"
    except Exception as ex:
        print("Available members:")
        for m in mlist:
            print(m.name)
        raise ex
    return matches[0]


def check_member(m: StructMember,
                 nested=None,
                 type_name="",
                 type_name_m=None,
                 line=None,
                 size=None,
                 bit_size=None,
                 offset=None,
                 bit_offset=None,
                 flags=StructMemberFlags.Unset,
                 array_items=None):
    """
    Helper to check a member row.
    """
    assert m.nested == nested
    if type_name_m:
        assert re.match(type_name_m, m.type_name)
    else:
        assert m.type_name == type_name
    assert m.line == line
    assert m.size == size
    assert m.bit_size == bit_size
    assert m.offset == offset
    assert m.bit_offset == bit_offset
    assert m.flags == flags
    assert m.array_items == array_items


@pytest.fixture
def extract_imprecise_task(find_scraper, fake_benchmark_factory):
    benchmark = fake_benchmark_factory(randomize_uuid=True)
    config = ExtractImpreciseSubobjectConfig(dwarf_data_sources=[
        PathMatchSpec(path=Path("tests/assets/riscv_purecap_test_unrepresentable_subobject"), matcher=None)
    ])
    task = ExtractImpreciseSubobject(benchmark, None, task_config=config)
    return task


@pytest.fixture
def imprecise_plot_task(mocker, extract_imprecise_task):
    """
    Build a test instance of ImpreciseMembersPlot with some backing data
    """
    extract_imprecise_task.run()

    mock_deps = mocker.patch.object(ImpreciseMembersPlot, "_get_datagen_tasks")
    mock_deps.return_value = [extract_imprecise_task]

    return ImpreciseMembersPlot(extract_imprecise_task.benchmark.session, AnalysisConfig())


@pytest.fixture
def html_layouts_task(mocker, extract_imprecise_task):
    """
    Build a test instance of ImpreciseSubobjectLayouts with some backing data
    """
    extract_imprecise_task.run()

    mock_deps = mocker.patch.object(ImpreciseSubobjectLayouts, "_get_datagen_tasks")
    mock_deps.return_value = [extract_imprecise_task]

    return ImpreciseSubobjectLayouts(extract_imprecise_task.benchmark.session, AnalysisConfig())


@pytest.mark.parametrize("asset_file", ["tests/assets/riscv_purecap_test_dwarf_simple"])
def test_raw_table_simple(find_scraper, fake_benchmark_factory, asset_file):
    """
    Check the content of the struct_type table after loading a simple dwarf file
    """
    benchmark = fake_benchmark_factory(randomize_uuid=True)
    config = ExtractImpreciseSubobjectConfig(dwarf_data_sources=[PathMatchSpec(path=Path(asset_file), matcher=None)])
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
        assert foo.name == "foo"
        assert foo.size == 48
        assert foo.flags == StructTypeFlags.IsStruct

        bar = session.scalars(select(StructType).where(StructType.name == "bar")).one()
        assert Path(bar.file).name == "test_dwarf_simple.c"
        assert bar.line == 4
        assert bar.name == "bar"
        assert bar.size == 8
        assert bar.flags == StructTypeFlags.IsStruct

        # Now check the corresponding member types
        assert len(foo.members) == 3
        a = find_member("a", foo.members)
        assert a.nested == None
        assert a.type_name == "int"
        assert a.line == 10
        assert a.size == 4
        assert a.bit_size == None
        assert a.offset == 0
        assert a.bit_offset == None
        assert a.flags == StructMemberFlags.Unset
        assert a.array_items == None

        b = find_member("b", foo.members)
        assert b.nested == None
        assert b.type_name == "char *"
        assert b.line == 11
        assert b.size == 0x10
        assert b.bit_size == None
        assert b.offset == 0x10
        assert b.bit_offset == None
        assert b.flags == StructMemberFlags.IsPtr
        assert b.array_items == None

        c = find_member("c", foo.members)
        assert c.nested == bar.id
        assert c.type_name == "bar"
        assert c.line == 12
        assert c.size == 0x8
        assert c.bit_size == None
        assert c.offset == 0x20
        assert c.bit_offset == None
        assert c.flags == StructMemberFlags.IsStruct
        assert c.array_items == None

        assert len(bar.members) == 2
        x = find_member("x", bar.members)
        assert x.nested == None
        assert x.type_name == "int"
        assert x.line == 5
        assert x.size == 4
        assert x.bit_size == None
        assert x.offset == 0
        assert x.bit_offset == None
        assert x.flags == StructMemberFlags.Unset
        assert x.array_items == None

        y = find_member("y", bar.members)
        assert y.nested == None
        assert y.type_name == "int"
        assert y.line == 6
        assert y.size == 4
        assert y.bit_size == None
        assert y.offset == 4
        assert y.bit_offset == None
        assert y.flags == StructMemberFlags.Unset
        assert y.array_items == None


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_hybrid_test_dwarf_struct_members", 8),
                                                 ("tests/assets/riscv_purecap_test_dwarf_struct_members", 16)])
def test_raw_table_members(find_scraper, fake_benchmark_factory, asset_file, ptr_size):
    """
    Check the contents of the struct_member table and relationship to struct_type entries.
    """
    benchmark = fake_benchmark_factory(randomize_uuid=True)
    config = ExtractImpreciseSubobjectConfig(dwarf_data_sources=[PathMatchSpec(path=Path(asset_file), matcher=None)])
    task = ExtractImpreciseSubobject(benchmark, None, task_config=config)

    task.run()

    # Open the database and check the raw table data
    with task.struct_layout_db.session() as session:
        foo = session.scalars(select(StructType).where(StructType.name == "foo")).one()
        bar = session.scalars(select(StructType).where(StructType.name == "bar")).one()
        baz = session.scalars(select(StructType).where(StructType.name == "baz_t")).one()

        assert len(foo.members) == 12
        a = find_member("a", foo.members)
        check_member(a, type_name="int", line=14, size=4, offset=0)
        b = find_member("b", foo.members)
        check_member(b, type_name="char *", line=15, size=ptr_size, offset=ptr_size, flags=StructMemberFlags.IsPtr)
        c = find_member("c", foo.members)
        check_member(c,
                     nested=bar.id,
                     type_name="bar",
                     line=16,
                     size=8,
                     offset=(2 * ptr_size),
                     flags=StructMemberFlags.IsStruct)
        d = find_member("d", foo.members)
        check_member(d,
                     type_name="const char *",
                     line=17,
                     size=ptr_size,
                     offset=(3 * ptr_size),
                     flags=StructMemberFlags.IsPtr)
        e = find_member("e", foo.members)
        check_member(e,
                     type_name="char *const",
                     line=18,
                     size=ptr_size,
                     offset=(4 * ptr_size),
                     flags=StructMemberFlags.IsPtr)
        f = find_member("f", foo.members)
        check_member(f,
                     type_name="const volatile void *",
                     line=19,
                     size=ptr_size,
                     offset=(5 * ptr_size),
                     flags=StructMemberFlags.IsPtr)
        g = find_member("g", foo.members)
        check_member(g,
                     type_name="int **",
                     line=20,
                     size=ptr_size,
                     offset=(6 * ptr_size),
                     flags=StructMemberFlags.IsPtr)
        h = find_member("h", foo.members)
        check_member(h,
                     type_name="int *[10]",
                     line=21,
                     size=(10 * ptr_size),
                     offset=(7 * ptr_size),
                     flags=StructMemberFlags.IsArray,
                     array_items=10)
        i = find_member("i", foo.members)
        check_member(i,
                     type_name="void (*)(int, int)",
                     line=22,
                     size=ptr_size,
                     offset=(17 * ptr_size),
                     flags=StructMemberFlags.IsFnPtr)
        j = find_member("j", foo.members)
        check_member(j,
                     type_name="int (*)[10]",
                     line=23,
                     size=ptr_size,
                     offset=(18 * ptr_size),
                     flags=StructMemberFlags.IsPtr)
        k = find_member("k", foo.members)
        check_member(k,
                     nested=baz.id,
                     type_name="baz_t",
                     line=24,
                     size=8,
                     offset=(19 * ptr_size),
                     flags=StructMemberFlags.IsStruct)
        l = find_member("l", foo.members)
        check_member(l,
                     type_name="forward *",
                     line=25,
                     size=ptr_size,
                     offset=(20 * ptr_size),
                     flags=StructMemberFlags.IsPtr)


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_hybrid_test_dwarf_struct_members_anon", 8),
                                                 ("tests/assets/riscv_purecap_test_dwarf_struct_members_anon", 16)])
def test_raw_table_anon(find_scraper, fake_benchmark_factory, asset_file, ptr_size):
    """
    Verify struct_type and struct_member naming with anonymous types and members.
    """
    benchmark = fake_benchmark_factory(randomize_uuid=True)
    config = ExtractImpreciseSubobjectConfig(dwarf_data_sources=[PathMatchSpec(path=Path(asset_file), matcher=None)])
    task = ExtractImpreciseSubobject(benchmark, None, task_config=config)

    task.run()

    # Open the database and check the raw table data
    with task.struct_layout_db.session() as session:
        foo = session.scalars(select(StructType).where(StructType.name == "foo")).one()
        bar = session.scalars(select(StructType).where(StructType.name == "bar")).one()
        baz = session.scalars(select(StructType).where(StructType.name == "baz_t")).one()
        anon_s_type = session.scalars(
            select(StructType).where(StructType.name.like("<anon@%test_dwarf_struct_members_anon.c+10>"))).one()
        anon_u_type = session.scalars(
            select(StructType).where(StructType.name.like("<anon@%test_dwarf_struct_members_anon.c+14>"))).one()

        assert len(foo.members) == 3
        a = find_member("a", foo.members)
        check_member(a,
                     type_name="baz_t",
                     line=9,
                     size=8,
                     offset=0,
                     nested=baz.id,
                     flags=StructMemberFlags.IsStruct | StructMemberFlags.IsAnon)
        anon_s = find_member(f"<field@{ptr_size}>", foo.members)
        check_member(anon_s,
                     type_name_m=r"<anon@.*test_dwarf_struct_members_anon\.c\+10>",
                     nested=anon_s_type.id,
                     line=10,
                     size=2 * ptr_size,
                     offset=ptr_size,
                     flags=StructMemberFlags.IsStruct | StructMemberFlags.IsAnon)
        anon_u = find_member(f"<field@{3 * ptr_size}>", foo.members)
        check_member(anon_u,
                     type_name_m=r"<anon@.*test_dwarf_struct_members_anon\.c\+14>",
                     nested=anon_u_type.id,
                     line=14,
                     size=ptr_size,
                     offset=3 * ptr_size,
                     flags=StructMemberFlags.IsUnion | StructMemberFlags.IsAnon)


@pytest.mark.parametrize("asset_file,ptr_size", [("tests/assets/riscv_purecap_test_dwarf_nested", 16),
                                                 ("tests/assets/riscv_hybrid_test_dwarf_nested", 8)])
def test_raw_table_flattened(find_scraper, fake_benchmark_factory, asset_file, ptr_size):
    """
    Verify the generation of the flattened structure layout.
    """
    benchmark = fake_benchmark_factory(randomize_uuid=True)
    config = ExtractImpreciseSubobjectConfig(dwarf_data_sources=[PathMatchSpec(path=Path(asset_file), matcher=None)])
    task = ExtractImpreciseSubobject(benchmark, None, task_config=config)

    task.run()

    # Open the database and check the raw table data
    with task.struct_layout_db.session() as session:
        # Inspect the flattened layout member_bounds table
        foo = session.scalars(select(StructType).where(StructType.name == "foo")).one()

        foo_layout = session.scalars(
            select(MemberBounds).where(MemberBounds.owner_entry == foo)
            .order_by(MemberBounds.offset, MemberBounds.name)).all()

        assert len(foo_layout) == 12

        def check_bounds(index, name, offset, base, top):
            mb = foo_layout[index]
            assert mb.name == name
            assert mb.offset == offset
            assert mb.base == base
            assert mb.top == top

        check_bounds(0, "foo::a", 0, 0, 4)
        check_bounds(1, "foo::b", 8, 8, 24)
        check_bounds(2, "foo::b::b_bar", 8, 8, 24)
        check_bounds(3, "foo::b::b_bar::x", 8, 8, 12)
        check_bounds(4, "foo::b::b_baz", 8, 8, 24)
        check_bounds(5, "foo::b::b_baz::v", 8, 8, 12)
        check_bounds(6, "foo::b::b_bar::u", 16, 16, 24)
        check_bounds(7, "foo::b::b_bar::u::x", 16, 16, 20)
        check_bounds(8, "foo::b::b_bar::u::y", 16, 16, 24)
        check_bounds(9, "foo::b::b_baz::u", 16, 16, 24)
        check_bounds(10, "foo::b::b_baz::u::x", 16, 16, 20)
        check_bounds(11, "foo::b::b_baz::u::y", 16, 16, 24)


def test_raw_table_flattened_subobject(extract_imprecise_task):
    """
    Verify the sub-object capability generation in the flattened struct layout
    """
    extract_imprecise_task.run()

    # Open the database and check the raw table data
    with extract_imprecise_task.struct_layout_db.session() as session:
        simple_struct = session.scalars(select(StructType).where(StructType.name == "test_simple")).one()
        complex_struct = session.scalars(select(StructType).where(StructType.name == "test_complex")).one()
        age_softc_struct = session.scalars(select(StructType).where(StructType.name == "test_age_softc_layout")).one()
        nested_struct = session.scalars(select(StructType).where(StructType.name == "test_nested")).one()

        def check_bounds(mb, name, offset, base, top):
            assert mb.name == name
            assert mb.offset == offset
            assert mb.base == base
            assert mb.top == top

        simple_layout = session.scalars(
            select(MemberBounds).where(MemberBounds.owner_entry == simple_struct)
            .order_by(MemberBounds.offset, MemberBounds.name)).all()
        assert len(simple_layout) == 2
        check_bounds(simple_layout[0], "test_simple::skew_offset", 0, 0, 4)
        check_bounds(simple_layout[1], "test_simple::large_buffer", 4, 0, 8192)

        complex_layout = session.scalars(
            select(MemberBounds).where(MemberBounds.owner_entry == complex_struct)
            .order_by(MemberBounds.offset, MemberBounds.name)).all()
        assert len(complex_layout) == 5
        check_bounds(complex_layout[0], "test_complex::before", 0, 0, 4)
        check_bounds(complex_layout[1], "test_complex::inner", 4, 0, 16384)
        check_bounds(complex_layout[2], "test_complex::inner::buf_before", 4, 0, 8192)
        check_bounds(complex_layout[3], "test_complex::inner::buf_after", 8192, 8192, 16384)
        check_bounds(complex_layout[4], "test_complex::after", 16379, 16379, 16389)

        age_layout = session.scalars(
            select(MemberBounds).where(MemberBounds.owner_entry == age_softc_struct).order_by(
                MemberBounds.offset)).all()
        assert len(age_layout) == 3
        check_bounds(age_layout[1], "test_age_softc_layout::cdata", 0x250, 0x240, 0x240 + 0x6140 + 0x20)

        nested_layout = session.scalars(
            select(MemberBounds).where(MemberBounds.owner_entry == nested_struct)
            .order_by(MemberBounds.offset, MemberBounds.name)).all()
        assert len(nested_layout) == 6
        check_bounds(nested_layout[0], "test_nested::a", 0, 0, 0x4020)
        check_bounds(nested_layout[1], "test_nested::a::before", 0, 0, 4)
        check_bounds(nested_layout[2], "test_nested::a::inner", 4, 0, 0x4000)
        check_bounds(nested_layout[3], "test_nested::a::inner::buf_before", 4, 0, 0x2000)
        check_bounds(nested_layout[4], "test_nested::a::inner::buf_after", 0x2000, 0x2000, 0x4000)
        check_bounds(nested_layout[5], "test_nested::a::after", 0x3ffb, 0x3ffb, 0x4005)


def test_raw_table_alias(extract_imprecise_task):
    """
    Check that the imprecise sub-object members in the member_bounds table are correctly
    linked via the capability aliasing relationship.
    """
    extract_imprecise_task.run()

    # Open the database and check the raw table data
    with extract_imprecise_task.struct_layout_db.session() as session:
        simple_struct = session.scalars(select(StructType).where(StructType.name == "test_simple")).one()
        complex_struct = session.scalars(select(StructType).where(StructType.name == "test_complex")).one()
        nested_struct = session.scalars(select(StructType).where(StructType.name == "test_nested")).one()

        def check_entry(entry, name, offset, base, top):
            assert entry.name == name
            assert entry.offset == offset
            assert entry.base == base
            assert entry.top == top

        # Inspect alias group for simple_struct
        alias_groups = session.scalars(
            select(MemberBounds).where((MemberBounds.owner_entry == simple_struct)
                                       & MemberBounds.aliasing_with.any())).all()
        assert len(alias_groups) == 1
        entry = alias_groups[0]
        check_entry(entry, "test_simple::large_buffer", 0x4, 0, 0x2000)
        assert len(entry.aliasing_with) == 1
        assert len(entry.aliased_by) == 0
        check_entry(entry.aliasing_with[0], "test_simple::skew_offset", 0, 0, 0x4)

        reverse_entry = entry.aliasing_with[0]
        assert len(reverse_entry.aliasing_with) == 0
        assert len(reverse_entry.aliased_by) == 1
        assert reverse_entry.aliased_by[0] == entry

        # Inspect alias groups for complex_struct
        alias_groups = session.scalars(
            select(MemberBounds).where((MemberBounds.owner_entry == complex_struct)
                                       & MemberBounds.aliasing_with.any())
            .order_by(MemberBounds.offset, MemberBounds.name)).all()
        assert len(alias_groups) == 3
        entry = alias_groups[0]
        check_entry(entry, "test_complex::inner", 0x4, 0, 0x4000)
        assert len(entry.aliasing_with) == 2
        check_entry(entry.aliasing_with[0], "test_complex::before", 0, 0, 0x4)
        check_entry(entry.aliasing_with[1], "test_complex::after", 0x3ffb, 0x3ffb, 0x4005)

        entry = alias_groups[1]
        check_entry(entry, "test_complex::inner::buf_before", 0x4, 0, 0x2000)
        assert len(entry.aliasing_with) == 1
        check_entry(entry.aliasing_with[0], "test_complex::before", 0, 0, 0x4)

        entry = alias_groups[2]
        check_entry(entry, "test_complex::inner::buf_after", 0x2000, 0x2000, 0x4000)
        assert len(entry.aliasing_with) == 1
        check_entry(entry.aliasing_with[0], "test_complex::after", 0x3ffb, 0x3ffb, 0x4005)

        # Inspect alias groups for nested_struct
        alias_groups = session.scalars(
            select(MemberBounds).where((MemberBounds.owner_entry == nested_struct)
                                       & MemberBounds.aliasing_with.any())
            .order_by(MemberBounds.offset, MemberBounds.name)).all()
        assert len(alias_groups) == 3
        entry = alias_groups[0]
        check_entry(entry, "test_nested::a::inner", 0x4, 0, 0x4000)
        assert len(entry.aliasing_with) == 2
        check_entry(entry.aliasing_with[0], "test_nested::a::before", 0, 0, 0x4)
        check_entry(entry.aliasing_with[1], "test_nested::a::after", 0x3ffb, 0x3ffb, 0x4005)

        entry = alias_groups[1]
        check_entry(entry, "test_nested::a::inner::buf_before", 0x4, 0, 0x2000)
        assert len(entry.aliasing_with) == 1
        check_entry(entry.aliasing_with[0], "test_nested::a::before", 0, 0, 0x4)

        entry = alias_groups[2]
        check_entry(entry, "test_nested::a::inner::buf_after", 0x2000, 0x2000, 0x4000)
        assert len(entry.aliasing_with) == 1
        check_entry(entry.aliasing_with[0], "test_nested::a::after", 0x3ffb, 0x3ffb, 0x4005)


def test_load_imprecise(imprecise_plot_task, extract_imprecise_task):
    """
    Verify the loading process of the unrepresentable sub-object dataset.

    This checks that the ImpreciseMembersPlot task gets the correct input data.
    """
    benchmark = extract_imprecise_task.benchmark
    data = imprecise_plot_task.load_imprecise_subobjects()
    assert data is not None

    assert (data["dataset_id"] == benchmark.uuid).all()
    assert (data["dataset_gid"] == benchmark.g_uuid).all()
    assert len(data) == 11

    def check_row(name, expect_pad_base, expect_pad_top):
        r = data.filter(pl.col("member_name") == name)
        assert len(r) == 1, "Expect unique member_name"
        assert r["padding_base"][0] == expect_pad_base, "Invalid base padding"
        assert r["padding_top"][0] == expect_pad_top, "Invalid top padding"

    def check_count(type_name, cnt):
        r = data.filter(pl.col("name") == type_name)
        assert len(r) == cnt, "Invalid # of imprecise members"

    check_count("test_simple", 1)
    check_row("test_simple::large_buffer", 4, 3)

    check_count("test_complex", 3)
    check_row("test_complex::inner", 4, 5)
    check_row("test_complex::inner::buf_before", 4, 0)
    check_row("test_complex::inner::buf_after", 0, 5)

    check_count("test_nested", 4)
    check_row("test_nested::a", 0, 24)
    check_row("test_nested::a::inner", 4, 5)
    check_row("test_nested::a::inner::buf_before", 4, 0)
    check_row("test_nested::a::inner::buf_after", 0, 5)

    check_count("test_age_softc_layout", 1)
    check_row("test_age_softc_layout::cdata", 16, 16)


def test_render_imprecise_subobject_plot(imprecise_plot_task):
    """
    Check that the imprecise plots task successfully renders the test data
    """
    imprecise_plot_task.run()


def test_load_layouts(html_layouts_task, extract_imprecise_task):
    """
    Check that the ImpreciseSubobjectLayouts task generates the correct input data
    """
    benchmark = extract_imprecise_task.benchmark
    data = html_layouts_task.load_layouts()

    assert len(data) == 32

    def check_count(type_name, cnt):
        r = data.filter(pl.col("name") == type_name)
        assert len(r) == cnt, "Invalid # of imprecise members"

    assert (data["dataset_id"] == benchmark.uuid).all()
    assert (data["dataset_gid"] == benchmark.g_uuid).all()

    check_count("test_simple", 2)
    check_count("test_complex", 5)
    check_count("test_nested", 6)


def test_render_imprecise_layout_html(html_layouts_task):
    """
    Check that the structure layout HTML view rendered works on the test data
    """
    html_layouts_task.run()
