from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
from test_dwarf import check_member

from pycheribenchplot.core.dwarf import DWARFManager, StructLayoutGraph
from pycheribenchplot.subobject.imprecise_subobject import (AllImpreciseMembersPlot, ExtractImpreciseSubobject,
                                                            ExtractImpreciseSubobjectConfig, RequiredSubobjectPrecision)

NodeID = StructLayoutGraph.NodeID


@pytest.fixture
def extract_task(fake_simple_benchmark):
    return ExtractImpreciseSubobject(fake_simple_benchmark, None,
                                     ExtractImpreciseSubobjectConfig(dwarf_data_sources=[]))


@pytest.fixture
def plot_imprecise_task(fake_session):
    return AllImpreciseMembersPlot(fake_session, None)


@pytest.fixture
def plot_precision_task(fake_session):
    return RequiredSubobjectPrecision(fake_session, None)


@pytest.fixture
def dwarf_manager(fake_simple_benchmark):
    return DWARFManager(fake_simple_benchmark)


@pytest.fixture
def expect_file_path():
    repo = Path.cwd().name
    expect_file_path = Path(repo) / "tests" / "assets" / "test_unrepresentable_subobject.c"
    return str(expect_file_path)


def test_find_imprecise(dwarf_manager, extract_task, expect_file_path):
    dw = dwarf_manager.register_object("tmp", "tests/assets/riscv_purecap_test_unrepresentable_subobject")
    gl = dw.build_struct_layout_graph()
    extract_task._find_imprecise(gl)

    # Shorthand to access the layouts graph
    g = gl.layouts

    # Filter nodes to only the ones relevent to us
    nodes = [n for n in g.nodes if n.file.find("test_unrepresentable_subobject") != -1]
    assert len(nodes) == 21

    def check_node(nid):
        if nid not in nodes:
            for n in nodes:
                print("check n=", n, "MATCH:", nid == n)
        return nid in nodes

    def check_noalias(nid):
        with pytest.raises(KeyError):
            g.nodes[nid]["alias_groups"]

    def check_aliasing(nid, groups: list[int] | int):
        if type(groups) == int:
            groups = [groups]
        assert set(g.nodes[nid]["alias_groups"]) == set(groups)

    def check_imprecise(nid, group: int, base: int, top: int):
        assert g.nodes[nid]["alias_group_id"] == group
        assert g.nodes[nid]["alias_aligned_base"] == base
        assert g.nodes[nid]["alias_aligned_top"] == top

    def check_not_imprecise(nid):
        with pytest.raises(KeyError):
            g.nodes[nid]["alias_group_id"]
        with pytest.raises(KeyError):
            g.nodes[nid]["alias_aligned_top"]
        with pytest.raises(KeyError):
            g.nodes[nid]["alias_aligned_base"]

    # Verify node information for the struct test_large_subobject root
    large = NodeID(file=expect_file_path, line=2, member="test_large_subobject", size=8192, member_size=None)
    assert check_node(large)
    assert g.in_degree(large) == 0
    assert g.nodes[large]["offset"] == 0
    assert g.nodes[large]["base_name"] == "test_large_subobject"
    assert g.nodes[large]["type_name"] == "struct test_large_subobject"
    assert g.nodes[large]["has_imprecise"]
    check_not_imprecise(large)
    check_noalias(large)

    # Now check the test_large_subobject members
    skew_offset = NodeID(file=expect_file_path,
                         line=2,
                         member="test_large_subobject.skew_offset",
                         size=8192,
                         member_size=4)
    assert check_node(skew_offset)
    assert g.nodes[skew_offset]["type_name"] == "int"
    assert g.nodes[skew_offset]["offset"] == 0
    check_not_imprecise(skew_offset)
    check_aliasing(skew_offset, 0)  # aliasing large_buffer
    large_buffer = NodeID(file=expect_file_path,
                          line=2,
                          member="test_large_subobject.large_buffer",
                          size=8192,
                          member_size=8185)
    check_node(large_buffer)
    assert g.nodes[large_buffer]["offset"] == 4
    check_imprecise(large_buffer, 0, 0, 8192)
    check_noalias(large_buffer)

    # Verify node information for the struct test_complex root
    nested = NodeID(file=expect_file_path, line=20, member="test_complex", size=16392, member_size=None)
    assert check_node(nested)
    assert g.nodes[nested]["has_imprecise"]
    assert g.nodes[nested]["type_name"] == "struct test_complex"
    assert g.nodes[nested]["offset"] == 0
    check_not_imprecise(nested)
    check_noalias(nested)

    # Verify node information for the test_complex members
    before = NodeID(file=expect_file_path, line=20, member="test_complex.before", size=16392, member_size=4)
    assert check_node(before)
    assert g.nodes[before]["type_name"] == "int"
    assert g.nodes[before]["offset"] == 0
    check_not_imprecise(before)
    check_aliasing(before, [0, 1])  # aliasing inner and inner.buf_before

    inner = NodeID(file=expect_file_path, line=20, member="test_complex.inner", size=16392, member_size=16375)
    assert check_node(inner)
    assert g.nodes[inner]["type_name"] == f"struct <anon>@{expect_file_path}+22"
    assert g.nodes[inner]["offset"] == 4
    check_imprecise(inner, 0, 0, 16384)
    check_noalias(inner)

    inner_buf_before = NodeID(file=expect_file_path,
                              line=20,
                              member="test_complex.inner.buf_before",
                              size=16392,
                              member_size=8188)
    assert check_node(inner_buf_before)
    assert g.nodes[inner_buf_before]["type_name"] == f"char [8188]"
    assert g.nodes[inner_buf_before]["offset"] == 4
    check_imprecise(inner_buf_before, 1, 0, 8192)
    check_noalias(inner_buf_before)

    inner_buf_after = NodeID(file=expect_file_path,
                             line=20,
                             member="test_complex.inner.buf_after",
                             size=16392,
                             member_size=8187)
    assert check_node(inner_buf_after)
    assert g.nodes[inner_buf_after]["type_name"] == f"char [8187]"
    assert g.nodes[inner_buf_after]["offset"] == 8192
    check_imprecise(inner_buf_after, 2, 8192, 16384)
    check_noalias(inner_buf_after)

    after = NodeID(file=expect_file_path, line=20, member="test_complex.after", size=16392, member_size=10)
    assert check_node(after)
    assert g.nodes[after]["type_name"] == f"char [10]"
    assert g.nodes[after]["offset"] == 16379
    check_not_imprecise(after)
    check_aliasing(after, [0, 2])  # aliasing inner and inner.buf_after

    ## Check the age_softc_layout regression
    # This uses a similar layout as the age_softc in the if_age.ko kernel module
    age_softc_layout = NodeID(file=expect_file_path,
                              line=14,
                              member="test_age_softc_layout",
                              size=25888,
                              member_size=None)
    assert check_node(age_softc_layout)
    assert g.nodes[age_softc_layout]["has_imprecise"]

    age_cdata = NodeID(file=expect_file_path,
                       line=14,
                       member="test_age_softc_layout.cdata",
                       size=25888,
                       member_size=24896)
    assert check_node(age_cdata)
    assert g.nodes[age_cdata]["offset"] == 0x250
    check_imprecise(age_cdata, 0, 0x240, 0x240 + 0x6140 + 0x20)
    check_noalias(age_cdata)

    ## Check that we properly detect the flexible array
    test_flexible = NodeID(file=expect_file_path, line=30, member="test_flexible", size=4, member_size=None)
    assert check_node(test_flexible)
    assert g.nodes[test_flexible]["type_name"] == "struct test_flexible"
    assert g.nodes[test_flexible]["has_flexarray"] == True
    assert g.nodes[test_flexible]["has_imprecise"] == False


@pytest.mark.parametrize("dump_filename", ["test_dump.json", "test_dump.json.gz"])
def test_graph_io(dwarf_manager, extract_task, tmp_path, dump_filename):
    dw = dwarf_manager.register_object("tmp", "tests/assets/riscv_purecap_test_unrepresentable_subobject")
    lg = dw.build_struct_layout_graph()
    extract_task._find_imprecise(lg)

    dump_fullpath = tmp_path / dump_filename

    lg.dump(dump_fullpath)
    lg2 = StructLayoutGraph.load(extract_task.benchmark, dump_fullpath)

    # Shorthand to access the layout graphs
    g, g2 = lg.layouts, lg2.layouts

    for n in g2.nodes:
        assert type(n) == NodeID
    assert g.graph["roots"] == g2.graph["roots"]
    assert nx.utils.nodes_equal(g, g2)
    assert nx.utils.edges_equal(g, g2)
    for node in g.nodes():
        assert g.nodes[node] == g2.nodes[node]


def test_graph_extract_imprecise_members(dwarf_manager, extract_task, plot_imprecise_task):
    dw = dwarf_manager.register_object("tmp", "tests/assets/riscv_purecap_test_unrepresentable_subobject")
    lg = dw.build_struct_layout_graph()
    extract_task._find_imprecise(lg)
    lg.layouts.graph["dataset_gid"] = str(extract_task.benchmark.g_uuid)

    df = plot_imprecise_task._collect_imprecise_members([lg])

    assert len(df) == 5

    large = df[df.index.get_level_values("base_name") == "test_large_subobject"]
    assert len(large) == 1
    assert (large.index.get_level_values("member_name") == "test_large_subobject.large_buffer").all()

    age = df[df.index.get_level_values("base_name") == "test_age_softc_layout"]
    assert len(age) == 1
    assert (age.index.get_level_values("member_name") == "test_age_softc_layout.cdata").all()

    cplx = df[df.index.get_level_values("base_name") == "test_complex"]
    assert len(cplx) == 3
    assert set(cplx.index.get_level_values("member_name")) == {
        "test_complex.inner.buf_before", "test_complex.inner.buf_after", "test_complex.inner"
    }


def test_subobject_precision(fake_simple_benchmark, plot_precision_task):
    """
    Check that the precision calculation makes sense.
    Note that we are testing the RISC-V variant
    """
    riscv_id = fake_simple_benchmark.g_uuid
    assert plot_precision_task._compute_precision(0x00000000, 0x00100000) == 1
    assert plot_precision_task._compute_precision(0x0FFFFFFF, 0x10000000) == 1
    assert plot_precision_task._compute_precision(0x00000004, 0x00001004) == 11
    assert plot_precision_task._compute_precision(0x00000FFF, 0x00002001) == 13

    for top_shift in range(12):
        top = 1 << top_shift
        assert plot_precision_task._compute_platform_precision(
            riscv_id, 0x0000, top_shift) == 12, f"Platform precision for 1 << {top_shift} does not match"
    assert plot_precision_task._compute_platform_precision(riscv_id, 0x0000, 0x1000) == 10

    assert plot_precision_task._compute_platform_precision(riscv_id, 0x0FFF, 0x1000) == 12
    assert plot_precision_task._compute_platform_precision(riscv_id, 0x0FFF, 0x2001) == 10
