from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
from test_dwarf import check_member

from pycheribenchplot.core.elf.dwarf import (DWARFManager, GraphConversionVisitor)
from pycheribenchplot.subobject.imprecise_subobject import (ExtractImpreciseSubobject, ExtractImpreciseSubobjectConfig,
                                                            ImpreciseSetboundsPlot)


@pytest.fixture
def extract_task(fake_simple_benchmark):
    return ExtractImpreciseSubobject(fake_simple_benchmark, None,
                                     ExtractImpreciseSubobjectConfig(dwarf_data_sources=[]))


@pytest.fixture
def plot_imprecise_task(fake_session):
    return ImpreciseSetboundsPlot(fake_session, None)


@pytest.fixture
def dwarf_manager(fake_simple_benchmark):
    return DWARFManager(fake_simple_benchmark)


def test_find_imprecise(dwarf_manager, extract_task):
    expect_file_path = str(Path("tests/assets/test_unrepresentable_subobject.c").absolute())
    dw = dwarf_manager.register_object("tmp", "tests/assets/riscv_purecap_test_unrepresentable_subobject")
    info = dw.load_type_info()
    g = dw.build_struct_layout_graph(info)
    extract_task._find_imprecise(g)

    # Filter nodes to only the ones relevent to us
    nodes = [n for n in g.nodes if n.file.find("test_unrepresentable_subobject") != -1]
    assert len(nodes) == 20

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

    NodeID = GraphConversionVisitor.NodeID

    # Verify node information
    large = NodeID(file=expect_file_path, line=2, base_name="test_large_subobject", member_name=None, member_offset=0)
    assert check_node(large)
    assert g.nodes[large]["type_name"] == "struct test_large_subobject"
    assert g.nodes[large]["has_imprecise"]
    check_not_imprecise(large)
    check_noalias(large)

    skew_offset = NodeID(file=expect_file_path,
                         line=2,
                         base_name="test_large_subobject",
                         member_name="skew_offset",
                         member_offset=0)
    assert check_node(skew_offset)
    assert g.nodes[skew_offset]["type_name"] == "int"
    check_not_imprecise(skew_offset)
    check_aliasing(skew_offset, 0)  # aliasing large_buffer
    large_buffer = NodeID(file=expect_file_path,
                          line=2,
                          base_name="test_large_subobject",
                          member_name="large_buffer",
                          member_offset=4)
    check_imprecise(large_buffer, 0, 0, 8192)
    check_noalias(large_buffer)

    nested = NodeID(file=expect_file_path, line=19, base_name="test_complex", member_name=None, member_offset=0)
    assert check_node(nested)
    assert g.nodes[large]["has_imprecise"]
    assert g.nodes[nested]["type_name"] == "struct test_complex"
    check_not_imprecise(nested)
    check_noalias(nested)

    before = NodeID(file=expect_file_path, line=19, base_name="test_complex", member_name="before", member_offset=0)
    check_node(before)
    assert g.nodes[before]["type_name"] == "int"
    check_not_imprecise(before)
    check_aliasing(before, [0, 1])  # aliasing inner and inner.buf_before
    inner = NodeID(file=expect_file_path, line=19, base_name="test_complex", member_name="inner", member_offset=4)
    check_node(inner)
    assert g.nodes[inner]["type_name"] == f"struct <anon>.{expect_file_path}.21"
    check_imprecise(inner, 0, 0, 16384)
    check_noalias(inner)
    inner_buf_before = NodeID(file=expect_file_path,
                              line=19,
                              base_name="test_complex",
                              member_name="buf_before",
                              member_offset=4)
    check_node(inner_buf_before)
    assert g.nodes[inner_buf_before]["type_name"] == f"char [8188]"
    check_imprecise(inner_buf_before, 1, 0, 8192)
    check_noalias(inner_buf_before)
    inner_buf_after = NodeID(file=expect_file_path,
                             line=19,
                             base_name="test_complex",
                             member_name="buf_after",
                             member_offset=8192)
    check_node(inner_buf_after)
    assert g.nodes[inner_buf_after]["type_name"] == f"char [8187]"
    check_imprecise(inner_buf_after, 2, 8192, 16384)
    check_noalias(inner_buf_after)
    after = NodeID(file=expect_file_path, line=19, base_name="test_complex", member_name="after", member_offset=16379)
    check_node(after)
    assert g.nodes[after]["type_name"] == f"char [10]"
    check_not_imprecise(after)
    check_aliasing(after, [0, 2])  # aliasing inner and inner.buf_after


def test_graph_io(dwarf_manager, extract_task, tmp_path):
    dw = dwarf_manager.register_object("tmp", "tests/assets/riscv_purecap_test_unrepresentable_subobject")
    info = dw.load_type_info()
    g = dw.build_struct_layout_graph(info)
    extract_task._find_imprecise(g)

    GraphConversionVisitor.dump(g, tmp_path / "test_dump.gml")
    g2 = GraphConversionVisitor.load(tmp_path / "test_dump.gml")
    assert g.graph["roots"] == g2.graph["roots"]
    assert nx.utils.nodes_equal(g, g2)
    assert nx.utils.edges_equal(g, g2)


def test_graph_extract_imprecise_members(dwarf_manager, extract_task, plot_imprecise_task):
    dw = dwarf_manager.register_object("tmp", "tests/assets/riscv_purecap_test_unrepresentable_subobject")
    info = dw.load_type_info()
    g = dw.build_struct_layout_graph(info)
    extract_task._find_imprecise(g)
    g.graph["dataset_gid"] = str(extract_task.benchmark.g_uuid)

    df = plot_imprecise_task._collect_imprecise_members([g])

    assert len(df) == 5
    print(df)

    large = df[df.index.get_level_values("base_name") == "test_large_subobject"]
    assert len(large) == 1
    assert (large.index.get_level_values("member_name") == "large_buffer").all()

    mixed = df[df.index.get_level_values("base_name") == "test_mixed"]
    assert len(mixed) == 1
    assert (mixed.index.get_level_values("member_name") == "buf").all()

    cplx = df[df.index.get_level_values("base_name") == "test_complex"]
    assert len(cplx) == 3
    assert set(cplx.index.get_level_values("member_name")) == {"buf_before", "buf_after", "inner"}
