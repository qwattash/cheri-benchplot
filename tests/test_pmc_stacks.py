from unittest.mock import patch

import pandas as pd
import pytest

from pycheribenchplot.core.elf import Symbolizer
from pycheribenchplot.pmc.dataset import CallTree


@pytest.fixture
def calltree_factory():
    def mk(keys):
        def df_factory():
            index = pd.MultiIndex.from_tuples(keys, names=["first", "second"])
            return pd.DataFrame(0, index=index, columns=["test"])

        ct = CallTree(df_factory)
        return ct

    return mk


@pytest.fixture
def symbolizer(fake_simple_benchmark):
    resolver = Symbolizer(fake_simple_benchmark)

    def lookup(addr):
        return resolver.lookup_fn(addr, as_key="fake")

    return lookup


def find_node(ct, name):
    for n in ct:
        if n.sym.name == name:
            return n
    assert f"Node not found {name}"


def test_calltree_simple(calltree_factory, symbolizer):
    """
    Try to add two known paths and check the resulting tree
    """
    cc0 = [0x02, 0x01]
    cc1 = [0x03, 0x01]

    # Test calltree
    calltree = calltree_factory([])

    # Add callchains to column "test" with different indexes
    # indexes match the 2 levels ["first", "second"]
    def inserter(node, **kwargs):
        if (0, 0) in node.df.index:
            node.df.loc[(0, 0), "test"] += 1
        else:
            node.df.loc[(0, 0), "test"] = 1

    calltree.insert_callchain(cc0, symbolizer, inserter)
    calltree.insert_callchain(cc1, symbolizer, inserter)

    # Check that we have root + 3 nodes
    assert len(calltree.ct) == 4
    n1 = find_node(calltree.ct, "0x1")
    n2 = find_node(calltree.ct, "0x2")
    n3 = find_node(calltree.ct, "0x3")
    assert (n1, n2) in calltree.ct.edges
    assert (n1, n3) in calltree.ct.edges
    assert (calltree.root, n1) in calltree.ct.edges
    assert calltree.ct.number_of_edges() == 3
    assert (calltree.root.df["test"] == [2]).all()


def test_calltree_common_midpoint(calltree_factory, symbolizer):
    cc0 = [0x04, 0x03, 0x01]
    cc1 = [0x05, 0x03, 0x02]

    # Test calltree
    calltree = calltree_factory([])

    def inserter(node, **kwargs):
        if (0, 0) in node.df.index:
            node.df.loc[(0, 0), "test"] += 1
        else:
            node.df.loc[(0, 0), "test"] = 1

    calltree.insert_callchain(cc0, symbolizer, inserter)
    calltree.insert_callchain(cc1, symbolizer, inserter)
    # root + 6 nodes
    assert len(calltree.ct) == 7
    n1 = find_node(calltree.ct, "0x1")
    n2 = find_node(calltree.ct, "0x2")
    n4 = find_node(calltree.ct, "0x4")
    n5 = find_node(calltree.ct, "0x5")
    assert (calltree.root, n1) in calltree.ct.edges
    assert (calltree.root, n2) in calltree.ct.edges
    assert calltree.ct.out_degree(n1) == 1
    assert calltree.ct.out_degree(n2) == 1
    n1_3 = list(calltree.ct.succ[n1].keys())[0]
    n2_3 = list(calltree.ct.succ[n2].keys())[0]
    assert n1_3 != n2_3
    assert (n1, n1_3) in calltree.ct.edges
    assert (n2, n2_3) in calltree.ct.edges
    assert (n1_3, n4) in calltree.ct.edges
    assert (n2_3, n5) in calltree.ct.edges
    assert calltree.ct.number_of_edges() == 6
    assert (calltree.root.df["test"] == [2]).all()


def test_calltree_combine(calltree_factory, symbolizer):
    ct0 = calltree_factory([("a", 0)])
    ct1 = calltree_factory([("a", 1)])

    # Initialize calltrees
    cc00 = [0x02, 0x01]
    cc01 = [0x05, 0x01]

    def ct0_inserter(node, **kwargs):
        if ("a", 0) in node.df.index:
            node.df.loc[("a", 0), "test"] += 1
        else:
            node.df.loc[("a", 0), "test"] = 1

    ct0.insert_callchain(cc00, symbolizer, ct0_inserter)
    ct0.insert_callchain(cc01, symbolizer, ct0_inserter)
    cc10 = [0x02, 0x01]
    cc11 = [0x04, 0x03]

    def ct1_inserter(node, **kwargs):
        if ("a", 1) in node.df.index:
            node.df.loc[("a", 1), "test"] += 1
        else:
            node.df.loc[("a", 1), "test"] = 1

    ct1.insert_callchain(cc10, symbolizer, ct1_inserter)
    ct1.insert_callchain(cc11, symbolizer, ct1_inserter)

    # Merge logic
    ctm = ct0.combine([ct1])

    # Check
    expect_index = pd.MultiIndex.from_tuples([("a", 0), ("a", 1)], names=["first", "second"])
    assert len(ctm.ct) == 6
    n1 = find_node(ctm.ct, "0x1")
    n2 = find_node(ctm.ct, "0x2")
    n3 = find_node(ctm.ct, "0x3")
    n4 = find_node(ctm.ct, "0x4")
    n5 = find_node(ctm.ct, "0x5")
    assert (ctm.root, n1) in ctm.ct.edges
    assert (ctm.root, n3) in ctm.ct.edges
    assert (n1, n2) in ctm.ct.edges
    assert (n1, n5) in ctm.ct.edges
    assert (n3, n4) in ctm.ct.edges
    assert ctm.ct.number_of_edges() == 5
    assert (n1.df["test"] == pd.Series([2, 1], index=expect_index)).all()
    assert (n2.df["test"] == pd.Series([1, 1], index=expect_index)).all()
    assert (n3.df["test"] == pd.Series([0, 1], index=expect_index)).all()
    assert (n4.df["test"] == pd.Series([0, 1], index=expect_index)).all()
    assert (n5.df["test"] == pd.Series([1, 0], index=expect_index)).all()
    assert (ctm.root.df["test"] == [2, 2]).all()
    assert (n1.df["test"] == [2, 1]).all()
    assert (n2.df["test"] == [1, 1]).all()
    assert (n3.df["test"] == [0, 1]).all()
    assert (n4.df["test"] == [0, 1]).all()
    assert (n5.df["test"] == [1, 0]).all()


@pytest.mark.parametrize("inplace_map", [False, True])
def test_calltree_diff(calltree_factory, symbolizer, inplace_map):
    ct0 = calltree_factory([("a", 0), ("a", 1)])
    ct1 = calltree_factory([("b", 0), ("b", 1)])

    # Just fill some nodes
    def fake_inserter(node, **kwargs):
        node.df.loc[(0, 0), "test"] = 0

    ct0.insert_callchain([0x02, 0x01], symbolizer, fake_inserter)
    ct0.insert_callchain([0x03, 0x01], symbolizer, fake_inserter)
    ct1.insert_callchain([0x04, 0x01], symbolizer, fake_inserter)

    # Simulate merged node counts
    ct0_index = pd.MultiIndex.from_tuples([("a", 0), ("a", 1)], names=["first", "second"])
    ct0.root.df = pd.DataFrame({"test": [10, 5]}, index=ct0_index)
    n1 = find_node(ct0.ct, "0x1")
    n1.df = pd.DataFrame({"test": [10, 5]}, index=ct0_index)
    n2 = find_node(ct0.ct, "0x2")
    n2.df = pd.DataFrame({"test": [5, 3]}, index=ct0_index)
    n3 = find_node(ct0.ct, "0x3")
    n3.df = pd.DataFrame({"test": [5, 2]}, index=ct0_index)

    ct1_index = pd.MultiIndex.from_tuples([("b", 0), ("b", 1)], names=["first", "second"])
    ct1.root.df = pd.DataFrame({"test": [6, 10]}, index=ct1_index)
    n1 = find_node(ct1.ct, "0x1")
    n1.df = pd.DataFrame({"test": [6, 10]}, index=ct1_index)
    n4 = find_node(ct1.ct, "0x4")
    n4.df = pd.DataFrame({"test": [6, 10]}, index=ct1_index)

    ctc = ct0.combine([ct1])

    # Compute medians as a test that reduces the data size
    def _op(df):
        return df.groupby("first").median()

    ctd = ctc.map(_op, inplace=inplace_map)

    # Check
    expect_index = pd.Index(["a", "b"], name="first")
    assert len(ctd.ct) == 5
    n1 = find_node(ctd.ct, "0x1")
    n2 = find_node(ctd.ct, "0x2")
    n3 = find_node(ctd.ct, "0x3")
    n4 = find_node(ctd.ct, "0x4")
    assert (ctd.root, n1) in ctd.ct.edges
    assert (n1, n2) in ctd.ct.edges
    assert (n1, n3) in ctd.ct.edges
    assert (n1, n4) in ctd.ct.edges
    assert ctd.ct.number_of_edges() == 4
    assert (ctd.root.df["test"] == pd.Series([7.5, 8], index=expect_index)).all()
    assert (n1.df["test"] == pd.Series([7.5, 8], index=expect_index)).all()
    assert (n2.df["test"] == pd.Series([4, 0], index=expect_index)).all()
    assert (n3.df["test"] == pd.Series([3.5, 0], index=expect_index)).all()
    assert (n4.df["test"] == pd.Series([0, 8], index=expect_index)).all()
