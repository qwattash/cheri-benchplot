import numpy as np
import pytest

from pycheribenchplot.core.plot.backend import Mosaic


def test_mosaic_alloc():
    m = Mosaic()

    assert m.layout.shape == (1, 0)
    m.allocate("foo", 1, 1)
    assert m.layout.shape == (1, 1)
    assert m.layout == [["foo"]]
    m.allocate("bar", 1, 1)
    assert m.layout.shape == (2, 1)
    assert (m.layout == [["foo"], ["bar"]]).all()
    m.allocate("baz", 2, 1)
    assert m.layout.shape == (4, 1)
    assert (m.layout == [["foo"], ["bar"], ["baz"], ["baz"]]).all()
    m.allocate("bob", 1, 2)
    assert m.layout.shape == (5, 2)
    assert (m.layout == [["foo", "foo"], ["bar", "bar"], ["baz", "baz"], ["baz", "baz"], ["bob", "bob"]]).all()
