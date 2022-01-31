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

    df = dw.extract_struct_info()
    assert len(df) == 2

    foo = df[df.index.get_level_values("name") == "foo"]
    assert len(foo) == 1
    foo = foo.reset_index().iloc[0]
    assert foo["name"] == "foo"
    assert foo["size"] == 24
    assert Path(foo["from_path"]).name == "test_dwarf_simple.c"
    assert foo["is_anon"] == False
    assert foo["src_file"] == "test_dwarf_simple.c"
    assert foo["src_line"] == 9

    bar = df[df.index.get_level_values("name") == "bar"]
    assert len(bar) == 1
    bar = bar.reset_index().iloc[0]
    assert bar["name"] == "bar"
    assert bar["size"] == 8
    assert Path(bar["from_path"]).name == "test_dwarf_simple.c"
    assert bar["is_anon"] == False
    assert bar["src_file"] == "test_dwarf_simple.c"
    assert bar["src_line"] == 4
