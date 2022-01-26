
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

    sinfo = dw.extract_struct_info(as_dict=True)
    assert len(sinfo) == 2
    keys = set(sinfo.keys())
    assert keys == {("foo", 24), ("bar", 8)}

    foo = sinfo[("foo", 24)]
    assert foo.name == "foo"
    assert foo.size == 24
    assert foo.from_path.name == "test_dwarf_simple.c"
    assert foo.is_anon == False
    assert foo.src_file == "test_dwarf_simple.c"
    assert foo.src_line == 9

    bar = sinfo[("bar", 8)]
    assert bar.name == "bar"
    assert bar.size == 8
    assert bar.from_path.name == "test_dwarf_simple.c"
    assert bar.is_anon == False
    assert bar.src_file == "test_dwarf_simple.c"
    assert bar.src_line == 4
