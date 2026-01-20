import subprocess
from pathlib import Path

import polars as pl
import pytest

from pycheribenchplot.subobject.scan_dwarf import (
    AnnotateImpreciseSubobjectLayouts,
    ExtractImpreciseConfig,
    ExtractImpreciseSubobject,
    LoadStructLayouts,
    PathMatchSpec,
)

from .util.session import helper_run_task, helper_hook_output_location, task_factory

# from pycheribenchplot.subobject.layouts_view import *
# from pycheribenchplot.subobject.stats_view import *


def check_row(row, row_expect):
    """Helper to check a dataframe row."""
    for key, expect in row_expect.items():
        assert key in row, f"Missing col {key}"
        if callable(expect):
            assert expect(row[key]), f"col {key}: failed predicate"
        else:
            assert row[key] == expect, f"col {key}: {row[key]} != {expect}"


@pytest.fixture(scope="session")
def find_scraper() -> Path:
    scraper = Path("build/dwarf-scanner/src/dwarf_scraper").resolve()
    assert scraper.exists()
    return scraper


def test_scraper_context(find_scraper, task_factory):
    """
    Check the scraper task context passed to the script
    """
    asset_file = Path("tests/assets/dwarf-scraper/riscv_dwarf_simple")
    config = ExtractImpreciseConfig(dwarf_data_sources=[PathMatchSpec(path=asset_file)])
    task = task_factory.quick_with_session(ExtractImpreciseSubobject, config)

    task.run()

    # Check that the right context is set on the script
    assert isinstance(task.script.context["dws_config"], ExtractImpreciseConfig)
    assert task.script.context["dws_config"].dwarf_scraper is None
    assert not task.script.context["dws_config"].verbose_scraper
    assert len(task.script.context["dws_config"].dwarf_data_sources) == 1
    assert task.script.context["dws_config"].dwarf_data_sources[0].path == Path(
        asset_file
    )
    assert task.script.context["dws_config"].dwarf_data_sources[0].matcher is None
    assert (
        task.benchmark.get_benchmark_data_path()
        in task.script.context["dws_database"].parents
    )


def test_scraper_load(find_scraper, task_factory, tmp_path):
    """
    Run the scraper manually and attempt to load the output using the SQLTarget
    """
    asset = Path("tests/assets/dwarf-scraper/riscv_dwarf_simple")
    database = tmp_path / f"test-{asset.name}.sqlite"
    subprocess.run(
        [find_scraper, "--database", database, "-i", asset, "flat-layout"], check=True
    )

    config = ExtractImpreciseConfig(dwarf_data_sources=[PathMatchSpec(path=asset)])
    extract_task = task_factory.quick_with_session(ExtractImpreciseSubobject, config)
    # Swap the output with a target file we control
    helper_hook_output_location(extract_task, struct_layout_db=database)

    task = task_factory.build_task(LoadStructLayouts)
    helper_run_task(task)

    df = task.struct_layouts.get()
    assert df.n_unique("id") == 3
    assert set(df["name"].unique()) == set(["Elf64C_Auxinfo", "foo", "bar"])

    foo = df.filter(pl.col("name") == "foo")
    assert foo.shape[0] == 5
    assert all(foo["line"] == 9)
    assert all(foo.select(pl.col("file").str.ends_with("simple.c"))["file"])
    assert all(foo["size"] == 48)

    bar = df.filter(pl.col("name") == "bar")
    assert bar.shape[0] == 2
    assert all(bar["line"] == 4)
    assert all(bar.select(pl.col("file").str.ends_with("simple.c"))["file"])
    assert all(bar["size"] == 8)


def test_scraper_annotate(find_scraper, task_factory, tmp_path):
    """
    Run the scraper manually and attempt to load the output using the SQLTarget
    """
    asset = Path("tests/assets/dwarf-scraper/riscv_dwarf_imprecise_simple")
    database = tmp_path / f"test-{asset.name}.sqlite"
    subprocess.run(
        [find_scraper, "--database", database, "-i", asset, "flat-layout"], check=True
    )

    config = ExtractImpreciseConfig(dwarf_data_sources=[PathMatchSpec(path=asset)])
    extract_task = task_factory.quick_with_session(ExtractImpreciseSubobject, config)
    # Swap the output with a target file we control
    helper_hook_output_location(extract_task, struct_layout_db=database)

    task = task_factory.build_task(AnnotateImpreciseSubobjectLayouts)
    helper_run_task(task)

    df = task.imprecise_layouts.get()
    assert df.n_unique("id") == 1
    assert all(df["name"].unique() == ["foo"])

    foo = df.filter(pl.col("name") == "foo").sort(by="byte_offset")
    assert foo.shape[0] == 3
    common = {
        "line": 6,
        "file": lambda x: x.endswith("/imprecise_simple.c"),
        "total_size": 0x8002,
    }
    check_row(
        foo.row(0, named=True),
        dict(
            byte_offset=0x0,
            bit_offset=0,
            byte_size=0x4000,
            bit_size=0,
            member_name="foo::hist",
            member_type="uint8_t[16384]",
            _alias_color=None,
            _aliased_by=[],
            **common,
        ),
    )
    check_row(
        foo.row(1, named=True),
        dict(
            byte_offset=0x4000,
            bit_offset=0,
            byte_size=0x2,
            bit_size=0,
            member_name="foo::histptr",
            member_type="uint16_t",
            _alias_color=None,
            _aliased_by=[0],
            **common,
        ),
    )
    check_row(
        foo.row(2, named=True),
        dict(
            byte_offset=0x4002,
            bit_offset=0,
            byte_size=0x4000,
            bit_size=0,
            member_name="foo::hash",
            member_type="uint16_t[8192]",
            _alias_color=0,
            _aliased_by=[],
            **common,
        ),
    )


# def test_scraper_annotate_flex()
