import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pycheribenchplot.core.artefact import Target
from pycheribenchplot.core.plot_grid import (DisplayGrid, DisplayGridConfig, PlotGrid, PlotGridConfig)


def test_1x1_grid(mocker):
    fake_target = mocker.MagicMock(Target)
    fake_target.paths.return_value = []
    config = PlotGridConfig()
    df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [1.0, 1.1, 1.2, 1.3]})

    plot_callback = {"ncalls": 0}

    def tile_callback(tile, data):
        plot_callback["ncalls"] += 1
        assert tile.ax is not None
        assert tile.row is None
        assert tile.col is None
        assert_frame_equal(data, df)

    with PlotGrid(fake_target, df, config) as g:
        g.map(tile_callback)
    assert plot_callback["ncalls"] == 1


def test_3x2_grid(mocker):
    fake_target = mocker.MagicMock(Target)
    fake_target.paths.return_value = []
    config = PlotGridConfig(tile_row="x", tile_col="y")
    df = pl.DataFrame({
        "x": [0, 0, 0, 1, 1, 1],
        "y": ["a", "b", "c", "a", "b", "c"],
        "z": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    })

    expect_calls = {
        (0, "a"): pl.DataFrame({
            "x": [0],
            "y": ["a"],
            "z": 1.0
        }),
        (0, "b"): pl.DataFrame({
            "x": [0],
            "y": ["b"],
            "z": 1.1
        }),
        (0, "c"): pl.DataFrame({
            "x": [0],
            "y": ["c"],
            "z": 1.2
        }),
        (1, "a"): pl.DataFrame({
            "x": [1],
            "y": ["a"],
            "z": 1.3
        }),
        (1, "b"): pl.DataFrame({
            "x": [1],
            "y": ["b"],
            "z": 1.4
        }),
        (1, "c"): pl.DataFrame({
            "x": [1],
            "y": ["c"],
            "z": 1.5
        }),
    }

    def tile_callback(tile, data):
        expected = expect_calls.pop((tile.row, tile.col))
        assert tile.ax is not None
        assert_frame_equal(data, expected)

    with PlotGrid(fake_target, df, config) as g:
        g.map(tile_callback)
    assert len(expect_calls) == 0


def test_display_grid_rename(mocker):
    fake_target = mocker.MagicMock(Target)
    fake_target.paths.return_value = []
    config = DisplayGridConfig(tile_row="x",
                               tile_col="y",
                               param_names={
                                   "x": "X",
                                   "z": "Z"
                               },
                               param_values={
                                   "x": {
                                       0: "zero",
                                       1: "one"
                                   },
                                   "y": {
                                       "a": "A",
                                       "b": "B"
                                   }
                               })
    df = pl.DataFrame({
        "x": [0, 0, 0, 1, 1, 1],
        "y": ["a", "b", "c", "a", "b", "c"],
        "z": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    })

    expect_calls = {
        ("zero", "A"): pl.DataFrame({
            "X": ["zero"],
            "_r_x": [0],
            "y": ["A"],
            "_r_y": ["a"],
            "Z": 1.0
        }),
        ("zero", "B"): pl.DataFrame({
            "X": ["zero"],
            "_r_x": [0],
            "y": ["B"],
            "_r_y": ["b"],
            "Z": 1.1
        }),
        ("zero", "c"): pl.DataFrame({
            "X": ["zero"],
            "_r_x": [0],
            "y": ["c"],
            "_r_y": ["c"],
            "Z": 1.2
        }),
        ("one", "A"): pl.DataFrame({
            "X": ["one"],
            "_r_x": [1],
            "y": ["A"],
            "_r_y": ["a"],
            "Z": 1.3
        }),
        ("one", "B"): pl.DataFrame({
            "X": ["one"],
            "_r_x": [1],
            "y": ["B"],
            "_r_y": ["b"],
            "Z": 1.4
        }),
        ("one", "c"): pl.DataFrame({
            "X": ["one"],
            "_r_x": [1],
            "y": ["c"],
            "_r_y": ["c"],
            "Z": 1.5
        }),
    }

    def tile_callback(tile, data):
        expected = expect_calls.pop((tile.row, tile.col))
        assert tile.ax is not None
        assert_frame_equal(data, expected, check_column_order=False)

    with DisplayGrid(fake_target, df, config) as g:
        g.map(tile_callback)
    assert len(expect_calls) == 0
