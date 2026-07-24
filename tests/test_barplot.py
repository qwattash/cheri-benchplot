from unittest.mock import MagicMock
import pytest
import polars as pl
from pycheribenchplot.core.plot_grid.barplot import BarPlotConfig, grid_barplot
from pycheribenchplot.core.plot_grid.plot_grid import PlotTile


@pytest.fixture
def mock_tile():
    tile = MagicMock(spec=PlotTile)
    tile.ax = MagicMock()
    # Mock a simple palette mapping
    tile.palette = {"H1": "blue", "H2": "orange"}
    # Mock hue
    tile.hue = "hue"
    # Mock ref_to_col
    tile.ref_to_col = MagicMock(side_effect=lambda x: x.strip("<>") if x else None)
    return tile


def test_grid_barplot_labels_disabled(mock_tile):
    df = pl.DataFrame(
        {"category": ["A", "B"], "hue": ["H1", "H2"], "val": [10.0, 20.0]}
    )
    config = BarPlotConfig(show_bar_label=False)

    # Run barplot
    grid_barplot(mock_tile, df, config, x="<category>", y="val")

    # Since show_bar_label is False, tile.ax.annotate should not be called
    mock_tile.ax.annotate.assert_not_called()


def test_grid_barplot_labels_enabled_x_orient(mock_tile):
    df = pl.DataFrame(
        {"category": ["A", "B"], "hue": ["H1", "H1"], "val": [10.0, 20.5]}
    )
    config = BarPlotConfig(show_bar_label=True, bar_label_orient="h", bar_label_size=12)

    grid_barplot(mock_tile, df, config, x="<category>", y="val")

    # There are two bars, so annotate should be called twice
    assert mock_tile.ax.annotate.call_count == 2

    # Inspect first call
    # xy should be (coord, val + stack)
    # Since there's no stack, stack is 0.
    # __gen_coord for first row (A) is 0.0, offset is 0.0. So coord is 0.0.
    # Thus, xy should be (0.0, 10.0)
    # xytext should be (0, 3)
    # label should be "10"
    first_call = mock_tile.ax.annotate.call_args_list[0]
    args, kwargs = first_call
    assert args[0] == "10"
    assert kwargs["xy"] == (0.0, 10.0)
    assert kwargs["xytext"] == (0, 3)
    assert kwargs["ha"] == "center"
    assert kwargs["va"] == "bottom"
    assert kwargs["rotation"] == 0
    assert kwargs["fontsize"] == 12


def test_grid_barplot_labels_alternative_column(mock_tile):
    df = pl.DataFrame(
        {
            "category": ["A", "B"],
            "hue": ["H1", "H1"],
            "val": [10.0, 20.0],
            "custom_label": ["Apples", "Oranges"],
        }
    )
    config = BarPlotConfig(show_bar_label=True, bar_label="<custom_label>")

    grid_barplot(mock_tile, df, config, x="<category>", y="val")

    # annotate should be called with "Apples" and "Oranges"
    calls = mock_tile.ax.annotate.call_args_list
    assert calls[0][0][0] == "Apples"
    assert calls[1][0][0] == "Oranges"


def test_grid_barplot_labels_with_error_bars_no_overlap(mock_tile):
    # Tests that the label is positioned above the error bar (upper limit)
    df = pl.DataFrame(
        {
            "category": ["A", "B"],
            "hue": ["H1", "H1"],
            "val": [10.0, 20.0],
            "err_lower": [8.0, 15.0],
            "err_upper": [13.0, 24.0],
        }
    )
    config = BarPlotConfig(show_bar_label=True)

    grid_barplot(
        mock_tile, df, config, x="<category>", y="val", err=("err_lower", "err_upper")
    )

    calls = mock_tile.ax.annotate.call_args_list
    # The first bar has upper error limit 13.0, so label should be at xy=(0.0, 13.0)
    assert calls[0][0][0] == "10"
    assert calls[0][1]["xy"] == pytest.approx((0.0, 13.0))
    # The second bar has upper error limit 24.0, so label should be at xy=(1.0, 24.0)
    assert calls[1][0][0] == "20"
    assert calls[1][1]["xy"] == pytest.approx((1.0, 24.0))


def test_grid_barplot_labels_orient_y(mock_tile):
    df = pl.DataFrame(
        {"category": ["A", "B"], "hue": ["H1", "H1"], "val": [10.0, 20.5]}
    )
    config = BarPlotConfig(show_bar_label=True, orient="y")

    grid_barplot(mock_tile, df, config, x="<val>", y="category")

    assert mock_tile.ax.annotate.call_count == 2

    # Inspect first call in horizontal mode
    # xy should be (val + stack, coord)
    # Since orient="y", val is x axis (dependent) and category is y axis (independent).
    # coordinate on independent axis (category) for A is 0.0, offset is 0.0.
    # val is 10.0. So xy should be (10.0, 0.0)
    # xytext should be (3, 0)
    first_call = mock_tile.ax.annotate.call_args_list[0]
    args, kwargs = first_call
    assert args[0] == "10"
    assert kwargs["xy"] == (10.0, 0.0)
    assert kwargs["xytext"] == (3, 0)
    assert kwargs["ha"] == "left"
    assert kwargs["va"] == "center"
