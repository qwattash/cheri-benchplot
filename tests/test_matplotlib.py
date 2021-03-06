import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox

from pycheribenchplot.core.analysis import AnalysisConfig
from pycheribenchplot.core.plot import BarPlotDataView, DataView, LegendInfo
from pycheribenchplot.core.plot.backend import Mosaic
from pycheribenchplot.core.plot.matplotlib import (BarRenderer, DynamicCoordAllocator, MplCellData, MplFigureManager,
                                                   align_y_at)


@pytest.fixture
def fake_manager():
    return MplFigureManager(AnalysisConfig())


@pytest.fixture
def fake_cell(fake_manager):
    fig = plt.figure()
    ax = fig.subplots(1, 1)
    cell = MplCellData(title="fake", figure=fig, ax=ax)
    cell.rax = ax.twinx()
    yield cell
    plt.close(fig)


@pytest.fixture
def bar_renderer():
    return BarRenderer()


@pytest.fixture(params=[
    {
        "yleft": [10, 20],
        "yright": [30, -30]
    },
    {
        "yleft": [10, -10],
        "yright": [30, -30]
    },
])
def yaxis_align_df(request):
    df = pd.DataFrame()
    # 2 bars with 2 groups, one for the left and one for the right axis
    # the data should produce an unaligned Y axis origin
    df["x"] = [0, 5]
    df["yleft"] = request.param["yleft"]
    df["yright"] = request.param["yright"]
    return df


def test_yaxis_aligment(yaxis_align_df):
    test_df = yaxis_align_df
    # plot on a pair of axes
    f, ax = plt.subplots(1, 1)
    rax = ax.twinx()

    ax.bar(test_df["x"], test_df["yleft"], width=0.8, color="g")
    l_ymin = min(test_df["yleft"].min(), 0)
    l_ymax = max(test_df["yleft"].max(), 0)
    ax.set_ylim(l_ymin, l_ymax)

    rax.bar(test_df["x"] + 1, test_df["yright"], width=0.8, color="r")
    r_ymin = min(test_df["yright"].min(), 0)
    r_ymax = max(test_df["yright"].max(), 0)
    rax.set_ylim(r_ymin, r_ymax)

    align_y_at(ax, 0, rax, 0)

    # check that origin axis is aligned
    _, l_origin = ax.transData.transform((0, 0))
    _, r_origin = ax.transData.transform((0, 0))
    assert l_origin == r_origin, "Axis alignment failed"

    # check that all the data is shown
    out_ylim = ax.get_ylim()
    assert out_ylim[0] <= l_ymin, "Left axis ymin clipped"
    assert out_ylim[1] >= l_ymax, "Left axis ymax clipped"
    out_ylim = rax.get_ylim()
    assert out_ylim[0] <= r_ymin, "Right axis ymin clipped"
    assert out_ylim[1] >= r_ymax, "Right axis ymax clipped"


def test_bar_render_xpos_simple(fake_cell, bar_renderer):
    """
    Test simple bar positioning with a single set of bars
    """
    df = pd.DataFrame()
    df["x"] = [1, 2, 3, 4]
    df["y"] = [10, 20, 15, 30]

    allocator = DynamicCoordAllocator(fake_cell.ax,
                                      axis=0,
                                      group_keys=[],
                                      stack_keys=[],
                                      group_width=0.8,
                                      group_align="center",
                                      group_order="sequential")
    out_df = allocator.compute_coords(df, x="x", left_cols=["y"], right_cols=[], prefix="__prefix")
    assert np.isclose(out_df["__prefix_x_l0"], [1, 2, 3, 4]).all()
    assert np.isclose(out_df["__prefix_width_l0"], [0.8] * 4).all()

    view = BarPlotDataView(df.copy(), x="x", yleft="y")
    view.bar_width = 0.8
    view.bar_group_location = "center"
    fake_cell.add_view(view)
    out_df = bar_renderer._compute_bar_x(fake_cell, view, fake_cell.ax, group_by=[])
    assert np.isclose(out_df["__bar_x_l0"], [1, 2, 3, 4]).all()
    assert np.isclose(out_df["__bar_width_l0"], [0.8] * 4).all()


def test_bar_render_xpos_2cols(fake_cell, bar_renderer):
    """
    Test simple bar positioning with 2 sets of bars corresponding to two Y columns
    on the same Y axis
    """
    df = pd.DataFrame()
    df["x"] = [2, 4, 6, 8]
    df["y0"] = [10, 20, 30, 40]
    df["y1"] = [15, 25, 35, 45]

    allocator = DynamicCoordAllocator(fake_cell.ax,
                                      axis=0,
                                      group_keys=[],
                                      stack_keys=[],
                                      group_width=0.5,
                                      group_align="center",
                                      group_order="sequential")
    out_df = allocator.compute_coords(df, x="x", left_cols=["y0", "y1"], right_cols=[], prefix="__prefix")
    assert np.isclose(out_df["__prefix_x_l0"], [1.75, 3.75, 5.75, 7.75]).all()
    assert np.isclose(out_df["__prefix_x_l1"], [2.25, 4.25, 6.25, 8.25]).all()
    assert np.isclose(out_df["__prefix_width_l0"], [0.5] * 4).all()
    assert np.isclose(out_df["__prefix_width_l1"], [0.5] * 4).all()

    view = BarPlotDataView(df.copy(), x="x", yleft=["y0", "y1"])
    view.bar_width = 0.5
    view.bar_group_location = "center"
    fake_cell.add_view(view)
    out_df = bar_renderer._compute_bar_x(fake_cell, view, fake_cell.ax, group_by=[])
    assert np.isclose(out_df["__bar_x_l0"], [1.75, 3.75, 5.75, 7.75]).all()
    assert np.isclose(out_df["__bar_x_l1"], [2.25, 4.25, 6.25, 8.25]).all()
    assert np.isclose(out_df["__bar_width_l0"], [0.5] * 4).all()
    assert np.isclose(out_df["__bar_width_l1"], [0.5] * 4).all()


def test_bar_render_xpos_2groups(fake_cell, bar_renderer):
    """
    Test simple bar positioning with 2 column groups on the same Y axis
    """
    df = pd.DataFrame()
    df["k"] = [0] * 4 + [1] * 4
    df["x"] = [2, 4, 6, 8] * 2
    df.set_index(["k", "x"], inplace=True)
    df["y"] = np.arange(8) * 10

    allocator = DynamicCoordAllocator(fake_cell.ax,
                                      axis=0,
                                      group_keys=["k"],
                                      stack_keys=[],
                                      group_width=0.5,
                                      group_align="center",
                                      group_order="sequential")
    out_df = allocator.compute_coords(df, x="x", left_cols=["y"], right_cols=[], prefix="__prefix")
    assert np.isclose(out_df["__prefix_x_l0"], [1.75, 3.75, 5.75, 7.75] + [2.25, 4.25, 6.25, 8.25]).all()
    assert np.isclose(out_df["__prefix_width_l0"], [0.5] * 8).all()

    view = BarPlotDataView(df.copy(), x="x", yleft=["y"], bar_group="k")
    view.bar_width = 0.5
    view.bar_group_location = "center"
    fake_cell.add_view(view)
    out_df = bar_renderer._compute_bar_x(fake_cell, view, fake_cell.ax, ["k"])
    assert np.isclose(out_df["__bar_x_l0"], [1.75, 3.75, 5.75, 7.75] + [2.25, 4.25, 6.25, 8.25]).all()
    assert np.isclose(out_df["__bar_width_l0"], [0.5] * 8).all()


def test_bar_render_xpos_2stacks(fake_cell, bar_renderer):
    """
    Test simple bar positioning with a single set of bars and 2 stacks for each bar
    """
    df = pd.DataFrame()
    df["k"] = [0] * 4 + [1] * 4
    df["x"] = [2, 4, 6, 8] * 2
    df.set_index(["k", "x"], inplace=True)
    df["y"] = [10, 20, 30, 50, 5, 5, 5, 5]

    allocator = DynamicCoordAllocator(fake_cell.ax,
                                      axis=0,
                                      group_keys=[],
                                      stack_keys=["k"],
                                      group_width=0.5,
                                      group_align="center",
                                      group_order="sequential")
    out_df = allocator.compute_coords(df, x="x", left_cols=["y"], right_cols=[], prefix="__prefix")
    assert np.isclose(out_df["__prefix_x_l0"], [2, 4, 6, 8] * 2).all()
    assert np.isclose(out_df["__prefix_width_l0"], [1] * 8).all()
    assert np.isclose(out_df["__prefix_base_l0"], [0, 0, 0, 0, 10, 20, 30, 50]).all()

    view = BarPlotDataView(df.copy(), x="x", yleft=["y"], stack_group="k")
    view.bar_width = 0.5
    view.bar_group_location = "center"
    fake_cell.add_view(view)
    out_df = bar_renderer._compute_bar_x(fake_cell, view, fake_cell.ax, ["k"])
    assert np.isclose(out_df["__bar_x_l0"], [2, 4, 6, 8] * 2).all()
    assert np.isclose(out_df["__bar_width_l0"], [1] * 8).all()
    assert np.isclose(out_df["__bar_base_l0"], [0, 0, 0, 0, 10, 20, 30, 50]).all()


def test_bar_render_xpos_2groups_2cols(fake_cell, bar_renderer):
    """
    Test composite bar positioning with 2 groups each with 2 bars corresponding to
    two columns.
    """
    df = pd.DataFrame()
    df["k"] = [0] * 4 + [1] * 4
    df["x"] = [2, 4, 6, 8] * 2
    df.set_index(["k", "x"], inplace=True)
    df["y0"] = [10, 20, 30, 50, 5, 5, 5, 5]
    df["y1"] = [5, 5, 5, 5, 100, 200, 300, 400]

    allocator = DynamicCoordAllocator(fake_cell.ax,
                                      axis=0,
                                      group_keys=["k"],
                                      stack_keys=[],
                                      group_width=1,
                                      group_align="center",
                                      group_order="sequential")
    out_df = allocator.compute_coords(df, x="x", left_cols=["y0", "y1"], right_cols=[], prefix="__prefix")
    assert np.isclose(out_df["__prefix_x_l0"], [1.25, 3.25, 5.25, 7.25] + [1.75, 3.75, 5.75, 7.75]).all()
    assert np.isclose(out_df["__prefix_x_l1"], [2.25, 4.25, 6.25, 8.25] + [2.75, 4.75, 6.75, 8.75]).all()
    assert np.isclose(out_df["__prefix_width_l0"], [0.5] * 8).all()

    view = BarPlotDataView(df.copy(), x="x", yleft=["y0", "y1"], bar_group="k")
    view.bar_width = 1
    view.bar_group_location = "center"
    fake_cell.add_view(view)
    out_df = bar_renderer._compute_bar_x(fake_cell, view, fake_cell.ax, ["k"])
    assert np.isclose(out_df["__bar_x_l0"], [1.25, 3.25, 5.25, 7.25] + [1.75, 3.75, 5.75, 7.75]).all()
    assert np.isclose(out_df["__bar_x_l1"], [2.25, 4.25, 6.25, 8.25] + [2.75, 4.75, 6.75, 8.75]).all()
    assert np.isclose(out_df["__bar_width_l0"], [0.5] * 8).all()


def test_bar_render_xpos_2groups_2cols_2stacks(fake_cell, bar_renderer):
    """
    Test composite bar positioning with 2 groups each with 2 bars corresponding to
    two columns.
    """
    df = pd.DataFrame()
    df["k"] = [0] * 4 + [1] * 4  # group
    df["s"] = ["a", "a", "b", "b"] * 2  # stack
    df["x"] = [4, 8, 4, 8] * 2
    df.set_index(["k", "s", "x"], inplace=True)
    df["y0"] = [10, 20] + [25, 15] + [30, 40] + [45, 35]
    df["y1"] = [110, 120] + [125, 115] + [130, 140] + [145, 135]

    allocator = DynamicCoordAllocator(fake_cell.ax,
                                      axis=0,
                                      group_keys=["k"],
                                      stack_keys=["s"],
                                      group_width=1,
                                      group_align="center",
                                      group_order="sequential")
    out_df = allocator.compute_coords(df, x="x", left_cols=["y0", "y1"], right_cols=[], prefix="__prefix")
    assert np.isclose(out_df["__prefix_x_l0"], [2.5, 6.5, 3.5, 7.5] * 2).all()
    assert np.isclose(out_df["__prefix_x_l1"], [4.5, 8.5, 5.5, 9.5] * 2).all()
    assert np.isclose(out_df["__prefix_width_l0"], [1] * 8).all()
    assert np.isclose(out_df["__prefix_width_l1"], [1] * 8).all()
    assert np.isclose(out_df["__prefix_base_l0"], [0, 0, 10, 20, 0, 0, 30, 40]).all()
    assert np.isclose(out_df["__prefix_base_l1"], [0, 0, 110, 120, 0, 0, 130, 140]).all()

    view = BarPlotDataView(df.copy(), x="x", yleft=["y0", "y1"], bar_group="k", stack_group="s")
    view.bar_width = 1
    view.bar_group_location = "center"
    fake_cell.add_view(view)
    out_df = bar_renderer._compute_bar_x(fake_cell, view, fake_cell.ax, ["k", "s"])
    assert np.isclose(out_df["__bar_x_l0"], [2.5, 6.5, 3.5, 7.5] * 2).all()
    assert np.isclose(out_df["__bar_x_l1"], [4.5, 8.5, 5.5, 9.5] * 2).all()
    assert np.isclose(out_df["__bar_width_l0"], [1] * 8).all()
    assert np.isclose(out_df["__bar_width_l1"], [1] * 8).all()
    assert np.isclose(out_df["__bar_base_l0"], [0, 0, 10, 20, 0, 0, 30, 40]).all()
    assert np.isclose(out_df["__bar_base_l1"], [0, 0, 110, 120, 0, 0, 130, 140]).all()


def test_bar_render_xpos_2groups_2axes_interleaved(fake_cell, bar_renderer):
    """
    Test interleaved bar positioning with 2 sets of bars corresponding to two Y columns
    on the same Y axis
    """
    df = pd.DataFrame()
    df["x"] = [2, 4, 6, 8]
    df["y0"] = [10, 20, 30, 40]
    df["y1"] = [15, 25, 35, 45]

    allocator = DynamicCoordAllocator(fake_cell.ax,
                                      axis=0,
                                      group_keys=[],
                                      stack_keys=[],
                                      group_width=0.5,
                                      group_align="center",
                                      group_order="interleaved")
    out_df = allocator.compute_coords(df, x="x", left_cols=["y0", "y1"], right_cols=[], prefix="__prefix")
    assert np.isclose(out_df["__prefix_x_l0"], [1.75, 3.75, 5.75, 7.75]).all()
    assert np.isclose(out_df["__prefix_x_l1"], [2.25, 4.25, 6.25, 8.25]).all()
    assert np.isclose(out_df["__prefix_width_l0"], [0.5] * 4).all()
    assert np.isclose(out_df["__prefix_width_l1"], [0.5] * 4).all()

    view = BarPlotDataView(df.copy(), x="x", yleft=["y0", "y1"])
    view.bar_axes_order = "interleaved"
    view.bar_width = 0.5
    view.bar_group_location = "center"
    fake_cell.add_view(view)
    out_df = bar_renderer._compute_bar_x(fake_cell, view, fake_cell.ax, group_by=[])
    assert np.isclose(out_df["__bar_x_l0"], [1.75, 3.75, 5.75, 7.75]).all()
    assert np.isclose(out_df["__bar_x_l1"], [2.25, 4.25, 6.25, 8.25]).all()
    assert np.isclose(out_df["__bar_width_l0"], [0.5] * 4).all()
    assert np.isclose(out_df["__bar_width_l1"], [0.5] * 4).all()


@pytest.mark.parametrize("ngroups, nleft, nright, expect", [
    (1, 2, 0, {
        (0, "l"): [0],
        (1, "l"): [1]
    }),
    (2, 2, 0, {
        (0, "l"): [0, 2],
        (1, "l"): [1, 3]
    }),
    (3, 2, 0, {
        (0, "l"): [0, 2, 4],
        (1, "l"): [1, 3, 5]
    }),
    (3, 1, 1, {
        (0, "l"): [0, 2, 4],
        (0, "r"): [1, 3, 5]
    }),
    (3, 2, 1, {
        (0, "l"): [0, 3, 6],
        (0, "r"): [1, 4, 7],
        (1, "l"): [2, 5, 8]
    }),
])
def test_bar_render_gen_matrix_slices_interleaved(ngroups, nleft, nright, expect, fake_cell):
    # Irrelevant if filled or not
    df = pd.DataFrame()
    naxes = nleft + nright
    yleft = [f"yl{i}" for i in range(nleft)]
    yright = [f"yr{i}" for i in range(nright)]
    fake_posmatrix = list(range(ngroups * naxes))

    allocator = DynamicCoordAllocator(fake_cell.ax,
                                      axis=0,
                                      group_keys=[],
                                      stack_keys=[],
                                      group_width=0.8,
                                      group_align="center",
                                      group_order="interleaved")
    gen = allocator._iter_group_slices(yleft, yright, ngroups, naxes)
    items = list(gen)
    result = {(i, lr): fake_posmatrix[s] for s, i, lr in items}
    assert result == expect


def test_bar_renderer_top_text(mocker, fake_cell, bar_renderer):
    df = pd.DataFrame()
    df["x"] = [1, 2, 3, 4]
    # fake generated x positionfs for column "y"
    df["__bar_x_l0"] = [1, 2, 3, 4]
    df["y"] = [10, 20, 15, 30]
    df.set_index("x", inplace=True)
    view = BarPlotDataView(df.copy(), x="x", yleft="y")
    view.bar_text = True
    view.bar_text_pad = 0
    view.bar_width = 0.8
    view.bar_group_location = "center"
    fake_cell.add_view(view)

    text_fn = mocker.patch.object(fake_cell.ax, "text")
    text_fn().get_window_extent().transformed.return_value = Bbox.null()

    bar_renderer._draw_columns_text(view, fake_cell)
    expect_kw = {
        "fontsize": mocker.ANY,
        "rotation": "vertical",
        "ha": mocker.ANY,
        "va": mocker.ANY,
    }
    text_fn.assert_any_call(1, 10, "10", **expect_kw)
    text_fn.assert_any_call(2, 20, "20", **expect_kw)
    text_fn.assert_any_call(3, 15, "15", **expect_kw)
    text_fn.assert_any_call(4, 30, "30", **expect_kw)


def test_bar_renderer_top_text_2col(mocker, fake_cell, bar_renderer):
    df = pd.DataFrame()
    df["x"] = [1, 2, 3, 4]
    # fake generated x positionfs for column "y"
    df["__bar_x_l0"] = [0.75, 1.75, 2.75, 3.75]
    df["__bar_x_l1"] = [1.25, 2.25, 3.25, 4.25]
    df["y0"] = [10, 20, 15, 30]
    df["y1"] = [11, 19, 14, 35]
    df.set_index("x", inplace=True)
    view = BarPlotDataView(df.copy(), x="x", yleft=["y0", "y1"])
    view.bar_text = True
    view.bar_text_pad = 0
    view.bar_width = 1
    view.bar_group_location = "center"
    fake_cell.add_view(view)

    text_fn = mocker.patch.object(fake_cell.ax, "text")
    text_fn().get_window_extent().transformed.return_value = Bbox.null()

    bar_renderer._draw_columns_text(view, fake_cell)
    expect_kw = {
        "fontsize": mocker.ANY,
        "rotation": "vertical",
        "ha": mocker.ANY,
        "va": mocker.ANY,
    }
    text_fn.assert_any_call(0.75, 10, "10", **expect_kw)
    text_fn.assert_any_call(1.75, 20, "20", **expect_kw)
    text_fn.assert_any_call(2.75, 15, "15", **expect_kw)
    text_fn.assert_any_call(3.75, 30, "30", **expect_kw)

    text_fn.assert_any_call(1.25, 11, "11", **expect_kw)
    text_fn.assert_any_call(2.25, 19, "19", **expect_kw)
    text_fn.assert_any_call(3.25, 14, "14", **expect_kw)
    text_fn.assert_any_call(4.25, 35, "35", **expect_kw)


def test_bar_renderer_top_text_2col_2stacks(mocker, fake_cell, bar_renderer):
    df = pd.DataFrame()
    df["k"] = [0] * 4 + [1] * 4
    df["x"] = [1, 2, 3, 4] * 2
    # fake generated x positionfs for column "y"
    df["__bar_x_l0"] = [0.75, 1.75, 2.75, 3.75] * 2
    df["__bar_x_l1"] = [1.25, 2.25, 3.25, 4.25] * 2
    df["y0"] = [10, 20, 15, 30] + [5, 6, 7, 4]
    df["y1"] = [3, 4, 10, 5] + [7, 16, 5, 10]
    df.set_index(["x", "k"], inplace=True)
    view = BarPlotDataView(df.copy(), x="x", yleft=["y0", "y1"], stack_group="k")
    view.bar_text = True
    view.bar_text_pad = 0
    view.bar_width = 1
    view.bar_group_location = "center"
    fake_cell.add_view(view)

    text_fn = mocker.patch.object(fake_cell.ax, "text")
    text_fn().get_window_extent().transformed.return_value = Bbox.null()

    bar_renderer._draw_columns_text(view, fake_cell)
    expect_kw = {
        "fontsize": mocker.ANY,
        "rotation": "vertical",
        "ha": mocker.ANY,
        "va": mocker.ANY,
    }
    text_fn.assert_any_call(0.75, 15, "15", **expect_kw)
    text_fn.assert_any_call(1.75, 26, "26", **expect_kw)
    text_fn.assert_any_call(2.75, 22, "22", **expect_kw)
    text_fn.assert_any_call(3.75, 34, "34", **expect_kw)

    text_fn.assert_any_call(1.25, 10, "10", **expect_kw)
    text_fn.assert_any_call(2.25, 20, "20", **expect_kw)
    text_fn.assert_any_call(3.25, 15, "15", **expect_kw)
    text_fn.assert_any_call(4.25, 15, "15", **expect_kw)


@pytest.fixture
def legend_info_2levels():
    index = pd.MultiIndex.from_product([["J0", "J1"], ["K0", "K1"]], names=["J", "K"])
    legend_info = LegendInfo.from_index(index, ["L0", "L1", "L2", "L3"], colors=["C0", "C1", "C2", "C3"])
    return legend_info


def test_legend_info_simple():
    legend_info = LegendInfo.from_index(["K0", "K1"], ["L0", "L1"], cmap_name="Greys")

    WHITE = [1, 1, 1, 1]
    BLACK = [0, 0, 0, 1]

    assert (legend_info.colors["K0"] == WHITE).all()
    assert (legend_info.colors["K1"] == BLACK).all()
    assert legend_info.labels["K0"] == "L0"
    assert legend_info.labels["K1"] == "L1"
    assert (legend_info.info_df.index == ["K0", "K1"]).all()

    new_info = legend_info.map_label(lambda l: "mapped-" + l)
    assert (new_info.colors["K0"] == WHITE).all()
    assert (new_info.colors["K1"] == BLACK).all()
    assert new_info.labels["K0"] == "mapped-L0"
    assert new_info.labels["K1"] == "mapped-L1"


def test_legend_info_concat():
    base_index = pd.Index(["K0", "K1"], name="K")
    legend_info_1 = LegendInfo.from_index(base_index, ["L0", "L1"], colors=["C0", "C1"])
    legend_info_2 = LegendInfo.from_index(base_index, ["L2", "L3"], colors=["C2", "C3"])
    legend_info = LegendInfo.combine("J", {"J0": legend_info_1, "J1": legend_info_2})

    assert legend_info.info_df.index.names == ["J", "K"]
    assert legend_info.colors[("J0", "K0")] == "C0"
    assert legend_info.colors[("J0", "K1")] == "C1"
    assert legend_info.colors[("J1", "K0")] == "C2"
    assert legend_info.colors[("J1", "K1")] == "C3"
    assert legend_info.labels[("J0", "K0")] == "L0"
    assert legend_info.labels[("J0", "K1")] == "L1"
    assert legend_info.labels[("J1", "K0")] == "L2"
    assert legend_info.labels[("J1", "K1")] == "L3"


def test_legend_info_map_colors(legend_info_2levels):
    # By default, map_color maps each value separately
    new_info = legend_info_2levels.map_color(lambda c: c + [f"-mapped-{i}" for i in np.arange(len(c))])
    assert new_info.colors[("J0", "K0")] == "C0-mapped-0"
    assert new_info.colors[("J0", "K1")] == "C1-mapped-0"
    assert new_info.colors[("J1", "K0")] == "C2-mapped-0"
    assert new_info.colors[("J1", "K1")] == "C3-mapped-0"

    # Check grouping with a level
    new_info = legend_info_2levels.map_color(lambda c: c + [f"-mapped-{i}" for i in np.arange(len(c))], group_by="J")
    assert new_info.colors[("J0", "K0")] == "C0-mapped-0"
    assert new_info.colors[("J0", "K1")] == "C1-mapped-1"
    assert new_info.colors[("J1", "K0")] == "C2-mapped-0"
    assert new_info.colors[("J1", "K1")] == "C3-mapped-1"

    # Check grouping with a level
    new_info = legend_info_2levels.map_color(lambda c: c + [f"-mapped-{i}" for i in np.arange(len(c))], group_by="K")
    assert new_info.colors[("J0", "K0")] == "C0-mapped-0"
    assert new_info.colors[("J0", "K1")] == "C1-mapped-0"
    assert new_info.colors[("J1", "K0")] == "C2-mapped-1"
    assert new_info.colors[("J1", "K1")] == "C3-mapped-1"


def test_legend_info_remap_colors():
    legend_info = LegendInfo.from_index(["K0", "K1"], ["L0", "L1"], colors=["C0", "C1"])

    new_info = legend_info.remap_colors("Greys")

    WHITE = [1, 1, 1, 1]
    BLACK = [0, 0, 0, 1]
    assert (new_info.colors["K0"] == WHITE).all()
    assert (new_info.colors["K1"] == BLACK).all()


def test_legend_info_remap_group_colors(legend_info_2levels):
    new_info = legend_info_2levels.remap_colors("Greys", groupby="K")

    WHITE = [1, 1, 1, 1]
    BLACK = [0, 0, 0, 1]
    assert (new_info.colors[("J0", "K0")] == WHITE).all()
    assert (new_info.colors[("J0", "K1")] == BLACK).all()
    assert (new_info.colors[("J1", "K0")] == WHITE).all()
    assert (new_info.colors[("J1", "K1")] == BLACK).all()


def test_legend_info_assign_colors(legend_info_2levels):
    WHITE = [1, 1, 1, 1]
    BLACK = [0, 0, 0, 1]

    def scale_colors(base_color, color_vec):
        assert len(color_vec) == 2
        J = color_vec.index.get_level_values("J")[0]
        if J == "J0":
            assert (base_color == WHITE).all()
        elif J == "J1":
            assert (base_color == BLACK).all()
        else:
            assert False, "This index should not exist"
        return color_vec.map(lambda c: c + 1)

    new_info = legend_info_2levels.assign_colors("Greys", ["J"], scale_colors)

    # Not valid colors, just for checking the result of the map
    NEW_WHITE = [2, 2, 2, 2]
    NEW_BLACK = [1, 1, 1, 2]
    assert (new_info.colors[("J0", "K0")] == NEW_WHITE).all()
    assert (new_info.colors[("J0", "K1")] == NEW_WHITE).all()
    assert (new_info.colors[("J1", "K0")] == NEW_BLACK).all()
    assert (new_info.colors[("J1", "K1")] == NEW_BLACK).all()


@pytest.mark.skip
def test_legend_info_assign_hsv(legend_info_2levels):
    legend = legend_info_2levels.assign_colors_hsv("J", h=(0.3, 1), s=(0.5, 1), v=(0.4, 0.9))
    ## XXX todo


def test_data_view_col():
    df = pd.DataFrame()
    df["J"] = ["J0", "J0", "J1", "J1"]
    df["K"] = ["K0", "K1", "k0", "K1"]
    df["v"] = [1, 2, 3, 4]
    df["w"] = [10, 20, 30, 40]
    df.set_index(["J", "K"], inplace=True)
    view = DataView(df)

    col = view.get_col("v")
    assert type(col) == pd.Series
    assert col.shape == (4, )
    assert (col == [1, 2, 3, 4]).all()

    col = view.get_col(["v"])
    assert type(col) == pd.DataFrame
    assert col.shape == (4, 1)
    assert (col["v"] == [1, 2, 3, 4]).all()

    col = view.get_col("J")
    assert type(col) == pd.Series
    assert col.shape == (4, )
    assert (col == ["J0", "J0", "J1", "J1"]).all()

    col = view.get_col(["v", "w"])
    assert col.shape == (4, 2)
    assert (col.columns == ["v", "w"]).all()
    assert (col.index == df.index).all()
    assert (col["v"] == df["v"]).all()
    assert (col["w"] == df["w"]).all()

    col = view.get_col(["J", "K"])
    assert col.shape == (4, 2)
    assert (col.columns == ["J", "K"]).all()
    assert (col.index == df.index).all()
    assert (col["J"] == df.index.get_level_values("J")).all()
    assert (col["K"] == df.index.get_level_values("K")).all()

    col = view.get_col(["w", "K"])
    assert col.shape == (4, 2)
    assert (col.columns == ["w", "K"]).all()
    assert (col.index == df.index).all()
    assert (col["w"] == df["w"]).all()
    assert (col["K"] == df.index.get_level_values("K")).all()
