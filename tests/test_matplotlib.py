import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pycheribenchplot.core.plot import BarPlotDataView
from pycheribenchplot.core.plot.backend import GridLayout
from pycheribenchplot.core.plot.matplotlib import (BarRenderer, MatplotlibPlotCell, MatplotlibSurface, align_y_at)


@pytest.fixture
def fake_surface():
    return MatplotlibSurface()


@pytest.fixture
def fake_ctx(fake_surface, tmp_path):
    fake_surface.set_layout(GridLayout(1, 1))
    ctx = fake_surface._make_draw_context("fake", tmp_path)
    ctx.ax = ctx.axes[0][0]
    ctx.rax = ctx.ax.twinx()
    return ctx


@pytest.fixture
def fake_cell(fake_surface):
    cell = MatplotlibPlotCell()
    cell.set_surface(fake_surface)
    return cell


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


def test_bar_render_xpos_simple(fake_ctx, fake_cell, bar_renderer):
    """
    Test simple bar positioning with a single set of bars
    """
    df = pd.DataFrame()
    df["x"] = [1, 2, 3, 4]
    df["y"] = [10, 20, 15, 30]
    view = BarPlotDataView(df.copy(), x="x", yleft="y")
    view.bar_width = 0.8
    view.bar_group_location = "center"
    fake_cell.add_view(view)

    out_df = bar_renderer._compute_bar_x(fake_cell, view, fake_ctx.ax)
    assert np.isclose(out_df["__bar_x_l0"], [1, 2, 3, 4]).all()
    assert np.isclose(out_df["__bar_width_l0"], [0.8] * 4).all()


def test_bar_render_xpos_2cols(fake_ctx, fake_cell, bar_renderer):
    """
    Test simple bar positioning with 2 sets of bars corresponding to two Y columns
    on the same Y axis
    """
    df = pd.DataFrame()
    df["x"] = [2, 4, 6, 8]
    df["y0"] = [10, 20, 30, 40]
    df["y1"] = [15, 25, 35, 45]
    view = BarPlotDataView(df.copy(), x="x", yleft=["y0", "y1"])
    view.bar_width = 0.5
    view.bar_group_location = "center"
    fake_cell.add_view(view)

    out_df = bar_renderer._compute_bar_x(fake_cell, view, fake_ctx.ax)
    assert np.isclose(out_df["__bar_x_l0"], [1.75, 3.75, 5.75, 7.75]).all()
    assert np.isclose(out_df["__bar_x_l1"], [2.25, 4.25, 6.25, 8.25]).all()
    assert np.isclose(out_df["__bar_width_l0"], [0.5] * 4).all()
    assert np.isclose(out_df["__bar_width_l1"], [0.5] * 4).all()


def test_bar_render_xpos_2groups(fake_ctx, fake_cell, bar_renderer):
    """
    Test simple bar positioning with 2 column groups on the same Y axis
    """
    df = pd.DataFrame()
    df["k"] = [0] * 4 + [1] * 4
    df["x"] = [2, 4, 6, 8] * 2
    df.set_index(["k", "x"], inplace=True)
    df["y"] = np.arange(8) * 10
    view = BarPlotDataView(df.copy(), x="x", yleft=["y"], bar_group="k")
    view.bar_width = 0.5
    view.bar_group_location = "center"
    fake_cell.add_view(view)

    out_df = bar_renderer._compute_bar_x(fake_cell, view, fake_ctx.ax)
    assert np.isclose(out_df["__bar_x_l0"], [1.75, 3.75, 5.75, 7.75] + [2.25, 4.25, 6.25, 8.25]).all()
    assert np.isclose(out_df["__bar_width_l0"], [0.5] * 8).all()


def test_bar_render_xpos_2stacks(fake_ctx, fake_cell, bar_renderer):
    """
    Test simple bar positioning with a single set of bars and 2 stacks for each bar
    """
    df = pd.DataFrame()
    df["k"] = [0] * 4 + [1] * 4
    df["x"] = [2, 4, 6, 8] * 2
    df.set_index(["k", "x"], inplace=True)
    df["y"] = [10, 20, 30, 50, 5, 5, 5, 5]
    view = BarPlotDataView(df.copy(), x="x", yleft=["y"], stack_group="k")
    view.bar_width = 0.5
    view.bar_group_location = "center"
    fake_cell.add_view(view)

    out_df = bar_renderer._compute_bar_x(fake_cell, view, fake_ctx.ax)
    assert np.isclose(out_df["__bar_x_l0"], [2, 4, 6, 8] * 2).all()
    assert np.isclose(out_df["__bar_width_l0"], [1] * 8).all()
    assert np.isclose(out_df["__bar_y_base_l0"], [0, 0, 0, 0, 10, 20, 30, 50]).all()


def test_bar_render_xpos_2groups_2cols(fake_ctx, fake_cell, bar_renderer):
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
    view = BarPlotDataView(df.copy(), x="x", yleft=["y0", "y1"], bar_group="k")
    view.bar_width = 1
    view.bar_group_location = "center"
    fake_cell.add_view(view)

    out_df = bar_renderer._compute_bar_x(fake_cell, view, fake_ctx.ax)
    assert np.isclose(out_df["__bar_x_l0"], [1.25, 3.25, 5.25, 7.25] + [1.75, 3.75, 5.75, 7.75]).all()
    assert np.isclose(out_df["__bar_x_l1"], [2.25, 4.25, 6.25, 8.25] + [2.75, 4.75, 6.75, 8.75]).all()
    assert np.isclose(out_df["__bar_width_l0"], [0.5] * 8).all()


def test_bar_render_xpos_2groups_2cols_2stacks(fake_ctx, fake_cell, bar_renderer):
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
    view = BarPlotDataView(df.copy(), x="x", yleft=["y0", "y1"], bar_group="k", stack_group="s")
    view.bar_width = 1
    view.bar_group_location = "center"
    fake_cell.add_view(view)

    out_df = bar_renderer._compute_bar_x(fake_cell, view, fake_ctx.ax)
    assert np.isclose(out_df["__bar_x_l0"], [2.5, 6.5, 3.5, 7.5] * 2).all()
    assert np.isclose(out_df["__bar_x_l1"], [4.5, 8.5, 5.5, 9.5] * 2).all()
    assert np.isclose(out_df["__bar_width_l0"], [1] * 8).all()
    assert np.isclose(out_df["__bar_width_l1"], [1] * 8).all()
    assert np.isclose(out_df["__bar_y_base_l0"], [0, 0, 10, 20, 0, 0, 30, 40]).all()
    assert np.isclose(out_df["__bar_y_base_l1"], [0, 0, 110, 120, 0, 0, 130, 140]).all()
