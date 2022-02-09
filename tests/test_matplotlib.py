import matplotlib.pyplot as plt
import pandas as pd
import pytest

from pycheribenchplot.core.matplotlib import (BarRenderer, MatplotlibPlotCell, MatplotlibSurface, align_y_at)
from pycheribenchplot.core.plot import BarPlotDataView, GridLayout


@pytest.fixture
def fake_ctx(tmp_path):
    s = MatplotlibSurface()
    s.set_layout(GridLayout(1, 1))
    ctx = s._make_draw_context("fake", tmp_path)
    ctx.ax = ctx.axes[0][0]
    ctx.rax = ctx.ax.twinx()
    return ctx


@pytest.fixture
def fake_cell():
    cell = MatplotlibPlotCell()
    return cell


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
