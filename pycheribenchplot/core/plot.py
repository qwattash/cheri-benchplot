"""
General purpose matplotlib helpers
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, local
from typing import List, Optional
from uuid import UUID

import matplotlib as mpl
import polars as pl
import seaborn as sns

from .analysis import AnalysisTask, DatasetAnalysisTask, SliceAnalysisTask
from .artefact import Target
from .config import Config
from .task import Task


class RcParamsThreadWrapper:
    """
    Evil matplotlib monkey patch to make rcParams thread local.
    This will allow concurrent plotting.
    """
    def __init__(self, mpl_rc):
        self.thr_params = local()
        self.thr_params.rc = mpl_rc

    def __getattr__(self, name):
        return getattr(self.thr_params.rc, name)

    def __contains__(self, name):
        return name in self.thr_params.rc

    def __getitem__(self, key):
        return self.thr_params.rc.__getitem__(key)

    def __setitem__(self, key, value):
        self.thr_params.rc.__setitem__(key, value)

    def __delitem__(self, key):
        del self.thr_params.rc[key]

    def __repr__(self):
        return f"ThreadedRcParams{{{self.thr_params.rc}}}"

    def setup_thread(self):
        self.thr_params.rc = getattr(self.thr_params, "rc", default_rc.copy())


@contextmanager
def wrap_rc_context(rc=None, fname=None):
    """
    Evil monkey patch to update rcParams correctly.
    Patched rc_context code for matplotlib to play well with the Rc params wrapper
    """
    orig = dict(mpl.rcParams.copy())
    del orig['backend']
    try:
        if fname:
            mpl.rc_file(fname)
        if rc:
            mpl.rcParams.update(rc)
        yield
    finally:
        mpl.rcParams.update(orig)  # Revert to the original rcs.


def setup_matplotlib_hooks():
    default_rc = mpl.rcParams
    mpl.rc_context = wrap_rc_context
    mpl.rcParams = RcParamsThreadWrapper(default_rc.copy())


@contextmanager
def new_figure(dest: Path | list[Path], bbox_inches="tight", **kwargs):
    """
    Helper context manager to produce a new figure
    """
    kwargs.setdefault("constrained_layout", True)
    fig = mpl.pyplot.figure(**kwargs)
    yield fig
    if isinstance(dest, Path):
        dest = [dest]
    for path in dest:
        fig.savefig(path, bbox_inches=bbox_inches)
    mpl.pyplot.close(fig)


class CustomFacetGrid(sns.FacetGrid):
    """
    Facet grid with an hack to force legend extraction from subfigures.
    """
    def map_dataframe(self, func, *args, **kwargs):
        if func is sns.histplot or func is sns.kdeplot:
            self._extract_legend_handles = True
        return super().map_dataframe(func, *args, **kwargs)


@contextmanager
def new_facet(dest: Path | list[Path], *args, savefig_kws: dict | None = None, **kwargs):
    """
    Helper context manager to produce a new seaborn facetgrid.
    Arguments and keyword arguments are given to the FacetGrid constructor,
    savefig_kws should be used to add savefig() function arguments.
    """
    if savefig_kws is None:
        savefig_kws = {}
    if isinstance(dest, Path):
        dest = [dest]

    facet = CustomFacetGrid(*args, **kwargs)
    yield facet
    for path in dest:
        facet.savefig(path, **savefig_kws)
    mpl.pyplot.close(facet.figure)


class PlotTarget(Target):
    """
    Target pointing to a plot path.

    The output path depends on whether the plot is associated to the session or to
    a single dataset.
    """
    def __init__(self, task: Task, output_id: str = "plot", **kwargs):
        kwargs.setdefault("ext", task.analysis_config.plot.plot_output_format)
        super().__init__(task, output_id, **kwargs)

    def iter_paths(self, **kwargs):
        for path in super().iter_paths(**kwargs):
            path.parent.mkdir(exist_ok=True)
            yield path


class PlotTaskMixin:
    """
    Base class for plotting tasks.
    Plot tasks generate one or more plots from some analysis task data.
    These are generally the public-facing tasks that are selected in the analysis
    configuration.
    Each plot task is responsible for setting up a figure and axes.
    """

    # XXX temporary mutex to serialize matplotlib rcParams overrides
    # This should be moved to the figure and facet context managers
    rc_mutex = Lock()

    def _plot_output(self, suffix: str = None) -> PlotTarget:
        """
        Deprecated way to build the plot target. Should use PlotTarget directly.
        """
        if suffix:
            name = f"{self.task_id}-{suffix}.pdf"
        else:
            name = f"{self.task_id}.pdf"
        if self.is_benchmark_task:
            base = self.benchmark.get_plot_path()
        else:
            base = self.session.get_plot_root_path()
        return PlotTarget(self, base / name)

    def _run_with_plot_sandbox(self):
        try:
            rc_override = getattr(self, "rc_params")
        except AttributeError:
            rc_override = {}

        def do_run():
            with mpl.rc_context():
                # Use default theme as base
                sns.set_theme()
                with sns.plotting_context(rc=rc_override):
                    self.run_plot()

        if self.analysis_config.plot.parallel:
            matplotlib.rcParams.setup_thread()
            do_run()
        else:
            with PlotTaskMixin.rc_mutex:
                do_run()

    def adjust_legend_on_top(self, fig, ax=None, **kwargs):
        """
        Helper function that adjusts the position of a legend in the given figure/axes.
        This works around some matplotlib quirks that are not handled cleanly by seaborn

        This works by forcibly removing the legend and re-creating it using the handles
        from the old legend.
        """
        if not fig.legends:
            return
        # Hack to remove the legend as we can not easily move it
        # Not sure why seaborn puts the legend in the figure here
        legend = fig.legends.pop()
        if ax is None:
            owner = fig
        else:
            owner = ax
        kwargs.setdefault("loc", "lower left")
        owner.legend(legend.legend_handles,
                     map(lambda t: t.get_text(), legend.texts),
                     bbox_to_anchor=(0., 1.02, 1, 0.2),
                     ncols=4,
                     **kwargs)

    def run_plot(self):
        """
        Plot task body.

        This runs within a matplotlib RC parameter context, so that local
        RC params are not propagated.
        """
        raise NotImplementedError("Must override")


class PlotTask(AnalysisTask, PlotTaskMixin):
    """
    Session-level plotting task.

    This task generates one or more plots that are unique within a session.
    This can be used to produce summary or aggregate plots from all the session datasets.
    """
    def run(self):
        self._run_with_plot_sandbox()


class DatasetPlotTask(DatasetAnalysisTask, PlotTaskMixin):
    """
    Dataset-level plotting task.

    This task generates one or more plots for each dataset collected.
    """
    def run(self):
        self._run_with_plot_sandbox()


class SlicePlotTask(SliceAnalysisTask, PlotTaskMixin):
    """
    SliceAnalysisTask that produces one or more plots.

    Note that the plot output will depend on the current slice ID.
    See :class:`PlotTarget`.
    """
    def run(self):
        self._run_with_plot_sandbox()
