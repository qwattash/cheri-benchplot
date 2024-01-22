"""
General purpose matplotlib helpers
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import List, Optional
from uuid import UUID

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc_context

from .analysis import AnalysisTask, DatasetAnalysisTask
from .artefact import Target
from .config import Config
from .task import Task


@contextmanager
def new_figure(dest: Path | list[Path], bbox_inches="tight", **kwargs):
    """
    Helper context manager to produce a new figure
    """
    kwargs.setdefault("constrained_layout", True)
    fig = plt.figure(**kwargs)
    yield fig
    if isinstance(dest, Path):
        dest = [dest]
    for path in dest:
        fig.savefig(path, bbox_inches=bbox_inches)
    plt.close(fig)


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
    plt.close(facet.figure)


@dataclass
class PlotTargetConfig(Config):
    """
    Configuration keys for a single plot target.
    """
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None


@dataclass
class PlotTaskConfig(Config):
    """
    Base class for plot task configurations.

    Note that it is not mandatory to use this base as the value of
    PlotTask.task_config_class.
    When the PlotTask configuration is a subclass of PlotTaskConfig,
    it is possible to customize plot outputs by key/value parameters.
    """
    target_config: List[PlotTargetConfig] = field(default_factory=list)


class PlotTarget(Target):
    """
    Target pointing to a plot path.

    The output path depends on whether the plot is associated to the session or to
    a single dataset.
    """
    def __init__(self, task: Task, output_id: str = "plot", **kwargs):
        kwargs.setdefault("ext", task.analysis_config.plot.plot_output_format)
        super().__init__(task, output_id, **kwargs)


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

        with PlotTaskMixin.rc_mutex:
            with rc_context():
                # Use default theme as base
                sns.set_theme()
                with sns.plotting_context(rc=rc_override):
                    self.run_plot()

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
        kwargs.setdefault("loc", "center")
        owner.legend(legend.legend_handles,
                     map(lambda t: t.get_text(), legend.texts),
                     bbox_to_anchor=(0, 1.02),
                     ncols=4,
                     **kwargs)

    def baseline_slice(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Extract the baseline cross section of the dataframe.

        The baseline must be specified in the analysis configuration.
        """
        baseline_sel = self.analysis_config.baseline
        if baseline_sel is None:
            self.logger.error("Missing baseline selector in analysis configuration")
            raise ValueError("Invalid Configuration")

        if type(baseline_sel) == dict:
            # If we have the 'instance' parameter, replace it with the corresponding
            # dataset_gid
            if "instance" in baseline_sel:
                name = baseline_sel["instance"]
                for b in self.session.all_benchmarks():
                    if b.config.instance.name == name:
                        baseline_sel["dataset_gid"] = b.config.g_uuid
                        del baseline_sel["instance"]
                        break
                else:
                    self.logger.error("Invalid 'instance' value in baseline configuration")
                    raise ValueError("Invalid configuration")
            baseline = df.filter(**baseline_sel)
        else:
            # Expect a UUID
            baseline = df.filter(dataset_id=baseline_sel)

        if len(baseline) >= 1 and len(baseline["dataset_id"].unique()) == 1:
            self.logger.error("Invalid baseline specifier %s", baseline_sel)
            raise ValueError("Invalid configuration")
        return baseline

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
