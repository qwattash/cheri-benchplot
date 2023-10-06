"""
General purpose matplotlib helpers
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from uuid import UUID

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc_context

from .analysis import AnalysisTask, DatasetAnalysisTask
from .artefact import LocalFileTarget
from .config import Config, InstanceConfig
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


class PlotTarget(LocalFileTarget):
    """
    Target pointing to a plot path.

    The output path depends on whether the plot is associated to the session or to
    a single dataset.
    """
    def __init__(self, task: Task, prefix: str = "", ext: str | None = None):
        super().__init__(task, prefix=prefix, ext=ext)
        if ext:
            self._plot_ext = [ext]
        else:
            self._plot_ext = self._task.analysis_config.plot.plot_output_format
        # Normalize extensions to start with a '.', this is needed by pathlib
        self._plot_ext = [f".{ext}" for ext in self._plot_ext if not ext.startswith(".")]

    def _session_paths(self):
        return [self._task.session.get_plot_root_path() / self._file_name]

    def _benchmark_paths(self):
        return [self._task.benchmark.get_plot_path() / self._file_name]

    def paths(self):
        """
        Generate multiple plots target paths with different extensions, as configured.
        """
        files = []
        for base_path in super().paths():
            for ext in self._plot_ext:
                files.append(base_path.with_suffix(ext))
        return files


class PlotTaskMixin:
    """
    Base class for plotting tasks.
    Plot tasks generate one or more plots from some analysis task data.
    These are generally the public-facing tasks that are selected in the analysis
    configuration.
    Each plot task is responsible for setting up a figure and axes.
    """
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

    def get_instance_config(self, g_uuid: UUID) -> InstanceConfig:
        """
        Helper to retreive an instance configuration for the given g_uuid.
        """
        gid_column = self.session.benchmark_matrix[g_uuid]
        return gid_column[0].config.instance

    def g_uuid_to_label(self, g_uuid: UUID | str) -> str:
        """
        Helper that maps group UUIDs to a human-readable label that describes the instance
        """
        if isinstance(g_uuid, str):
            g_uuid = UUID(g_uuid)
        instance_config = self.get_instance_config(g_uuid)
        return instance_config.name

    def _run_with_plot_sandbox(self):
        with rc_context():
            self.run_plot()

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
