from threading import Lock
from warnings import deprecated

import matplotlib as mpl

from .analysis import AnalysisTask, DatasetAnalysisTask, SliceAnalysisTask
from .artefact import Target
from .plot_util.theme import default_theme
from .task import Task


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

    # Note that this is necessary to serialize matplotlib rcParams overrides.
    # If we could defer this until the PlotGrid context it would probably
    # allow some more parallel processing of plot data.
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
        theme = default_theme | rc_override

        self.setup_plot()
        with PlotTaskMixin.rc_mutex:
            self.logger.debug("Override matplotlib rcParams: %s", rc_override)
            with mpl.rc_context(rc=theme):
                self.run_plot()

    def setup_plot(self):
        """
        Pre-process plot data.

        This runs outside the plot context but in parallel with other tasks.
        This should be used to prepare the dataframe used for plotting.
        """
        pass

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


@deprecated("Use SlicePlotTask instead")
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
