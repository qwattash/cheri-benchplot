import threading

import matplotlib as mpl

mpl.use("pdf")

from .analysis import BootstrapAnalysisTask, SliceAnalysisTask
from .artefact import Target
from .plot_grid import default_theme
from .task import Task

# Capture the original matplotlib global configurations
_global_rc_blueprint = dict(mpl.rcParams)


class ThreadLocalRcParams:
    """
    A transparent proxy wrapper that routes all Matplotlib rcParams
    interactions to thread-isolated dictionaries.
    """

    def __init__(self):
        self.__dict__["_local"] = threading.local()

    @property
    def _current_rc(self):
        # Dynamically initialize a fresh thread-local config
        if not hasattr(self._local, "config"):
            # Using mpl.RcParams ensures validation logic remains intact
            self._local.config = mpl.RcParams(_global_rc_blueprint)
        return self._local.config

    def __getattr__(self, name):
        return getattr(self._current_rc, name)

    def __setattr__(self, name, value):
        setattr(self._current_rc, name, value)

    def __delattr__(self, name):
        delattr(self._current_rc, name)

    # Route all dictionary operations to the thread-local instance
    def __getitem__(self, key):
        return self._current_rc[key]

    def __setitem__(self, key, value):
        self._current_rc[key] = value

    def __delitem__(self, key):
        del self._current_rc[key]

    def __contains__(self, key):
        return key in self._current_rc

    def __iter__(self):
        return iter(self._current_rc)

    def __len__(self):
        return len(self._current_rc)

    def __repr__(self):
        return repr(self._current_rc)


# monkey-patch matplotlib to swap out the global dictionary with the thread-local proxy
mpl.rcParams = ThreadLocalRcParams()


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


class PlotTask(BootstrapAnalysisTask, PlotTaskMixin):
    """
    Session-level plotting task.

    This task generates one or more plots that are unique within a session.
    This can be used to produce summary or aggregate plots from all the session datasets.
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
