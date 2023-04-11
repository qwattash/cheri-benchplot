"""
General purpose matplotlib helpers
"""
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rc_context

from .analysis import AnalysisTask
from .artefact import AnalysisFileTarget
from .config import AnalysisConfig, Config
from .task import Task


@contextmanager
def new_figure(dest: Path | list[Path], **kwargs):
    fig = plt.figure(constrained_layout=True, **kwargs)
    yield fig
    if isinstance(dest, Path):
        dest = [dest]
    for path in dest:
        fig.savefig(path)
    plt.close(fig)


class PlotTarget(AnalysisFileTarget):
    """
    Target pointing to a plot path
    """
    pass


class PlotTask(AnalysisTask):
    """
    Base class for plotting tasks.
    Plot tasks generate one or more plots from some analysis task data.
    These are generally the public-facing tasks that are selected in the analysis
    configuration.
    Each plot task is responsible for setting up a figure and axes.
    Note that the ID of the task is generated assuming that there is only one plot per session.
    """
    task_namespace = "analysis.plot"

    def __init__(self, session: "Session", analysis_config: AnalysisConfig, task_config: Config = None):
        super().__init__(session, analysis_config, task_config=task_config)

    def _plot_output(self, suffix: str = None) -> PlotTarget:
        if suffix:
            name = f"{self.task_id}-{suffix}.pdf"
        else:
            name = f"{self.task_id}.pdf"
        return PlotTarget(self.session.get_plot_root_path() / name)

    def run(self):
        with rc_context():
            self.run_plot()

    def run_plot(self):
        """
        Plot task body.

        This runs within a matplotlib RC parameter context, so that local
        RC params are not propagated.
        """
        raise NotImplementedError("Must override")
