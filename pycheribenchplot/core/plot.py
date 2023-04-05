"""
General purpose matplotlib helpers
"""
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt

from .analysis import AnalysisTask
from .config import AnalysisConfig, Config
from .task import Target


@contextmanager
def new_figure(dest: Path | list[Path], **kwargs):
    fig = plt.figure(constrained_layout=True, **kwargs)
    yield fig
    if isinstance(dest, Path):
        dest = [dest]
    for path in dest:
        fig.savefig(path)
    plt.close(fig)


class PlotTarget(Target):
    """
    Target pointing to a plot path
    """
    def __init__(self, path):
        self.path = path


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
