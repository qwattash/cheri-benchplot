"""
General purpose matplotlib helpers
"""
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt


@contextmanager
def new_figure(dest: Path | list[Path], **kwargs):
    fig = plt.figure(constrained_layout=True, **kwargs)
    yield fig
    if isinstance(dest, Path):
        dest = [dest]
    for path in dest:
        fig.savefig(path)
    plt.close(fig)
