import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import AnalysisConfig
from ..util import new_logger
from .data_view import *


class ViewRenderer(ABC):
    """
    Base class that renders plots.
    This must be specialized for each supported plot type by each surface
    """
    @abstractmethod
    def render(self, view: DataView, cell: CellData):
        ...


class Mosaic:
    """
    Wrapper around subplot mosaic lists
    """
    def __init__(self,
                 layout: typing.List[typing.List[str]] = None,
                 subplots: typing.Dict[str, "BenchmarkSubPlot"] = None):
        if subplots is None:
            subplots = {}
        if layout is not None:
            layout = np.atleast_2d(np.array(layout, dtype=object))
        # Normalized mosaic list to be at least 2D
        self._layout = layout
        self.subplots = subplots

    @property
    def layout(self):
        if self._layout is None:
            return np.empty((1, 0))
        return self._layout

    def allocate(self, name: str, nrows: int, ncols: int):
        """
        Assign nrows,ncols of space to the given plot name.
        """
        if self._layout is None:
            self._layout = np.full((nrows, ncols), name)
            return
        # pad = [[before_ax_0, before_ax_1], [after_ax_0, after_ax_1]]
        pad_before = [0, 0]
        pad_after = [0, 0]
        if self._layout.shape[1] < ncols:
            pad_after[1] = ncols - self._layout.shape[1]
        layout = np.pad(self._layout, [pad_before, pad_after], mode="edge")
        chunk = np.full((nrows, ncols), name)
        self._layout = np.concatenate([layout, chunk])

    def extract(self, name: str):
        """
        Fetch mosaic subset for given subplot
        """
        r, c = np.where(self._layout == name)
        cut = self._layout[min(r):max(r) + 1, min(c):max(c) + 1]
        blank = np.where(cut != name)
        cut[blank] = "BLANK"
        return cut

    @property
    def shape(self):
        """
        A mosaic is always at least 2-D. It must be a square matrix so we can
        extract rows and columns.
        Returns (rows, cols)
        """
        m = np.atleast_2d(np.array(self.layout, dtype=object))
        return m.shape

    def __iter__(self):
        yield from self.subplots.values()


class FigureManager(ABC):
    """
    The figure manager handles the rendering of a group of subplots..
    Each plot rendering is wrapped by a Cell, which represents a set of subplot axes.
    Each cell can display multiple views that wrap a plot rendering method.
    """
    def __init__(self, config: AnalysisConfig):
        self.logger = new_logger(str(self))
        self.config = config
        self._renderers = {}

    @abstractmethod
    def allocate_cells(self, mosaic: Mosaic):
        """
        Allocate cell data for each entry in the mosaic and setup the figure.
        It is passed a mosaic of subplots. The subplots in the mosaic are assigned a cell.
        """
        ...

    def draw(self, mosaic: Mosaic, title: str, dest: Path):
        """Draw all the registered views into the surface."""
        self.logger.debug("Drawing...")

    def __str__(self):
        return self.__class__.__name__
