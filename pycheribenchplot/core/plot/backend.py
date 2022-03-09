import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..analysis import AnalysisConfig
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
    def __init__(self, layout: list[list[str]], subplots: dict[str, "BenchmarkSubPlot"]):
        m = np.atleast_2d(np.array(layout, dtype=object))
        # Normalized mosaic list to be at least 2D
        self.layout = m.tolist()
        self.subplots = subplots

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
