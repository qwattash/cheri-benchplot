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
    def render(self, view: DataView, cell: CellData, surface: "Surface", **kwargs):
        ...


class GridLayout:
    """
    Base class abstracting the layout of a plot surface.
    This manages the indexing of the plot cells and holds the
    data that is to be rendered at a given position.
    Some surfaces may not support some layouts and instead flatten the layout.
    """
    def __init__(self, nrows, ncols):
        self._layout = np.full([nrows, ncols], None)

    @property
    def shape(self):
        return self._layout.shape

    def __iter__(self):
        # r, c = np.meshgrid(self._layout.shape)
        # rc = np.dstack([r.ravel(), c.ravel()])
        for row in self._layout:
            yield row

    def next_cell(self) -> typing.Optional[np.ndarray]:
        """Find next empty position"""
        for i, stride in enumerate(self._layout):
            for j, cell in enumerate(stride):
                if cell is None:
                    return np.array((i, j))
        return None

    def set_cell(self, row: int, col: int, cell: CellData):
        self._layout[(row, col)] = cell


class ColumnLayout(GridLayout):
    def __init__(self, nrows):
        super().__init__(nrows, 1)


class RowLayout(GridLayout):
    def __init__(self, ncols):
        super().__init__(1, ncols)


class Surface(ABC):
    """
    The surface manages the plot rendering.
    Each surface has a configured layout to manage subplot axes.
    Each plot rendering is wrapped by a Cell, which represents a set of subplot axes.
    Each cell can display multiple views that wrap a plot rendering method.
    """
    @dataclass
    class DrawContext:
        title: str
        dest: Path
        row: int
        col: int

    def __init__(self):
        self.logger = new_logger(str(self))
        self._layout = None
        self._renderers = {}
        self.config = AnalysisConfig()

    def set_layout(self, layout: GridLayout):
        """
        Set the layout to use for the surface.
        Note: this must be called before any cell is added to the surface.
        """
        self._layout = layout

    def set_config(self, config: AnalysisConfig):
        self.config = config

    def next_cell(self, cell: CellData):
        """
        Add the given cell to the next position available in the layout.
        """
        i, j = self._layout.next_cell()
        cell.set_surface(self)
        self._layout.set_cell(i, j, cell)

    def set_cell(self, row: int, col: int, cell: CellData):
        """
        Set the given data to plot on a cell
        """
        cell.set_surface(self)
        self._layout.set_cell(row, col, cell)

    @abstractmethod
    def make_cell(self, **kwargs) -> CellData:
        """Cell factory function"""
        ...

    def _make_draw_context(self, title, dest, **kwargs):
        return self.DrawContext(title, dest, row=0, col=0, **kwargs)

    def _finalize_draw_context(self, ctx):
        pass

    def get_renderer(self, data_view: DataView) -> ViewRenderer:
        try:
            return self._renderers[data_view.key]()
        except KeyError:
            self.logger.debug("Skipping data view %s unsupported by %s", data_view.key, self)
            return None

    def _draw_cell(cell, ctx):
        cell.validate()
        cell.draw(ctx)

    def _draw_split(self, title, dest):
        base = dest.parent / "split"
        base.mkdir(exist_ok=True)
        stem = dest.stem

        for ri, row in enumerate(self._layout):
            for ci, cell in enumerate(row):
                cell_dest = base / dest.with_stem(f"{stem}-{ri}-{ci}").name
                ctx = self._make_draw_context(title, cell_dest)
                ctx.row = 0
                ctx.col = 0
                try:
                    cell.validate()
                    cell.draw(ctx)
                finally:
                    self._finalize_draw_context(ctx)

    def _draw_combined(self, title, dest):
        ctx = self._make_draw_context(title, dest)
        try:
            for ri, row in enumerate(self._layout):
                for ci, cell in enumerate(row):
                    ctx.row = ri
                    ctx.col = ci
                    cell.validate()
                    cell.draw(ctx)
        finally:
            self._finalize_draw_context(ctx)

    def draw(self, title: str, dest: Path):
        """Draw all the registered views into the surface."""
        self.logger.debug("Drawing...")

        if self.config.split_subplots:
            self._draw_split(title, dest)
        else:
            self._draw_combined(title, dest)

    def __str__(self):
        return self.__class__.__name__
