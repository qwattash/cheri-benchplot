import logging

import numpy as np
import pandas as pd

from .plot import DataView, CellData, Surface, PlotError


class SpreadsheetSurface(Surface):
    """
    Draw plots to static HTML files.
    """
    def output_file_ext(self):
        return "xls"

    def draw(self, title, dest):
        self.logger.debug("Drawing...")
        with open(dest, "wb+") as dest_file:
            for cell in self._layout.ravel():
                cell.to_excel(dest_file)

    def make_cell(self, **kwargs):
        return SpreadsheetPlotCell(**kwargs)

    def make_view(self, plot_type, **kwargs):
        if plot_type == "table":
            return SpreadsheetTable(**kwargs)


class SpreadsheetPlotCell(CellData):
    """
    Base HTML dataset rendering class. Add wrapper functions to allow running the rendering
    step within the jinja templates, so that we can access cell and view properties within the
    template if needed.
    """
    def to_excel(self, fd):
        name = self.title
        if len(self.views) > 1:
            self.logger.warning("Only a single plot view is supported for each excel surface cell")
        if len(self.views):
            self.views[0].render(self, self.surface, excel_fd=fd)


class SpreadsheetTable(DataView):
    def render(self, cell, surface, excel_fd):
        """
        Render the dataframe as a table in an excel sheet.
        """
        self.df.to_excel(excel_fd, sheet_name=cell.title)
