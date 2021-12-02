import re
from pathlib import Path

import numpy as np
import pandas as pd

from .plot import CellData, DataView, PlotError, Surface


class SpreadsheetSurface(Surface):
    """
    Draw plots to static HTML files.
    """
    def draw(self, title, dest):
        self.logger.debug("Drawing...")
        with pd.ExcelWriter(dest.with_suffix(".xlsx"), mode="w", engine="xlsxwriter") as writer:
            for row in self._layout:
                for cell in row:
                    cell.to_excel(writer)

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
    def to_excel(self, excel_writer):
        name = self.title
        if len(self.views) > 1:
            self.logger.warning("Only a single plot view is supported for each excel surface cell")
        if len(self.views):
            self.views[0].render(self, self.surface, excel_writer)


class SpreadsheetTable(DataView):
    def render(self, cell, surface, excel_writer):
        """
        Render the dataframe as a table in an excel sheet.
        """
        title = Path(cell.title).name
        sheet_name = re.sub("[:]", "", title)
        self.df.to_excel(excel_writer, sheet_name=sheet_name, columns=self.yleft, index=True)
        sheet = excel_writer.sheets[sheet_name]
        # Estimate columns witdh
        # render_cols = list(self.df.index.names) + list(self.yleft)
        # width_list = map(len, render_cols)
        # for i, width in enumerate(width_list):
        #     sheet.set_column(i, i, width)
        # Freeze index and headers
        sheet.freeze_panes(1, len(self.df.index.names) - 1)
