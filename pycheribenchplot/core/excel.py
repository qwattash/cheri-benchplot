import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors

from .plot import CellData, DataView, PlotError, PlotUnsupportedError, Surface


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
        raise PlotUnsupportedError(f"Plot type {plot_type} is not supported by the spreadsheet surface")


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
        if self.yright:
            surface.logger.warning("Excel table does not support right Y axis")
        if self.colormap:
            surface.logger.warning("Excel table does not support per-sample colormap")

        title = Path(cell.title).name
        sheet_name = re.sub("[:]", "", title)
        self.df.to_excel(excel_writer, sheet_name=sheet_name, columns=self.yleft, index=True, float_format="%.2f")
        book = excel_writer.book
        sheet = excel_writer.sheets[sheet_name]
        # Set column colors and width
        nindex = len(self.df.index.names)
        if cell.legend_map:
            for idx, column in enumerate(self.yleft):
                col_idx = nindex + idx
                color = cell.legend_map.get_color(column)
                xfmt = book.add_format({"bg_color": mcolors.to_hex(color), "border_color": "#000000", "border": 1})
                # Assume width 10 is enough for %.2f numbers, may want something more robust though
                # if pd.api.types.is_float_dtype(self.df.dtypes[column]):
                #     xfmt.set_num_format("0.00")
                sheet.set_column(col_idx, col_idx, 10, xfmt)
        max_header_size = 0
        for idx, column in enumerate(self.yleft):
            col_idx = nindex + idx
            # Rewrite the pandas-generated header to use a custom format
            hdr_format = book.add_format({
                "bold": True,
                "text_wrap": False,
                "valign": "center",
                "align": "center",
                "rotation": 90,
                "border_color": "#000000",
                "border": 1
            })
            if cell.legend_map:
                color = cell.legend_map.get_color(column)
                hdr_format.set_bg_color(mcolors.to_hex(color))
            sheet.write(0, col_idx, column, hdr_format)
            if len(column) > max_header_size:
                max_header_size = len(column)
        # XXX replace with something more robust
        sheet.set_row(0, max_header_size * 8)

        # Resize index columns to fit text
        for idx, level in enumerate(self.df.index.names):
            width = self.df.index.get_level_values(level).map(str).map(len).max()
            width = max(width, len(level)) + 2  # some padding
            sheet.set_column(idx, idx, width)

        # Freeze index and headers
        sheet.freeze_panes(1, len(self.df.index.names) - 1)
