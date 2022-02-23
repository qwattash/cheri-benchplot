import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from openpyxl.styles import Border, PatternFill, Side
from openpyxl.utils import get_column_letter

from .backend import Surface, ViewRenderer
from .data_view import CellData, DataView, TableDataView


class SpreadsheetSurface(Surface):
    """
    Draw plots to static HTML files.
    """
    @dataclass
    class DrawContext(Surface.DrawContext):
        writer: pd.ExcelWriter

    def __init__(self):
        super().__init__()
        self._renderers = {"table": SpreadsheetTableRenderer}

    def _make_draw_context(self, title, dest, **kwargs):
        writer = pd.ExcelWriter(dest.with_suffix(".xlsx"), mode="w", engine="openpyxl")
        return super()._make_draw_context(title, dest, writer=writer, **kwargs)

    def _finalize_draw_context(self, ctx):
        if len(ctx.writer.sheets):
            ctx.writer.close()
        else:
            del ctx.writer

    def make_cell(self, **kwargs):
        return SpreadsheetPlotCell(**kwargs)


class SpreadsheetPlotCell(CellData):
    """
    Base HTML dataset rendering class. Add wrapper functions to allow running the rendering
    step within the jinja templates, so that we can access cell and view properties within the
    template if needed.
    """
    def draw(self, ctx: SpreadsheetSurface.DrawContext):
        for view in self.views:
            r = self.surface.get_renderer(view)
            if not r:
                continue
            if len(self.views) > 1:
                self.surface.logger.warning("Only a single plot view is supported for each excel surface cell")
            r.render(view, self, self.surface, excel_writer=ctx.writer)
            break


class SpreadsheetTableRenderer(ViewRenderer):
    """
    Render table in a spreadsheet.
    """
    def _get_cell_color_styles(self, legend_info):
        fill_styles = {}
        if legend_info is None:
            return fill_styles

        for key, color in legend_info.color_items():
            # openpyxl uses aRGB colors while matplotlib uses RGBa
            # we need to swap the alpha here
            rgba_hex = [int(round(c * 255)) for c in color]
            argb_hex = rgba_hex[-1:] + rgba_hex[0:3]
            hexcolor = "".join(map(lambda c: f"{c:02x}", argb_hex))
            style = PatternFill(start_color=hexcolor, end_color=hexcolor, fill_type="solid")
            fill_styles[key] = style
        return fill_styles

    def render(self, view: TableDataView, cell, surface, excel_writer):
        """
        Render the dataframe as a table in an excel sheet.
        """
        assert isinstance(view, TableDataView), "Table renderer can only handle TableDataView"

        title = Path(cell.title).name
        sheet_name = re.sub("[:]", "", title)
        view.df.to_excel(excel_writer, sheet_name=sheet_name, columns=view.columns, index=True)
        book = excel_writer.book
        sheet = excel_writer.sheets[sheet_name]

        fill_styles = self._get_cell_color_styles(view.legend_info)
        nindex = len(view.df.index.names)
        nheader = len(view.df.columns.names)

        side_thin = Side(border_style="dashed", color="000000")
        side_medium = Side(border_style="thin", color="000000")
        side_thick = Side(border_style="thick", color="000000")
        border_thin = Border(top=side_thin, bottom=side_thin, left=side_thin, right=side_thin)
        border_medium = Border(top=side_thin, bottom=side_thin, left=side_medium, right=side_thin)
        border_thick = Border(top=side_thin, bottom=side_thin, left=side_thick, right=side_thin)

        # Set column colors and width
        # Note that this will also adjust the column headers width
        for (idx, column), xcells in zip(enumerate(view.columns),
                                         sheet.iter_cols(min_row=nheader + 1, min_col=nindex + 1)):
            col_idx = nindex + idx + 1
            col_width = max(map(len, column))
            # We introduce a thicker border for column groups at the first two column index levels
            if len(view.columns.names) > 1 and idx > 0:
                if len(view.columns.names) > 2 and column[1] != view.columns[idx - 1][1]:
                    # generally the 'delta' or 'aggregate' levels
                    cell_border = border_medium
                elif column[0] != view.columns[idx - 1][0]:
                    # generally the 'metric' level
                    cell_border = border_thick
                else:
                    cell_border = border_thin
            else:
                cell_border = border_thin

            for xcell in xcells:
                if pd.api.types.is_float_dtype(type(xcell.value)):
                    cell_width = len(f"{xcell.value:.2f}")
                    xcell.number_format = "0.00"
                else:
                    cell_width = len(str(xcell.value))
                if view.legend_info:
                    xcell.fill = fill_styles[column]
                col_width = max(col_width, cell_width)
                xcell.border = cell_border
            sheet.column_dimensions[get_column_letter(col_idx)].width = col_width + 2

        # # Resize index columns to fit text
        for idx, level in enumerate(view.df.index.names):
            col_idx = idx + 1
            width = view.df.index.get_level_values(level).map(str).map(len).max()
            if level is None:
                level = ""
            width = max(width, len(level)) + 2  # some padding
            sheet.column_dimensions[get_column_letter(col_idx)].width = width

        # Freeze index and headers, +1 is required because sheet columns start from 1, not zero.
        sheet.freeze_panes = sheet.cell(row=len(view.df.columns.names) + 1, column=len(view.df.index.names) + 1)
