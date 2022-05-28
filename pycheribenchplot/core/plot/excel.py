import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors

from ..util import timing
from .backend import FigureManager, Mosaic, ViewRenderer
from .data_view import CellData, DataView, TableDataView


class ExcelFigureManager(FigureManager):
    """
    Draw plots to static HTML files.
    """
    def __init__(self, config):
        super().__init__(config)
        self.writer = None

    def allocate_cells(self, mosaic: Mosaic):
        for subplot in mosaic:
            subplot.cell = ExcelCellData(title=subplot.get_cell_title())
            subplot.cell.figure_manager = self

    def draw(self, mosaic, title, dest):
        super().draw(mosaic, title, dest)
        if self.config.split_subplots:
            self.logger.warning("Split subplots unsupported for excel figures")
        with pd.ExcelWriter(dest.with_suffix(".xlsx"), mode="w", engine="xlsxwriter") as writer:
            for subplot in mosaic:
                subplot.cell.writer = writer
                subplot.cell.draw()


class ExcelCellData(CellData):
    """
    Base HTML dataset rendering class. Add wrapper functions to allow running the rendering
    step within the jinja templates, so that we can access cell and view properties within the
    template if needed.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._renderers = {"table": SpreadsheetTableRenderer}
        # Set at drawing time
        self.writer = None

    def draw(self):
        if len(self.views) > 1:
            self.figure_manager.logger.warning("Only a single plot view is supported for each excel sheet")
        try:
            view = self.views[0]
        except IndexError:
            return
        r = self.get_renderer(view)
        r.render(view, self)


class SpreadsheetTableRenderer(ViewRenderer):
    """
    Render table in a spreadsheet.
    """
    def _gen_fill_styles(self, view, book):
        """
        Generate a dataframe with the same index and columns as the view dataframe, containing the
        colors for each cell.
        """
        default = book.add_format()
        cell_fmt = pd.DataFrame(None, index=view.df.index, columns=view.df.columns)
        legend_df = view.legend_info.info_df

        color_fmt = {}
        for color in legend_df["colors"]:
            fmt = book.add_format({"fg_color": mcolors.to_hex(color)})
            fmt.set_border(1)  # solid
            color_fmt[tuple(color)] = fmt

        if view.legend_axis == "index":
            # Align the legend to the data frame on the legend levels present in the legend index
            tmp_df, _ = legend_df.align(view.df, axis=0, join="inner")
            if tmp_df["colors"].isna().any():
                raise RuntimeError("Something went wrong with the legend join operation")
            tmp_df["colors"] = tmp_df["colors"].map(lambda c: color_fmt[tuple(c)])
            cell_fmt.loc[:, :] = np.repeat(tmp_df["colors"].values, len(cell_fmt.columns)).reshape(cell_fmt.shape)
        elif view.legend_axis == "column":
            # Just assign the columns
            tmp_df, _ = legend_df.T.align(view.df, axis=1, join="inner")
            for col in tmp_df.columns:
                color = tmp_df[col]["colors"]
                fmt = color_fmt[tuple(color)]
                cell_fmt.loc[:, col] = fmt
        elif view.legend_axis == "cell":
            raise NotImplementedError("Cell coloring not yet supported")
        else:
            raise ValueError(f"Invalid legend_axis {view.legend_axis}")
        return cell_fmt

    def _set_cell_styles(self, view, book, sheet):
        if view.legend_info is None:
            return

        cell_fill_df = self._gen_fill_styles(view, book)
        base_row = len(view.df.columns.names)
        base_col = len(view.df.index.names)
        for i, index in enumerate(view.df.index):
            for j, col in enumerate(view.columns):
                # We are forced to rewrite the value as well
                v = view.df.loc[index, col]
                if isinstance(v, float):
                    v = f"{v:.2f}"
                fmt = cell_fill_df.loc[index, col]
                sheet.write(base_row + i + 1, base_col + j, v, fmt)

    def _fixup_column_size(self, view, book, sheet):
        """
        Adjust column sizes based on content
        """
        def fixup_cell(value):
            if isinstance(value, float):
                value = f"{value:.2f}"
            else:
                value = str(value)
            return len(value)

        fixup_index = view.df.index.to_frame().applymap(fixup_cell).max()
        for i, (col, width) in enumerate(fixup_index.items()):
            width = max(width, len(col))
            sheet.set_column(i, i, width)

        base_column = len(view.df.index.names)
        fixup = view.df[view.columns].applymap(fixup_cell).max()
        for i, (col, width) in enumerate(fixup.items()):
            if isinstance(col, tuple):
                hdr_width = max(map(len, col))
            else:
                hdr_width = len(col)
            width = max(width, hdr_width)
            sheet.set_column(base_column + i, base_column + i, width)

    def render(self, view: TableDataView, cell):
        """
        Render the dataframe as a table in an excel sheet.
        """
        assert isinstance(view, TableDataView), "Table renderer can only handle TableDataView"
        excel_writer = cell.writer
        book = excel_writer.book

        title = Path(cell.title).name
        sheet_name = re.sub("[:]", "", title)
        cell.figure_manager.logger.debug("Populate excel sheet %s", sheet_name)
        if len(sheet_name) > 31:
            sheet_name = f"Sheet {len(book.worksheets())}"
        with timing("Write excel table", logger=cell.figure_manager.logger):
            view.df.to_excel(excel_writer, sheet_name=sheet_name, columns=view.columns, index=True)
        sheet = book.get_worksheet_by_name(sheet_name)
        # Add sheet title
        # sheet.cell(0,0,sheet_name)
        # Columns size fixup
        self._fixup_column_size(view, book, sheet)
        self._set_cell_styles(view, book, sheet)
        sheet.freeze_panes(len(view.df.columns.names) + 1, len(view.df.index.names))
