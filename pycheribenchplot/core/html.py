import logging

import numpy as np
import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape

from .plot import CellData, DataView, PlotError, Surface


class HTMLSurface(Surface):
    """
    Draw plots to static HTML files.
    """

    env = Environment(loader=PackageLoader("pycheribenchplot"), autoescape=select_autoescape())
    default_template = "base_layout.html"

    def __init__(self, template=None):
        super().__init__()
        if template is None:
            template = self.default_template
        self.template = self.get_template(template)

    def get_template(self, name: str):
        return self.env.get_template(name)

    def draw(self, title, dest):
        self.logger.debug("Drawing...")
        rows, cols = self._layout.shape
        html = self.template.render(nrows=rows, ncols=cols, title=title, cell_layout=self._layout)
        with open(dest.with_suffix(".html"), "w+") as dest_file:
            dest_file.write(html)

    def make_cell(self, **kwargs):
        return HTMLPlotCell(**kwargs)

    def make_view(self, plot_type, **kwargs):
        if plot_type == "table":
            return HTMLTable(**kwargs)


class HTMLPlotCell(CellData):
    """
    Base HTML dataset rendering class. Add wrapper functions to allow running the rendering
    step within the jinja templates, so that we can access cell and view properties within the
    template if needed.
    """
    def to_html(self):
        html = []
        for view in self.views:
            html.append(view.render(self, self.surface))
        return "".join(html)


class HTMLTable(DataView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.colormap and self.color_col is None:
            self.color_col = self.df.index.names

    def _apply_colormap(self, view_df):
        color = view_df[self.color_col].map(lambda color_key: self.colormap.get_color(color_key)).map(
            lambda color: f"background-color: {color};" if color else "")
        cp = view_df.copy()
        cp.loc[:, :] = np.tile(color.to_numpy(), (len(view_df.columns), 1)).transpose()
        return cp

    def render(self, cell, surface):
        """
        Render the dataframe as an HTML table.
        """
        hide_cols = set(self.df.columns) - set(self.yleft)
        table_template = surface.get_template("table.html")
        styler = self.df[self.yleft].reset_index().style
        if self.colormap is not None:
            styler.apply(lambda df: self._apply_colormap(df), axis=None)
        styler.format(self.fmt, precision=3)
        styler.set_table_attributes('class="table table-striped table-responsive"')
        table_html = styler.hide_index().render()
        return table_template.render(cell_num=cell.cell_id, table=table_html)
