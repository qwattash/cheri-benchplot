import logging

import numpy as np
import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape

from .plot import DataView, CellData, Surface, PlotError

class HTMLSurface(Surface):
    """
    Draw plots to static HTML files.
    """

    env = Environment(loader=PackageLoader("pycheribenchplot"),
                      autoescape=select_autoescape())
    default_template="base_layout.html"

    def __init__(self, template=None):
        super().__init__()
        if template is None:
            template = self.default_template
        self.template = self.env.get_template(template)

    def draw(self, title, dest):
        self.logger.debug("Drawing...")
        rows, cols = self.layout_shape
        html = self.template.render(nrows=rows, ncols=cols, title=title, cell_layout=self._layout)
        with open(dest, "w+") as dest_file:
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


class HTMLDataView(DataView):
    pass


class HTMLTable(HTMLDataView):
    def render(self, cell, surface):
        """
        Render the dataframe as an HTML table.
        """
        styler = self.df.style
        styler.set_table_attributes('class="table table-responsive"')
        return styler.render()
