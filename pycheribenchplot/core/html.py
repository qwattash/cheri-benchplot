import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape

from .plot import CellData, Surface, TableDataView, ViewRenderer


class HTMLSurface(Surface):
    """
    Draw plots to static HTML files.
    """

    env = Environment(loader=PackageLoader("pycheribenchplot"), autoescape=select_autoescape())
    default_template = "base_layout.html"

    @dataclass
    class DrawContext(Surface.DrawContext):
        rendered_cells: list = field(default_factory=list)

    def __init__(self, template=None):
        super().__init__()
        if template is None:
            template = self.default_template
        self.template = self.get_template(template)
        self._renderers = {"table": HTMLTableRenderer}

    def get_template(self, name: str):
        return self.env.get_template(name)

    def _make_draw_context(self, title, dest, **kwargs):
        out_cells = np.full(self._layout.shape, "")
        return super()._make_draw_context(title, dest, rendered_cells=out_cells, **kwargs)

    def _finalize_draw_context(self, ctx):
        rows, cols = self._layout.shape
        html = self.template.render(nrows=rows, ncols=cols, title=ctx.title, cell_layout=ctx.rendered_cells)
        with open(ctx.dest.with_suffix(".html"), "w+") as dest_file:
            dest_file.write(html)

    def make_cell(self, **kwargs):
        return HTMLPlotCell(**kwargs)


class HTMLPlotCell(CellData):
    """
    Base HTML dataset rendering class. Add wrapper functions to allow running the rendering
    step within the jinja templates, so that we can access cell and view properties within the
    template if needed.
    """
    def draw(self, ctx):
        html = []
        for view in self.views:
            r = self.surface.get_renderer(view)
            html.append(r.render(view, self, self.surface))
        ctx.rendered_cells[ctx.row][ctx.col] = "".join(html)


class HTMLTableRenderer(ViewRenderer):
    """
    Render a table with HTML
    """
    def render(self, view: TableDataView, cell, surface):
        assert isinstance(view, TableDataView), "Table renderer can only handle TableDataView"

        hide_cols = set(view.df.columns) - set(view.columns)
        table_template = surface.get_template("table.html")
        styler = view.df[view.columns].reset_index().style
        if view.colormap is not None:
            color_col = view.color_col if view.color_col else view.df.index.names

            def _apply_colormap(view_df):
                color = view_df[color_col].map(lambda color_key: view.colormap.get_color(color_key)).map(
                    lambda color: f"background-color: {color};" if color else "")
                cp = view_df.copy()
                cp.loc[:, :] = np.tile(color.to_numpy(), (len(view_df.columns), 1)).transpose()
                return cp

            styler.apply(_apply_colormap, axis=None)
        # styler.format(view.fmt, precision=3)
        styler.set_table_attributes('class="table table-striped table-responsive"')
        table_html = styler.hide_index().render()
        return table_template.render(cell_num=cell.cell_id, table=table_html)
