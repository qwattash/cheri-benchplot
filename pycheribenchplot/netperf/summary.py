from dataclasses import dataclass
from typing import Optional

import polars as pl
import seaborn as sns

from ..core.config import config_field
from ..core.plot import PlotTarget, PlotTask, new_figure
from ..core.task import dependency
from ..core.tvrs import TVRSParamsMixin, TVRSTaskConfig


@dataclass
class NetperfPlotConfig(TVRSTaskConfig):
    tile_parameter: Optional[str] = config_field(None, desc="Parameter to use for the facet column tiles")


class NetperfSummaryPlot(TVRSParamsMixin, PlotTask):
    """
    Produce a box plot of the netperf metrics.
    """
    task_namespace = "netperf"
    task_name = "summary-plot"
    task_config_class = NetperfPlotConfig
    public = True

    # def _draw_box_metric(self, df, metric: str):
    #     """
    #     Produce the box plot for a given metric, we keep grouping by g_uuid.
    #     """
    #     with new_figure(self._plot_output(metric).path) as fig:
    #         ax = fig.subplots()
    #         # There may be multiple parameterization axes, join as strings for now
    #         # probably not the best.
    #         if self.session.parameter_keys:
    #             param_desc = df.index.to_frame()[self.session.parameter_keys].agg("-".join, axis=1)
    #         else:
    #             # Should be only one benchmark, get its name
    #             name = self.session.config.configurations[0].name
    #             param_desc = pd.Series(name, index=df.index)
    #         hue = df.index.get_level_values("dataset_gid").map(lambda gid: self.session.machine_configuration_name(gid))
    #         sns.boxplot(ax=ax, y=metric, x=param_desc, hue=hue, data=df, palette="pastel")

    # def dependencies(self):
    #     self.stats = NetperfStatsMergedParams(self.session, self.analysis_config)
    #     yield self.stats

    def run_plot(self):
        # df = self.stats.output_map["merged_df"].df
        # # Determine how many metrics and parameters we have
        # metrics = df.columns.unique("metric")
        # for m in metrics:
        #     self._draw_box_metric(df, m)
        pass

    # def outputs(self):
    #     df = self.stats.output_map["merged_df"].df
    #     for m in df.columns.unique("metric"):
    #         yield f"plot-{m}", self._plot_output(metric)
