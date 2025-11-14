from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from ..core.analysis import AnalysisTask
from ..core.artefact import Target, ValueTarget
from ..core.config import Config
from ..core.plot import PlotTarget, PlotTask, new_facet, new_figure
from ..core.task import dependency, output
from ..core.tvrs import TVRSParamsMixin, TVRSPlotConfig
from ..pmc.pmc_exec import PMCExec
from .ingest import IngestQPSData


@dataclass
class QPSPmcConfig(QPSPlotConfig):
    #: Filter only the given subset of columns/counters
    metrics_filter: Optional[List[str]] = None


class QPSPerfCountersPlot(TVRSParamsMixin, PlotTask):
    """
    Generate mixed qps/pmc metrics when hardware performance counters data is available.
    """
    task_config_class = QPSPmcConfig
    task_namespace = "qps"
    task_name = "pmc-metrics"
    public = True

    derived_metrics = {
        "ex_entry_per_msg": {
            "requires": ["executive_entry", "message_count"],
            "expr": (pl.col("executive_entry") / pl.col("message_count"))
        },
        "ex_entry_per_byte": {
            "requires": ["executive_entry", "message_count", "resp_size"],
            "expr": (pl.col("executive_entry") / (pl.col("message_count") * pl.col("resp_size")))
        }
    }

    @dependency(optional=True)
    def pmc(self):
        for bench in self.session.all_benchmarks():
            task = bench.find_exec_task(PMCExec)
            yield task.counter_data.get_loader()

    @dependency
    def qps_data(self):
        return LoadQPSData(self.session, self.analysis_config)

    @output
    def qps_metrics(self):
        return PlotTarget(self, "metrics")

    @output
    def qps_table(self):
        return PlotTarget(self, "tbl-metrics")

    def run_plot(self):
        if self.pmc is None:
            self.logger.info("Skip task %s: missing PMC data", self)
            return

        # Merge the counter data from everywhere
        pmc_df = pl.concat([loader.df.get() for loader in self.pmc]).with_columns(pl.col("iteration").cast(pl.Int32))
        assert len(pmc_df.filter(iteration=-1)) == 0, "Missing iteration number on PMC frame"
        # Join with the QPS data
        qps_df = self.qps_data.merged_df.get().with_columns(pl.col("iteration").cast(pl.Int32))
        assert len(qps_df.filter(iteration=-1)) == 0, "Missing iteration number on QPS frame"

        df = qps_df.join(pmc_df, on=["dataset_id", "iteration"])
        assert df.shape[0] == pmc_df.shape[0], "Unexpected dataframe shape change"
        assert df.shape[0] == qps_df.shape[0], "Unexpected dataframe shape change"

        # Generate derived metrics
        found_metrics = []
        for name, spec in self.derived_metrics.items():
            has_cols = True
            for required_column in spec["requires"]:
                if required_column not in df.columns:
                    self.logger.info("Skip derived metric %s: requires missing column %s", name, required_column)
                    has_cols = False
                    break
            if not has_cols:
                continue
            df = df.with_columns((spec["expr"]).alias(name))
            found_metrics.append(name)

        if not found_metrics:
            self.logger.info("Skipping plot, no data")
            return

        # XXX aggregate over iterations?
        ctx = self.make_param_context(df)
        ctx.derived_param_strcat("_flavor", ["variant", "runtime"], sep="/")
        ctx.melt(value_vars=found_metrics, variable_name="metric", value_name="value")
        ctx.relabel(default={"_flavor": "flavor/protection"})

        # Filter the counters based on configuration
        if self.config.metrics_filter:
            ctx.df = ctx.df.filter(pl.col("metric").is_in(self.config.metrics_filter))

        palette = ctx.build_palette_for("_flavor")
        self.logger.info("Generate QPS PMC metrics")
        with new_facet(self.qps_metrics.paths(),
                       ctx.df,
                       col=ctx.r.scenario,
                       row="metric",
                       sharex="col",
                       sharey="row",
                       margin_titles=True,
                       aspect=0.85) as facet:
            facet.map_dataframe(sns.barplot, x=ctx.r.target, y="value", hue=ctx.r._flavor, dodge=True, palette=palette)
            facet.add_legend()

        group_cols = [ctx.r.target, ctx.r.scenario, ctx.r._flavor, "metric"]
        ctx.df.group_by(group_cols).mean().write_csv(self.qps_table.single_path())
