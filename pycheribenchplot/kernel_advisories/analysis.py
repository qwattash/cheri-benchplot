from dataclasses import dataclass
from datetime import date

import polars as pl
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

from ..core.analysis import AnalysisTask
from ..core.artefact import DataFrameTarget, Target
from ..core.config import (Config, ConfigPath, config_field, validate_file_exists)
from ..core.plot import PlotTarget, PlotTask, new_figure
from ..core.task import SessionDataGenTask, dependency, output
from .ingest import IngestAdvisoriesTask

DEFAULT_EVENTS = {
    date(2021, 4, 13): "Added KASAN kernel option",
    date(2017, 10, 17): "Syzkaller FreeBSD support",
    date(2016, 5, 27): "First KASAN mention in a fix",
    date(2009, 10, 2): "User NULL mappings forbidden"
}


@dataclass
class AdvisoryHistoryConfig(Config):
    events: dict[date, str] = config_field(lambda: DEFAULT_EVENTS, help="Events to mark on the top X axis timeline")


class CheriBSDAdvisoriesHistory(PlotTask):
    """
    Generate plot of the distribution of advisories over time.
    """
    public = True
    task_namespace = "kernel-advisories"
    task_name = "advisories-timeline"
    task_config_class = AdvisoryHistoryConfig

    @dependency
    def sa(self):
        task = self.session.find_exec_task(IngestAdvisoriesTask)
        return task.advisories.get_loader()

    @output
    def timeline(self):
        return PlotTarget(self, "timeline")

    def run_plot(self):
        df = self.sa.df.get()
        df = df.with_columns(pl.col("date").str.to_datetime("%Y-%m-%d %H:%M:%S").dt.year().alias("Year"))

        total_col = "Total SA"
        memory_safety_col = "Memory safety SA"
        spatial_safety_col = "Spatial memory safety SA"

        cnt_memory_safety = pl.col("kind_memory_safety").sum().alias(memory_safety_col)
        cnt_temporal_safety = (pl.col("kind_memory_safety") & ~pl.col("kind_uaf")
                               & ~pl.col("kind_race")).sum().alias(spatial_safety_col)
        agg_tmp = df.group_by("Year").agg(pl.count().alias(total_col), cnt_memory_safety, cnt_temporal_safety)
        view = agg_tmp.unpivot(index=["Year"],
                               on=[total_col, memory_safety_col, spatial_safety_col],
                               variable_name="Category",
                               value_name="count")

        with new_figure(self.timeline.paths()) as fig:
            ax = fig.subplots()
            ax.grid(axis="y", linestyle="--")
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            sns.lineplot(ax=ax,
                         data=view,
                         x="Year",
                         y="count",
                         hue="Category",
                         palette="pastel",
                         markers=True,
                         style="Category")
            ax.set_ylabel("Number of advisories")

            event_years = [d.year for d in self.config.events.keys()]
            for year in event_years:
                ax.axvline(year, linestyle="-.", linewidth=1, color="gray")

            # Add the twin top X axis with history markers
            ax_t = ax.twiny()
            ax_t.grid(visible=False)
            ax_t.set_xlim(*ax.get_xlim())
            ax_t.set_xticks(event_years)
            ax_t.set_xticklabels(self.config.events.values(), rotation=-25, ha="right", size=8)


class CheriBSDAdvisoriesPie(PlotTask):
    """
    Generate pie chart with different mitigation combinations.

    Advisories mitigated by multiple CHERI features are accounted
    separately.
    """
    public = True
    task_namespace = "kernel-advisories"
    task_name = "mitigation-pie-chart"

    @dependency
    def sa(self):
        task = self.session.find_exec_task(IngestAdvisoriesTask)
        return task.advisories.get_loader()

    @output
    def pie(self):
        return PlotTarget(self, "mitigated-pie")

    def run_plot(self):
        pass


class CheriBSDAdvisoriesTables(AnalysisTask):
    """
    Generate summary tables CSV files from the main vulnerability analysis table.
    """
    public = True
    task_namespace = "kernel-advisories"
    task_name = "summary-tables"

    @dependency
    def sa(self):
        task = self.session.find_exec_task(IngestAdvisoriesTask)
        return task.advisories.get_loader()

    @output
    def mitigation_summary(self):
        return Target(self, "mitigated", ext="csv")

    @output
    def unmitigated_summary(self):
        return Target(self, "unmitigated", ext="csv")

    def _advisories_summary(self):
        df = self.sa.df.get()

        selectors = [
            pl.col("kind_memory_safety").sum().alias("all_memory_safety"),
            (~pl.col("mitigation_unmitigated")).sum().alias("all_mitigated"),
            # mitigated by pointer bounds but not involving sub-object bounds
            (pl.col("mitigation_ptr_bounds") & ~pl.col("mitigation_subobject")).sum().alias("ptr_bounds_only"),
            # mitigated by sub-object bounds
            pl.col("mitigation_subobject").sum().alias("ptr_subobject"),
            # mitigated by provenance or integrity
            (pl.col("mitigation_ptr_provenance") | pl.col("mitigation_ptr_integrity")).sum().alias("ptr_integrity"),
            # mitigated by temporal safety
            pl.col("mitigation_temporal").sum().alias("ptr_temporal"),
        ]
        stats = df.select(selectors)

        table = {
            "Category": [
                "Mitigated by pointer provenance and integrity",
                "Mitigated by bounds checking",
                "Mitigated by sub-object bounds checking",
                "Total mitigated",
                "Possibly mitigated by temporal safety",
                "Total mitigated with temporal safety",
            ],
            "Number of advisories": [
                stats["ptr_integrity"][0], stats["ptr_bounds_only"][0], stats["ptr_subobject"][0],
                stats["all_mitigated"][0] - stats["ptr_temporal"][0], stats["ptr_temporal"][0],
                stats["all_mitigated"][0]
            ]
        }
        table_df = pl.DataFrame(table).with_columns(
            (pl.col("Number of advisories") * 100 /
             stats["all_memory_safety"][0]).alias("% Of all memory safety advisories").round(2),
            (pl.col("Number of advisories") * 100 /
             stats["all_mitigated"][0]).alias("% Of mitigated advisories").round(2))

        table_df.write_csv(self.mitigation_summary.single_path())

    def _unmitigated_summary(self):
        df = self.sa.df.get()

        selectors = [
            pl.col("kind_memory_safety").sum().alias("all_memory_safety"),
            (pl.col("kind_memory_safety") & pl.col("mitigation_unmitigated")).sum().alias("all_unmitigated"),
            (pl.col("unmitigated_category") == "uninit-memory").sum().alias("uninit_memory"),
            (pl.col("unmitigated_category") == "stack-uaf").sum().alias("stack_uaf"),
            (pl.col("unmitigated_category") == "vm-subsystem").sum().alias("vm"),
            (pl.col("unmitigated_category") == "other").sum().alias("other")
        ]
        stats = df.select(selectors)

        renames = {
            "uninit_memory": "Missing initialization of padding bytes",
            "stack_uaf": "Stack temporal safety",
            "vm": "Direct VM subsystem operations",
            "other": "Other"
        }
        stats = stats.rename(renames)

        table_df = stats.unpivot(on=list(renames.values()), variable_name="Category", value_name="Number of advisories")
        table_df = table_df.with_columns(
            (pl.col("Number of advisories") * 100 /
             stats["all_memory_safety"][0]).alias("% Of memory safety advisories").round(2),
            (pl.col("Number of advisories") * 100 /
             stats["all_unmitigated"][0]).alias("% Of unmitigated advisories").round(2))
        table_df.write_csv(self.unmitigated_summary.single_path())

    def run(self):
        self._advisories_summary()
        self._unmitigated_summary()


class CheriBSDAdvisoriesCDF(PlotTask):
    """
    Generate a plot showing the mitigation spectrum.
    On the X axis, we have CHERI features, on the Y axis the % of mitigated vulnerabilities
    """
    public = True
    task_namespace = "kernel-advisories"
    task_name = "feature-mitigation-cdf"

    @dependency
    def sa_load(self):
        return CheriBSDKernelVulnLoad(self.session, self.analysis_config)

    @output
    def cdf(self):
        return PlotTarget(self, "cdf")

    def run_plot(self):
        df = self.sa_load.advisories_df.df.get()

        all_memory = (df["not_memory_safety_bug"] == Marker.EMPTY).sum()

        mitigated_by_spatial_safety = (((df["cheri_bounds_checking"] == Marker.YES) |
                                        (df["cheri_ptr_integrity"] == Marker.YES) |
                                        (df["cheri_ptr_provenance"] == Marker.YES)) &
                                       (df["cheri_subobject"] != Marker.YES)).sum()
        mitigated_by_subobject = (df["cheri_subobject"] == Marker.YES).sum()
        mitigated_by_temporal = (df["cheri_temporal"] == Marker.TEMPORAL).sum()

        mitigation_df = pd.DataFrame({
            "CHERI mitigation": [
                "Spatial safety", "Spatial safety + sub-object bounds", "Spatial + temporal safety",
                "Compartmentalisation"
            ],
            "Number mitigated": [
                mitigated_by_spatial_safety, mitigated_by_spatial_safety + mitigated_by_subobject,
                mitigated_by_spatial_safety + mitigated_by_subobject + mitigated_by_temporal, None
            ]
        })
        mitigation_df["Percent mitigated"] = 100 * mitigation_df["Number mitigated"] / all_memory

        sns.set_theme()
        with new_figure(self.cdf.paths()) as fig:
            ax = fig.subplots()
            sns.pointplot(mitigation_df, x="CHERI mitigation", y="Percent mitigated", ax=ax)
            ax.grid(axis="x", linestyle="--", color="gray", linewidth=.5)
            ax.grid(axis="y", linestyle="--", color="gray", linewidth=.5)
            ax.set_ylim(0, 100)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize="x-small")
