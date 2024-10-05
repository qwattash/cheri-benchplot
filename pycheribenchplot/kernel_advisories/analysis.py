from datetime import date

import polars as pl
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

from ..core.analysis import AnalysisTask
from ..core.artefact import DataFrameTarget, Target, make_dataframe_loader
from ..core.config import Config, ConfigPath, validate_file_exists
from ..core.plot import PlotTarget, PlotTask, new_figure
from ..core.task import SessionDataGenTask, dependency, output
from .ingest import IngestAdvisoriesTask


class CheriBSDAdvisoriesHistory(PlotTask):
    """
    Generate plot of the distribution of advisories over time.
    """
    public = True
    task_namespace = "kernel-advisories"
    task_name = "advisories-timeline"

    events = {
        date(2021, 4, 13): "Added KASAN kernel option",
        date(2017, 10, 17): "Syzkaller FreeBSD support",
        date(2016, 5, 27): "First KASAN mention in a fix",
        date(2009, 10, 2): "User NULL mappings forbidden"
    }

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

        view = df.group_by("Year").agg(pl.count().alias(total_col),
                                       pl.col("kind_memory_safety").sum().alias(memory_safety_col),
                                       (pl.col("kind_memory_safety") & ~pl.col("kind_uaf")
                                        & ~pl.col("kind_race")).sum().alias(spatial_safety_col)).unpivot(
                                            index=["Year"],
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

            event_years = [d.year for d in self.events.keys()]
            for year in event_years:
                ax.axvline(year, linestyle="-.", linewidth=1, color="gray")

            # Add the twin top X axis with history markers
            ax_t = ax.twiny()
            ax_t.grid(visible=False)
            ax_t.set_xlim(*ax.get_xlim())
            ax_t.set_xticks(event_years)
            ax_t.set_xticklabels(self.events.values(), rotation=-25, ha="right", size=8)

        # view = df.with_columns(
        #     pl.col("date").str.to_datetime("%Y-%m-%d %H:%M:%S").dt.year().alias("Year")
        # ).group_by("Year").agg(
        #     pl.col("kind_memory_safety").sum().alias("memory_safety_advisories"),
        #     pl.col("mitigation_temporal").sum().alias("temporal_safety"),
        #     (pl.col("kind_memory_safety") & ~(pl.col("mitigation_unmitigated") | pl.col("mitigation_temporal"))).sum().alias("spatial_safety")
        # )

        # df["year"] = df.index.get_level_values("date").year
        # df["advisory"] = df.index.get_level_values("advisory")
        # # Some intermediate columns
        # df["is_memory_safety"] = (df["not_memory_safety_bug"] == Marker.EMPTY)
        # df["is_memory_safety_spatial"] = (df["is_memory_safety"] & ~df["uaf_uma_kmalloc"])
        # df["is_mitigated_spatial"] = (df["cheri_does_not_help"] == Marker.EMPTY)
        # df["is_mitigated_temporal"] = (df["cheri_does_not_help"] == Marker.TEMPORAL)
        # df["is_mitigated"] = df["is_mitigated_spatial"] | df["is_mitigated_temporal"]


class CheriBSDAdvisoriesTables(AnalysisTask):
    """
    Generate summary tables CSV files from the main vulnerability analysis table.
    """
    public = True
    task_namespace = "kernel-vuln"
    task_name = "summary-tables"

    @dependency
    def sa_load(self):
        return CheriBSDKernelVulnLoad(self.session, self.analysis_config)

    @output
    def mitigation_summary(self):
        return Target(self, "advisories-summary", ext="csv")

    @output
    def unmitigated_summary(self):
        return Target(self, "unmitigated-summary", ext="csv")

    def _advisories_summary(self):
        df = self.sa_load.advisories_df.df.get()

        all_memory = (df["not_memory_safety_bug"] == Marker.EMPTY).sum()
        all_mitigated = ((df["cheri_does_not_help"] != Marker.YES) &
                         (df["cheri_does_not_help"] != Marker.UNKNOWN)).sum()

        mitigation_df = pd.DataFrame({
            "Description": [
                "Mitigated by bounds checking",
                "Mitigated by sub-object bounds checking",
                "Mitigated by pointer provenance and integrity",
                "Total mitigated",
                "Possibly mitigated by temporal safety",
                "Total mitigated with temporal safety",
            ]
        })
        mitigation_df["Number of advisories"] = [
            (df["cheri_bounds_checking"] == Marker.YES).sum(), (df["cheri_subobject"] == Marker.YES).sum(),
            ((df["cheri_ptr_integrity"] == Marker.YES) & (df["cheri_ptr_provenance"] == Marker.YES)).sum(),
            (df["cheri_does_not_help"] == Marker.EMPTY).sum(), (df["cheri_temporal"] == Marker.TEMPORAL).sum(),
            ((df["cheri_does_not_help"] == Marker.EMPTY) | (df["cheri_does_not_help"] == Marker.TEMPORAL)).sum()
        ]
        # Format % columns as string percentages
        mitigation_df["% Of memory safety advisories"] = (mitigation_df["Number of advisories"] * 100 /
                                                          all_memory).round(decimals=2)
        mitigation_df["% Of advisories that can be mitigated"] = (mitigation_df["Number of advisories"] * 100 /
                                                                  all_mitigated).round(decimals=2)
        mitigation_df.to_csv(self.mitigation_summary.path, index=False, header=True)

    def _unmitigated_summary(self):
        advisory_df = self.sa_load.advisories_df.df.get()
        df = self.sa_load.unmitigated_df.df.get()

        all_unmitigated = ((advisory_df["not_memory_safety_bug"] == Marker.EMPTY) &
                           (advisory_df["cheri_does_not_help"] == Marker.YES)).sum()

        out_df = pd.DataFrame({
            "Description": [
                "Missing initialization of padding bytes", "Direct VM subsystem operations", "Stack temporal safety",
                "Other"
            ]
        })
        out_df["Number of advisories"] = [(df["reason_padding_initialization"] == Marker.YES).sum(),
                                          (df["reason_direct_vm_subsystem_access"] == Marker.YES).sum(),
                                          (df["reason_stack_use_after_free"] == Marker.YES).sum(),
                                          (df["reason_other"] == Marker.YES).sum()]
        out_df["% Of unmitigated advisories"] = (out_df["Number of advisories"] * 100 /
                                                 all_unmitigated).round(decimals=2)
        out_df.to_csv(self.unmitigated_summary.path, index=False, header=True)

    def run(self):
        self._advisories_summary()
        self._unmitigated_summary()


class CheriBSDAdvisoriesCDF(PlotTask):
    """
    Generate a plot showing the mitigation spectrum.
    On the X axis, we have CHERI features, on the Y axis the % of mitigated vulnerabilities
    """
    public = True
    task_namespace = "kernel-vuln"
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
