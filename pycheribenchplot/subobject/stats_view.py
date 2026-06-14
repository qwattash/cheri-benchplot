from dataclasses import dataclass
import re

import numpy as np
import polars as pl

from ..core.analysis import SliceAnalysisTask
from ..core.artefact import Target
from ..core.config import Config, config_field
from ..core.plot import PlotTarget, SlicePlotTask
from ..core.plot_grid import BarPlotConfig, PlotGrid, PlotGridConfig, grid_barplot
from ..core.task import dependency, output
from .scan_dwarf import AnnotateImpreciseSubobjectLayouts, AnnotateLayoutsWithVLA


@dataclass
class ReportConfig(Config):
    show_parameters: list[str] | None = config_field(
        None, desc="Only include these parameter axes in the output tables."
    )
    ignore_files: list[str] | None = config_field(
        None,
        desc="Files matching patterns in this list are not included in the report.",
    )
    field_groups: dict[str, str] | None = config_field(
        None, desc="Map a group name to a regex to match the file column."
    )


class ImpreciseSubobjectReport(SliceAnalysisTask):
    """
    Produce information about imprecise sub-objects in tabular form.

    A summary table lists the number of imprecise sub-objects for each
    relevant target group, with details on how many are arrays and allow
    overflow into a pointer.

    A detail table lists each imprecise sub-object with classification tags.
    """

    public = True
    task_namespace = "subobject"
    task_name = "imprecise-report"
    task_config_class = ReportConfig

    @dependency
    def layouts(self):
        return AnnotateImpreciseSubobjectLayouts(
            self.session, self.slice_info, self.analysis_config
        )

    @output
    def summary(self):
        return Target(self, "summary", ext="csv")

    @output
    def fields(self):
        return Target(self, "fields", ext="csv")

    @output
    def field_groups(self):
        return Target(self, "field-groups", ext="csv")

    def _gen_fields_table(self, df, table_cols):
        data_cols = [
            "member_name",
            "file",
            "line",
            "byte_size",
            "is_array",
            "is_overrun_into_ptr",
        ]

        self.logger.info(
            "Generate imprecise sub-object fields report: %s", self.slice_info
        )
        output_cols = [*table_cols, *data_cols]
        table = df.filter(pl.col("_alias_color").is_not_null())
        table = table.with_columns(
            pl.col("_alias_pointer_member").alias("is_overrun_into_ptr"),
            pl.col("array_items").is_not_null().alias("is_array"),
        )

        table = table.select(output_cols)
        table = table.sort([*table_cols, "member_name", "file", "line"])
        table.write_csv(self.fields.single_path())

        # Prepare field group counts if configured
        if self.config.field_groups:

            def _group_match(value):
                for name, pattern in self.config.field_groups.items():
                    if re.match(pattern, value):
                        return name
                return "other"

            table = table.with_columns(
                pl.col("file")
                .map_elements(_group_match, return_dtype=str)
                .alias("group")
            )
            table = table.group_by("group").agg(
                pl.col("member_name").count().alias("count"),
                pl.col("is_array").sum(),
                pl.col("is_overrun_into_ptr").sum(),
            )
            table.write_csv(self.field_groups.single_path())

    def _gen_summary_table(self, df, table_cols):
        data_cols = [
            "imprecise_layouts",
            "percent_of_total",
            "count",
            "is_array",
            "is_overrun_into_ptr",
        ]

        self.logger.info(
            "Generate imprecise sub-object summary report: %s", self.slice_info
        )
        table = df.filter(pl.col("_alias_color").is_not_null())
        table = table.with_columns(
            pl.col("_alias_pointer_member").alias("is_overrun_into_ptr"),
            pl.col("array_items").is_not_null().alias("is_array"),
        )

        summary = (
            table.group_by(table_cols)
            .agg(
                pl.col("member_name").count().alias("count"),
                pl.col("id").n_unique().alias("imprecise_layouts"),  # layout ID
                pl.col("total_layouts").first(),
                pl.col("is_array").sum(),
                pl.col("is_overrun_into_ptr").sum(),
            )
            .with_columns(
                pl.format(
                    "{}\\%",
                    (100 * pl.col("imprecise_layouts") / pl.col("total_layouts")).round(
                        2
                    ),
                ).alias("percent_of_total")
            )
            .select(table_cols + data_cols)
        )
        summary.write_csv(self.summary.single_path())

    def run(self):
        df = self.layouts.imprecise_layouts.get()

        if self.config.ignore_files:
            for pattern in self.config.ignore_files:
                df = df.filter(~pl.col("file").str.contains(pattern))

        if self.config.show_parameters:
            table_cols = [
                p for p in self.param_columns if p in self.config.show_parameters
            ]
        else:
            table_cols = self.param_columns

        self._gen_fields_table(df, table_cols)
        self._gen_summary_table(df, table_cols)


class VLASubobjectReport(SliceAnalysisTask):
    """
    Produce information about VLA members in tabular form.

    A summary table lists the number of imprecise sub-objects for each
    relevant slice group.

    A detail table lists each VLA member with the maximum representable size.
    """

    public = True
    task_namespace = "subobject"
    task_name = "vla-report"
    task_config_class = ReportConfig

    @dependency
    def vla_fields(self):
        return AnnotateLayoutsWithVLA(
            self.session, self.slice_info, self.analysis_config
        )

    @output
    def summary(self):
        return Target(self, "summary", ext="csv")

    @output
    def fields(self):
        return Target(self, "fields", ext="csv")

    def run(self):
        df = self.vla_fields.layouts_with_vla.get()
        df = df.filter(pl.col("max_vla_size").is_not_null())

        if self.config.ignore_files:
            for pattern in self.config.ignore_files:
                df = df.filter(~pl.col("file").str.contains(pattern))

        if self.config.show_parameters:
            table_cols = [
                p for p in self.param_columns if p in self.config.show_parameters
            ]
        else:
            table_cols = self.param_columns

        self.logger.info("Generate VLA fields report: %s", self.slice_info)
        data_cols = ["member_name", "file", "line", "byte_offset", "max_vla_size"]
        table = df.select(table_cols + data_cols).sort(table_cols + data_cols)
        table.write_csv(self.fields.single_path())

        self.logger.info("Generate VLA summary report: %s", self.slice_info)
        summary = table.group_by(table_cols).agg(pl.col("member_name").count())
        summary.write_csv(self.summary.single_path())


@dataclass
class ImpreciseSizeDistPlotConfig(PlotGridConfig, BarPlotConfig):
    pass


class ImpreciseSizeDistPlot(SlicePlotTask):
    """
    Produce a plot that shows the distribution of subobject imprecision sizes.

    This reports the number of extra padding in the base and top of sub-object
    capabilities.
    """

    public = True
    task_namespace = "subobject"
    task_name = "imprecise-size-distribution"
    task_config_class = ImpreciseSizeDistPlotConfig

    rc_params = {"text.usetex": True}

    @dependency
    def layouts(self):
        return AnnotateImpreciseSubobjectLayouts(
            self.session, self.slice_info, self.analysis_config
        )

    @output
    def size_dist(self):
        return PlotTarget(self)

    def run_plot(self):
        """
        Generate the distribution plot.
        """
        df = self.layouts.imprecise_layouts.get()
        # XXX maybe add knob to narrow bounds as much as possible on bitfields
        # top = offset + byte_size + ((bit_offset + bit_size) / 8).ceil()
        req_top = pl.col("byte_offset") + pl.col("byte_size")
        df = df.with_columns(
            (pl.col("byte_offset") - pl.col("base")).alias("base_padding"),
            (pl.col("top") - req_top).alias("top_padding"),
        )
        assert (df["base_padding"] >= 0).all(), "Negative base padding!"
        assert (df["top_padding"] >= 0).all(), "Negative top padding!"
        # Compute padding size range for bins
        bin_max = max(df["base_padding"].max(), df["top_padding"].max())
        bin_pow = np.arange(0, np.log2(bin_max) + 1, dtype=int)
        bin_edges = [2**n for n in bin_pow]

        # Now produce the histograms for base and top padding and combine them into long form,
        # keyed on the `side` column.
        base_pad_hist = self.histogram(
            df, "base_padding", prefix="hist_", bins=bin_edges
        ).with_columns(pl.lit("base").alias("side"))
        top_pad_hist = self.histogram(
            df, "top_padding", prefix="hist_", bins=bin_edges
        ).with_columns(pl.lit("top").alias("side"))
        hist = pl.concat([base_pad_hist, top_pad_hist], how="vertical").cast(
            {"hist_bin": pl.UInt64}
        )

        # Build an helper metadata column that can be used for the bin labels

        grid_config = self.config.with_config_default(
            hue="<side>", tile_xaxis="<hist_bin>", tile_yaxis="<hist_count>"
        )
        # Build a default mapping for the hist_bin labels
        # grid_config = grid_config.set_display_defaults(
        #     param_values={"hist_bin": {
        #         2**n: f"$2^{{{n}}}$"
        #         for n in bin_pow
        #     }})

        with PlotGrid(self.size_dist, hist, grid_config) as grid:
            grid.map(
                grid_barplot,
                x=grid_config.tile_xaxis,
                y=grid_config.tile_yaxis,
                config=grid_config,
            )
            grid.add_legend()
