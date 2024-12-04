import os
import re
from dataclasses import dataclass
from functools import cached_property

import polars as pl
import polars.selectors as cs
import seaborn as sns

from ..core.artefact import Target
from ..core.config import Config, ConfigPath, config_field
from ..core.plot import PlotTarget, PlotTask
from ..core.plot_util import (DisplayGrid, DisplayGridConfig, ParamWeight, WeightMode, grid_barplot)
from ..core.task import dependency, output
from .cloc_exec import CheriBSDClocExecTask, ClocExecTask


@dataclass
class FilterConfig(Config):
    accept: list[str] = config_field(list, desc="Accept filter regex list")
    reject: list[str] = config_field(list, desc="Reject filter regex list, run after accept")


@dataclass
class ClocByComponentConfig(DisplayGridConfig):
    components: dict[str, str] = config_field(dict,
                                              desc="Component matchers: the key is the component name, "
                                              "the value is a regex that matches the file path")
    cdb: ConfigPath | None = config_field(None, desc="Compilation database to filter relevant files")
    # This is used map components from a repository to the
    # correct path in the compilation database
    cdb_prefix: dict[str, str] = config_field(
        dict, desc="Optional mappings to prefix the paths for cross repo changes for each scenario")
    filters: dict[str, FilterConfig] = config_field(dict, desc="Additional filter specifications for each scenario")


class ClocByComponent(PlotTask):
    """
    Generate a bar plot showing the number of lines changed by source code component.

    The separation of components is arbitrarily defined in the task options and can
    group data from multiple scenarios into the same component.

    The task allows filtering the data by compilation database.
    """
    task_namespace = "cloc"
    task_name = "cloc-by-component"
    task_config_class = ClocByComponentConfig
    public = True

    @dependency
    def cloc_data(self):
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(CheriBSDClocExecTask)
            if task:
                yield task.cloc_output.get_loader()
                continue

            task = b.find_exec_task(ClocExecTask)
            if task:
                yield task.cloc_output.get_loader()

    @dependency
    def cloc_baseline(self):
        for b in self.session.all_benchmarks():
            task = b.find_exec_task(CheriBSDClocExecTask)
            if task:
                yield task.cloc_baseline.get_loader()
                continue

            task = b.find_exec_task(ClocExecTask)
            if task:
                yield task.cloc_baseline.get_loader()

    @output
    def cdb_extra_audit(self):
        """
        Output file containing the files found in the compilation DB but not found
        in the CLoC baseline count.
        These should match only expected paths that have been excluded.
        """
        return Target(self, "audit-cdb-extra", ext="csv")

    @output
    def diff_audit(self):
        """
        Output csv file containing the changed files sorted by # of changes
        This is useful to audit differences.
        """
        return Target(self, "audit-diff", ext="csv")

    @output
    def cloc_plot(self):
        return PlotTarget(self)

    @output
    def cloc_table(self):
        return Target(self, prefix="raw", ext="csv")

    @cached_property
    def compilation_db(self):
        """
        Fetch the compilation DB data, if specified
        """
        if self.config.cdb is None:
            return None
        df = pl.read_csv(self.config.cdb).unique("file")
        df = df.with_columns(pl.col("file").map_elements(lambda p: os.path.normpath(p), return_dtype=str))
        return df

    def _filter_by_cdb(self, df) -> pl.DataFrame:
        if self.compilation_db is None:
            return df

        return df.join(self.compilation_db, on="file")

    def _filter_by_file(self, df: pl.DataFrame, scenario: str, fc: FilterConfig) -> pl.DataFrame:
        # Accept filters will allow only matching paths to be considered
        if fc.accept:
            accept_fn = lambda path: any([re.match(p, path) is not None for p in fc.accept])
            is_accept = pl.col("file").map_elements(accept_fn, return_dtype=pl.Boolean)
            df = df.filter((pl.col("scenario") != scenario) | ((pl.col("scenario") == scenario) & is_accept))
        if fc.reject:
            reject_fn = lambda path: any([re.match(p, path) is not None for p in fc.reject])
            is_reject = pl.col("file").map_elements(reject_fn, return_dtype=pl.Boolean)
            df = df.filter((pl.col("scenario") != scenario) | ((pl.col("scenario") == scenario) & ~is_reject))
        return df

    def _patch_cdb_file_paths(self, df) -> pl.DataFrame:
        """
        When the compilation DB measures the build files, diff extracted from
        a different repo path may not match.
        Apply `cdb_prefix` mappings to each scenario.
        """
        if self.compilation_db is None:
            return df

        for key, prefix in self.config.cdb_prefix.items():
            patch = pl.concat_str([pl.lit(prefix + "/"), pl.col("file")])
            df = df.with_columns(pl.when(pl.col("scenario") == key).then(patch).otherwise(pl.col("file")).alias("file"))
        df = df.with_columns(pl.col("file").map_elements(lambda p: os.path.normpath(p), return_dtype=str))
        return df

    def _assign_component(self, df) -> pl.DataFrame:
        """
        Assign the component name to each file.

        If components are configured, these will override completely the default
        assignment of scenarios to components.
        This will automatically define a catch-all component named 'other' that collects
        anything that does not match.
        """
        if not self.config.components:
            return df.with_columns(pl.col("scenario").alias("component"))

        df = df.with_columns(pl.lit("other").alias("component"))
        for name, pattern in self.config.components.items():
            cname = pl.lit(name)
            default = pl.col("component")
            is_matching = pl.col("file").map_elements(lambda path: re.match(pattern, path) is not None,
                                                      return_dtype=pl.Boolean)
            df = df.with_columns(pl.when(is_matching).then(cname).otherwise(default).alias("component"))
        return df

    def collect_data(self):
        """
        Coalesce the data for each scenario execution.
        """
        data = [dep.df.get() for dep in self.cloc_data]
        df = pl.concat(data, how="vertical")
        df = self._patch_cdb_file_paths(df)
        df = self._filter_by_cdb(df)
        return df

    def collect_baseline(self):
        """
        Coalesce the baseline count for each scenario execution.
        """
        data = [dep.df.get() for dep in self.cloc_baseline]
        df = pl.concat(data, how="vertical")
        df = self._patch_cdb_file_paths(df)

        # Sanity check: we expect the compilation DB to include *only*
        # files that have been counted or excluded during generation
        # manual audit is required.
        if self.compilation_db is not None:
            self.logger.info("Generate CDB audit file %s", self.cdb_extra_audit.single_path())
            # keep rows in cdb that are not in baseline
            extra = self.compilation_db.join(df, on="file", how="anti")
            extra.write_csv(self.cdb_extra_audit.single_path())
        df = self._filter_by_cdb(df)
        return df

    def run_plot(self):
        head_df = self.collect_data()
        baseline_df = self.collect_baseline()

        # Apply extra filters (e.g. tests)
        for scenario, filter_config in self.config.filters.items():
            head_df = self._filter_by_file(head_df, scenario, filter_config)
            baseline_df = self._filter_by_file(baseline_df, scenario, filter_config)

        # Assign components to each
        head_df = self._assign_component(head_df)
        baseline_df = self._assign_component(baseline_df)

        # Dump absolute diff data sorted by # of changes
        dump_df = head_df.sort("component",
                               pl.col("added") + pl.col("modified") + pl.col("removed")).select(
                                   "component", "file", "added", "modified", "removed")
        dump_df.write_csv(self.diff_audit.single_path())

        # Create per-component baseline counts
        group_cols = ["target", "variant", "runtime", "scenario", "component"]
        baseline_agg_df = baseline_df.group_by(group_cols).agg(pl.col("count").sum())
        head_agg_df = head_df.group_by(group_cols).agg(
            pl.col("added").sum(),
            pl.col("removed").sum(),
            pl.col("modified").sum(),
            pl.col("same").sum())
        df = head_agg_df.join(baseline_agg_df, on=group_cols, how="left").with_columns(pl.col("count").fill_null(0))

        # Now create the absolute and % difference data in long-form
        # This is going to be used for the DisplayGrid.
        df = df.with_columns(
            pl.lit("# of lines").alias("metric"),
            cs.by_name("added", "removed", "modified", "same").cast(pl.Float64))
        rel_delta_df = df.with_columns(
            pl.col("added") * 100 / pl.col("count"),
            pl.col("removed") * 100 / pl.col("count"),
            pl.col("modified") * 100 / pl.col("count"),
            pl.col("same") * 100 / pl.col("count"),
            pl.lit("% of lines").alias("metric"))
        view_df = pl.concat([df, rel_delta_df], how="vertical").select(cs.exclude("count"))
        how = ["added", "modified", "removed"]
        view_df = view_df.unpivot(on=how,
                                  index=[*self.param_columns, "component", "metric"],
                                  variable_name="how",
                                  value_name="count")

        # Ensure we have the proper theme with default colors
        sns.set_theme()
        cmap = sns.color_palette(as_cmap=True)

        config = self.config.set_fixed(
            tile_col="metric",
            tile_col_as_xlabel=True,
            tile_sharey=True,
            hue="how",
            hue_colors={
                "added": cmap[2],  # green
                "modified": cmap[1],  # orange
                "removed": cmap[3]  # red
            },
        )
        config.set_default(
            param_sort_weight={
                "component": ParamWeight(mode=WeightMode.SortAscendingAsStr, base=10, step=10),
                "how": ParamWeight(mode=WeightMode.Custom, weights={
                    "added": 0,
                    "modifier": 1,
                    "removed": 2
                })
            })

        with DisplayGrid(self.cloc_plot, view_df, config) as grid:
            grid.map(grid_barplot, x="count", y="component", orient="y", stack=True, coordgen_kwargs={"pad_ratio": 0.5})
            grid.map(lambda tile, chunk: tile.ax.tick_params(axis="y", labelsize="x-small"))

            # Generate text annotations for the Absolute SLoC
            # if self.config.show_text_annotations:
            #     totals = show_df.sum(axis=1)
            #     for y, value in zip(ax.get_yticks(), totals):
            #         magnitude = np.log10(value)
            #         if magnitude > 6:
            #             txt_value = f"{np.round(value / 10**6, 2):.2f}M"
            #         elif magnitude > 3:
            #             txt_value = f"{np.round(value / 10**3, 2):.2f}K"
            #         else:
            #             txt_value = f"{value:d}"
            #         ax.text(value, y, txt_value, fontsize="xx-small", va="center")
            grid.add_legend()
