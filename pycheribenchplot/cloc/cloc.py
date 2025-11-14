import operator
import os
import re
from dataclasses import dataclass
from functools import cached_property, reduce

import polars as pl
import polars.selectors as cs
import seaborn as sns

from ..core.artefact import Target
from ..core.config import Any, Config, ConfigPath, config_field
from ..core.plot import PlotTarget, PlotTask
from ..core.plot_util import PlotGrid, PlotGridConfig, grid_barplot
from ..core.task import dependency, output
from .cloc_exec import CheriBSDClocExecTask, ClocExecTask


@dataclass
class FilterConfig(Config):
    when: dict[str, Any] = config_field(Config.REQUIRED,
                                        desc="Key/value parameterisation matcher to enable the filter.")
    accept: list[str] = config_field(list, desc="Accept filter regex list")
    reject: list[str] = config_field(list, desc="Reject filter regex list, run after accept")


@dataclass
class ComponentSpec(Config):
    """
    Configure component mappings from the input cloc data.

    The component name can be assigned by matching a regex on the files, or by using
    a combination of the parameterisation axes.
    """
    name: str = config_field(Config.REQUIRED, desc="Component name to assign.")
    when: dict[str, Any] = config_field(dict, desc="Match a key/value parameterisation.")
    regex: str | None = config_field(None, desc="Regular expression to match.")


@dataclass
class ClocByComponentConfig(PlotGridConfig):
    """
    Configure component assignment and data filtering.

    The :class:`ClocByComponent` plot aggregates data from all the cloc
    runs and produces a per-session plot with all data separated by component.

    The :att:`ClocByComponentConfig.cdb` can be used to reference an external
    compilation DB listing the files to include.
    """
    components: list[ComponentSpec] = config_field(Config.REQUIRED, desc="List of component assignment configurations.")

    cdb: ConfigPath | None = config_field(None, desc="Compilation database to filter relevant files")

    # This is used map components from a repository to the
    # correct path in the compilation database
    cdb_prefix: dict[str, str] = config_field(
        dict, desc="Optional mappings to prefix the paths for cross repo changes for each scenario")

    filters: list[FilterConfig] = config_field(list,
                                               desc="Additional filter specifications for selected parameterisations.")


class ClocByComponent(PlotTask):
    """
    Generate a bar plot showing the number of lines changed by source code component.

    The separation of components is arbitrarily defined in the task options and can
    group data from multiple scenarios into the same component.

    The task allows filtering the data by compilation database.

    The task generates the following auxiliary columns:
    - how: The type of change, one of 'added', 'modified', 'removed'.
    - component: Name of the component from the :attr:`ClocByComponentConfig.components`,
      this identifies the LoC counts for each of the 'how' values.
    - count: The line count.
    - metric: Identifies the meaning of the count column, this is the 'delta' count or
      'overhead' count.
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
        """
        Output csv file containing the tabular data corresponding to the
        plot output.
        """
        return Target(self, "table", ext="csv")

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

    def _apply_file_filter(self, df: pl.DataFrame, fc: FilterConfig) -> pl.DataFrame:
        """
        The data in the dataframe is filtered using the given filter configuration.

        The :attr:`FilterConfig.when` is used to select a dataframe slice where the filter
        applies. The remaining portion of the dataframe is left unchanged.
        The :attr:`FilterConfig.accept` regexes are used to whitelist paths.
        The :attr:`FilterConfig.reject` regexes are used to blacklist paths.
        Note that the whitelist is applied first, then the blacklist is applied on top
        of the whitelisted paths.
        """
        when_stmts = [pl.col(pkey) == pval for pkey, pval in fc.when.items()]
        when_expr = reduce(operator.or_, when_stmts, False)
        if fc.accept:
            accept_fn = lambda path: any([re.match(p, path) is not None for p in fc.accept])
            is_accept = pl.col("file").map_elements(accept_fn, return_dtype=pl.Boolean)
            df = df.filter(~when_expr | is_accept)
        if fc.reject:
            reject_fn = lambda path: any([re.match(p, path) is not None for p in fc.reject])
            is_reject = pl.col("file").map_elements(reject_fn, return_dtype=pl.Boolean)
            df = df.filter(~when_expr | ~is_reject)
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
        df = df.with_columns(pl.lit("other").alias("component"))
        true_expr = pl.lit(True, dtype=pl.Boolean)
        for spec in self.config.components:
            if spec.when:
                when_stmts = [pl.col(pkey) == pval for pkey, pval in spec.when.items()]
                when_expr = reduce(operator.and_, when_stmts, true_expr)
            else:
                when_expr = true_expr
            if spec.regex:
                # It would be nice to do this, but Rust regex does not support lookarounds
                # regex_expr = pl.col("file").str.find(spec.regex).is_not_null()
                match_fn = lambda path: re.match(spec.regex, path) is not None
                regex_expr = pl.col("file").map_elements(match_fn, return_dtype=pl.Boolean)
            else:
                regex_expr = true_expr

            component = pl.lit(spec.name)
            existing = pl.col("component")
            df = df.with_columns(pl.when(when_expr & regex_expr).then(component).otherwise(existing).alias("component"))

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
        for filter_config in self.config.filters:
            head_df = self._apply_file_filter(head_df, filter_config)
            baseline_df = self._apply_file_filter(baseline_df, filter_config)

        # Assign components to each
        head_df = self._assign_component(head_df)
        baseline_df = self._assign_component(baseline_df)

        # Dump absolute diff data sorted by # of changes
        dump_df = head_df.sort("component",
                               pl.col("added") + pl.col("modified") + pl.col("removed")).select(
                                   "component", "file", "added", "modified", "removed")
        dump_df.write_csv(self.diff_audit.single_path())

        # Create per-component baseline counts
        group_cols = [*self.param_columns, "component"]
        baseline_agg_df = baseline_df.group_by(group_cols).agg(pl.col("count").sum())
        head_agg_df = head_df.group_by(group_cols).agg(
            pl.col("added").sum(),
            pl.col("removed").sum(),
            pl.col("modified").sum(),
            pl.col("same").sum())
        df = head_agg_df.join(baseline_agg_df, on=group_cols, how="left").with_columns(pl.col("count").fill_null(0))

        # Now create the absolute and % difference data in long-form
        # This is going to be used for the PlotGrid.
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

        # Grab the color palette as a matplotlib cmap
        cmap = sns.color_palette(as_cmap=True)

        grid_config = self.config.set_fixed(
            tile_col="metric",
            tile_col_as_xlabel=True,
            tile_sharey=True,
            hue="how",
            hue_colors={
                "added": cmap[2],  # green
                "modified": cmap[1],  # orange
                "removed": cmap[3]  # red
            }).with_config_default(sort_order={"<how>": ["added", "modified", "removed"]})

        with PlotGrid(self.cloc_plot, view_df, grid_config) as grid:
            # Dump the sorted tabular data using the ordering specified by the grid config
            dump_df = grid.get_grid_df().select(["component", "how", "metric", "count"])
            dump_df.write_csv(self.cloc_table.single_path())

            # Generate the grid plot
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
