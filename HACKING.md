
# Hacking Guide

This document is aimed at contributors who want to extend `cheri-benchplot` with support
for new benchmarks or analysis tasks. It covers the internal architecture, coding conventions,
and step-by-step guides for the most common extension points.

For an introduction to the tool and its CLI commands, refer to `README.md` first.

---

## Architecture

### The Pipeline at a Glance

Each benchmarking session goes through two independent phases:

1. **Execution phase** — tasks produce shell scripts that are bundled and shipped to the
   benchmark host (a CheriBSD machine). Each script runs one combination of parameterisation
   values for a fixed number of iterations and collects raw data.
2. **Analysis phase** — tasks load the raw data from completed runs, compute statistics
   (medians, confidence intervals, overhead relative to a baseline), and produce plots.

The two phases are driven by separate configuration files (pipeline config and analysis
config) and can be re-run independently. The unit of work in both phases is a `Task`.

### Tasks

`Task` is the base class for all scheduled work (`pycheribenchplot/core/task.py`). Every
concrete task class is automatically registered by the `TaskRegistry` metaclass when the
module is imported. The registry key is the pair `(task_namespace, task_name)`.

A task is described by four class attributes:

```python
class MyTask(ExecutionTask):
    task_namespace = "mybenchmark"   # dot-separated namespace
    task_name = "exec"               # name within the namespace
    public = True                    # visible in config files and CLI
    task_config_class = MyConfig     # optional Config subclass
```

Only tasks with `public = True` can be referenced in configuration files or from the CLI.
Tasks without a `task_name` / `task_namespace` are treated as abstract base classes and are
not registered.

Tasks are deduplication units: the scheduler assigns a unique `task_id` to each task
instance and uses the Borg pattern to collapse multiple references to the same logical
task into a single shared object (see `pycheribenchplot/core/borg.py`).

The task hierarchy is:

```
Task
├── SessionTask          — singleton per session
│   └── AnalysisTask     — base for all analysis/plot tasks
│       ├── PlotTask           (+ PlotTaskMixin) — session-wide plots
│       ├── GenericAnalysisTask  — the public "analysis.dynamic" dispatcher
│       └── SliceAnalysisTask  — receives a subset of benchmarks
│           └── SlicePlotTask  (+ PlotTaskMixin) — per-slice plots
└── DatasetTask          — one instance per parameterisation combination
    └── ExecutionTask    — drives script generation for one benchmark run
        └── TimingExecTask — adds timing collection context
```

### Outputs and Dependencies

Task inputs and outputs are declared as class-level descriptors using `@output` and
`@dependency`. Never override the `outputs()` or `dependencies()` methods directly.

```python
from ..core.task import ExecutionTask, dependency, output

class MyTask(ExecutionTask):
    @output
    def results(self):
        # Called each time an output reference is needed.
        # Must return a Target (or list of Targets).
        return RemoteBenchmarkIterationTarget(self, "results", loader=MyLoader, ext="json")

    @dependency
    def my_dep(self):
        # Called to enumerate dependencies.
        # Must yield or return Task instances.
        for bench in self.slice_benchmarks:
            task = bench.find_exec_task(MyExecTask)
            yield task.results.get_loader()
```

The `@output` decorator registers the method in `_output_registry`. The `@dependency`
decorator registers it in `_deps_registry`. Both registries are inherited by subclasses.

### Targets

A `Target` is an output artefact produced by a task. Targets are also Borg objects: all
references to the same `(task_id, output_id)` pair share state.

Commonly used target types:

| Class | Use case |
|---|---|
| `RemoteBenchmarkIterationTarget` | A file produced per iteration on the benchmark host; carries a loader that parses it into a DataFrame |
| `BenchmarkIterationTarget` | Same, but local (generated during the execution phase on the analysis host) |
| `Target` | A plain local file (e.g. a CSV stats dump) |
| `PlotTarget` | A plot image file; accepted by `PlotGrid` |
| `ValueTarget` | An in-memory value with an optional validator; useful for passing data between tasks |
| `SQLTarget` | An SQLite database managed via SQLAlchemy |

Always use `Target.iter_paths()` or `shell_path_builder()` to resolve output paths.
Never hardcode paths.

### Configuration

All task configuration is a Python `@dataclass` that inherits `Config`
(`pycheribenchplot/core/config.py`). Fields are declared with `config_field`:

```python
from dataclasses import dataclass
from ..core.config import Config, config_field

@dataclass
class MyConfig(Config):
    binary_path: Path = config_field(Config.REQUIRED, desc="Path to the benchmark binary.")
    iterations: int  = config_field(10, desc="Number of warm-up iterations to discard.")
    extra_args: str  = config_field("", desc="Extra arguments passed to the binary.")
```

Rules:
- Always use `config_field(default, desc="...")`. Never use `dataclasses.field()` directly;
  it bypasses marshmallow serialization and omits the field from CLI documentation.
- Use `Config.REQUIRED` as the default for mandatory fields.
- The `desc=` string is surfaced by `benchplot-cli info task <handler>`.

`Config` subclasses can also define marshmallow validators using the `@validates` and
`@validates_schema` decorators from marshmallow to enforce constraints at load time.

### Analysis Tasks

`AnalysisTask` (`pycheribenchplot/core/analysis.py`) is the base for all session-scoped
work. Its most important method is:

```python
stats = self.compute_overhead(df, "metric_column", how="median", overhead_scale=100)
```

This groups the dataframe by the parameterisation axes, computes medians (or means) with
bootstrapped confidence intervals (BCa method), and returns a long-form DataFrame
containing three `_metric_type` values per group:

- `"absolute"` — the raw statistic for each group.
- `"delta"` — the absolute difference from the baseline group.
- `"overhead"` — the relative difference from the baseline, scaled by `overhead_scale`.

Each row also carries an `_is_baseline` boolean. The baseline group is identified by the
`baseline` filter in the analysis configuration file.

---

## Code Conventions

- **Python 3.12+**: use `type` aliases, PEP 604 `X | Y` unions, and structural pattern
  matching where appropriate.
- **Docstrings**: every public class and method must have a docstring. Task class
  docstrings are displayed verbatim by `benchplot-cli info task <handler>`.
- **Logging**: use `self.logger` (available on every `Task`). Never use `print`.
  Use `XXX` comments to flag known issues and TODOs.
- **Type hints**: annotate all function signatures.
- **DataFrames**: use [Polars](https://pola.rs/) exclusively. Use `polars.selectors`
  (`import polars.selectors as cs`) for column selection patterns. Prefer long-form
  data layout for analysis outputs.
- **Module registration**: every new module must import its submodules in `__init__.py`
  so that task classes are registered before the CLI or config parser needs them:

  ```python
  # pycheribenchplot/mybenchmark/__init__.py
  from . import exec, plot
  ```

---

## The Jinja2 Script Template System

### Base Template

All benchmark runner scripts extend `runner-script.sh.jinja`. This template contains the
outer execution loop and dispatches to hooks registered by the execution tasks.
It is also the default template used when no `set_template()` call is made.

The template exposes the following named blocks:

| Block | Override needed? | Purpose |
|---|---|---|
| `functions` | optional | Shell helper function definitions |
| `global_setup` | optional | Runs once before the iteration loop |
| `iteration_setup` | optional | Per-iteration setup |
| `iteration_exec` | **primary override point** | The workload invocation |
| `iteration_teardown` | optional | Per-iteration teardown |
| `global_teardown` | optional | Runs once after the loop |
| `post_process_iteration` | optional | Per-iteration post-processing (second loop) |

All blocks except `functions` dispatch registered hooks before (or instead of) custom
content. Always call `{{ super() }}` when overriding a block that should preserve hook
dispatch — omitting it will silently suppress any hooks that other generator tasks have
registered.

The following context variables are always available, injected by `ScriptContext`:

| Variable | Description |
|---|---|
| `dataset_id` | Unique identifier of this benchmark run |
| `iterations` | Total number of iterations |
| `instance` | The instance configuration object |
| `parameters` | Dict of parameterisation key/value pairs for this run |
| `current_iteration` | Shell variable reference `${ITERATION}`, used inside loop blocks |
| `setup_hooks` | List of global setup hooks |
| `teardown_hooks` | List of global teardown hooks |
| `iter_setup_hooks` | List of per-iteration setup hooks |
| `iter_teardown_hooks` | List of per-iteration teardown hooks |
| `iter_exec_hooks` | List of per-iteration exec hooks |

### Writing a Benchmark Template

Create a new `*.sh.jinja` file in `pycheribenchplot/templates/`. Always extend the base
template and override at least `iteration_exec`:

```jinja
{#
 # Runner script for My Benchmark.
 # Supports the hwpmc and timing extensions.
 #}
{% extends "runner-script.sh.jinja" %}

{% import "hwpmc.inc.jinja" as hwpmc with context %}
{% import "timing.inc.jinja" as timing with context %}

{% block iteration_exec %}
  echo "Run my-benchmark iteration {{ current_iteration }}"

  {% call() timing.timeit(current_iteration) -%}
    {{ hwpmc.pmcstat(current_iteration) }} \
      my-benchmark-binary \
      --output {{ my_gen_output(current_iteration) }}
  {%- endcall %}
{% endblock %}

{% block global_teardown %}
  {{ super() }}
  {% for i in range(iterations) %}
    {{ hwpmc.pmcstat_postprocess(i) }}
  {% endfor %}
{% endblock %}
```

Key points:
- `{% import "hwpmc.inc.jinja" as hwpmc with context %}` and
  `{% import "timing.inc.jinja" as timing with context %}` should always be included.
  The macros are no-ops when the corresponding add-on is not configured, so importing
  them unconditionally is safe and makes the template forward-compatible.
- `timing.timeit` is a call-block macro: the workload command goes in the `{% call %}`
  body; the macro wraps it with the configured timing tool.
- `hwpmc.pmcstat(iter)` prepends the `pmcstat` invocation (or `libstatcounters` env
  vars) before the command.
- `hwpmc.pmcstat_postprocess(i)` converts sampling-mode pmcstat binary logs to text;
  it is a no-op in counting mode.
- `my_gen_output` is a callable injected into the context by the execution task via
  `self.script.extend_context({"my_gen_output": self.my_output.shell_path_builder()})`.
  It generates the per-iteration output file path as a shell expression.

### Composable Add-ons

Two mechanisms allow adding optional instrumentation to any benchmark.

#### Template Macro Libraries

These are `{% import %}`-ed into your template. Each library is a no-op when its
activating context variable is absent:

| Import | Macros provided | Activated by |
|---|---|---|
| `hwpmc.inc.jinja` | `pmcstat(iter)`, `pmcstat_postprocess(iter)` | `hwpmc_config` in context |
| `timing.inc.jinja` | `timeit(iter)` call-block | `timing_config` in context |
| `c18n_utrace.inc.jinja` | `c18n_utrace(iter)`, `post_process_iteration()` | `c18n_utrace_config` in context |

#### Auxiliary Generator Tasks

These are separate `ExecutionTask` subclasses that contribute context variables to the
shared `ScriptContext` without replacing the template. Add them to the `generators` list
in the pipeline config alongside the primary workload task:

```json
{
  "generators": [
    { "handler": "mybenchmark.exec", "task_options": { ... } },
    { "handler": "pmc.exec", "task_options": { "group": "instr" } }
  ]
}
```

| Handler | Contributes | Pairs with |
|---|---|---|
| `pmc.exec` | `hwpmc_config`, `hwpmc_counters`, `hwpmc_gen_output`, `PMCType` enum | `hwpmc.inc.jinja` |
| `generic.timing` | `timing_config`, `timing_gen_output_path`, `TimingTool` enum | `timing.inc.jinja` |
| `generic.sysctl` | `sysctl_config`, sysctl output paths; registers iter hooks | `sysctl.hook.jinja` (auto) |

An auxiliary generator **must not** call `self.script.set_template()`. Only the primary
benchmark task controls the script structure. An auxiliary generator only calls
`self.script.extend_context({...})` and, if needed, `self.script.register_global(...)`.

The `pmc.exec` task (`pycheribenchplot/pmc/pmc_exec.py`) is the canonical example of an
auxiliary generator. It provides `PMCType` and `PMCSet` enums (with pre-defined counter
groups such as `Instr`, `CheriInstr`, `L1Cache`, `L2Cache`, `Branch`, `TLB`, `Revocation`)
and supports both `hwpmc` (via `pmcstat(8)`) and `statcounters` (via `libstatcounters`)
backends, controlled by the `pmc_type` config field.

---

## Adding a New Benchmark Module

### Step 1 — Create the Module

```
pycheribenchplot/
└── mybenchmark/
    ├── __init__.py     # must import all submodules
    ├── exec.py         # ExecutionTask + loader
    └── plot.py         # SlicePlotTask
```

The `__init__.py` must import all submodules so that tasks are registered at import time:

```python
# pycheribenchplot/mybenchmark/__init__.py
from . import exec, plot
```

Register the new module by adding an import to `pycheribenchplot/__init__.py` (or
wherever the top-level module imports are collected for your entry point).

### Step 2 — Define the Configuration

```python
# pycheribenchplot/mybenchmark/exec.py
from dataclasses import dataclass
from pathlib import Path
from ..core.config import Config, ConfigPath, config_field

@dataclass
class MyExecConfig(Config):
    """Configure the My Benchmark workload."""

    binary_path: ConfigPath = config_field(
        Config.REQUIRED,
        desc="Path to the benchmark binary on the target host.",
    )
    extra_args: str = config_field(
        "",
        desc="Additional command-line arguments passed verbatim to the binary.",
    )
```

### Step 3 — Implement the Execution Task

```python
from ..core.artefact import RemoteBenchmarkIterationTarget
from ..core.task import ExecutionTask, output

class MyExecTask(ExecutionTask):
    """
    Execute My Benchmark and collect JSON output.

    This task runs my-benchmark-binary for each parameterisation combination
    and collects the output for each iteration.
    Integration with pmc is supported via the pmc.exec auxiliary generator.
    """

    task_namespace = "mybenchmark"
    task_name = "exec"
    public = True
    task_config_class = MyExecConfig

    @output
    def results(self):
        return RemoteBenchmarkIterationTarget(
            self, "results", loader=LoadMyResults, ext="json"
        )

    def run(self):
        super().run()  # must be called first — dispatches hooks
        self.script.set_template("mybenchmark.sh.jinja")
        self.script.extend_context({
            "my_config": self.config,
            "my_gen_output": self.results.shell_path_builder(),
        })
```

`shell_path_builder()` returns a callable that, when called with the current iteration
variable (e.g. `"${ITERATION}"`), produces the correct shell expression for the
per-iteration output file path. Pass it to the template as a callable context variable.

### Step 4 — Write the Loader

```python
from pathlib import Path
import polars as pl
from ..core.artefact import DataFrameLoadTask

class LoadMyResults(DataFrameLoadTask):
    """Load and normalise My Benchmark output files."""

    task_namespace = "mybenchmark"
    task_name = "ingest"

    def _load_one(self, path: Path) -> pl.DataFrame:
        """Parse a single iteration output file into a Polars DataFrame."""
        # ... parse JSON / CSV / text ...
        return pl.DataFrame({"metric": [value]})
```

`DataFrameLoadTask` handles the iteration loop and concatenation. You only need to
implement `_load_one` for a single file.

### Step 5 — Write the Template

Create `pycheribenchplot/templates/mybenchmark.sh.jinja` (see the template guide above).

---

## Adding an Analysis / Plot Task

### Choosing the Base Class

| Use case | Base class |
|---|---|
| One plot per combination of `fixed_axes` values | `SlicePlotTask` |
| A single plot spanning the full session | `PlotTask` |

For most benchmarks, `SlicePlotTask` is the right choice. The `GenericAnalysisTask`
dispatcher (`analysis.dynamic`) instantiates slice tasks automatically for each fixed-axes
combination, so the task itself never needs to split the data manually.

### Implementing a SlicePlotTask

```python
# pycheribenchplot/mybenchmark/plot.py
from dataclasses import dataclass
import polars as pl

from ..core.config import config_field
from ..core.plot import PlotTarget, SlicePlotTask
from ..core.plot_util import BarPlotConfig, PlotGrid, PlotGridConfig, grid_barplot
from ..core.task import dependency, output
from .exec import MyExecTask


@dataclass
class MyPlotConfig(PlotGridConfig, BarPlotConfig):
    """
    Configuration for My Benchmark bar plots.
    """
    drop_baseline_from_relative: bool = config_field(
        True,
        desc="Omit the baseline row from delta and overhead metric views.",
    )


class MySlicePlotTask(SlicePlotTask):
    """
    Bar plot of My Benchmark metrics, split by the configured fixed axes.

    Produces absolute, delta, and overhead views of the primary metric.
    """

    task_namespace = "mybenchmark"
    task_name = "bar-plot-slice"
    public = True
    task_config_class = MyPlotConfig

    @dependency
    def data(self):
        for bench in self.slice_benchmarks:
            task = bench.find_exec_task(MyExecTask)
            yield task.results.get_loader()

    @output
    def plot(self):
        return PlotTarget(self, "metric")

    def run_plot(self):
        # 1. Collect raw data from all loaders in this slice.
        df = pl.concat(
            [loader.df.get() for loader in self.data],
            how="vertical",
            rechunk=True,
        )

        # 2. Compute statistics (absolute / delta / overhead).
        stats = self.compute_overhead(df, "metric", how="median", overhead_scale=100)

        # 3. Optionally drop the baseline row from relative views.
        if self.config.drop_baseline_from_relative:
            view_df = stats.filter(
                (pl.col("_metric_type") == "absolute") | ~pl.col("_is_baseline")
            )
        else:
            view_df = stats

        # 4. Render the plot via PlotGrid.
        with PlotGrid(self.plot, view_df, self.config) as grid:
            grid.map(
                grid_barplot,
                x=self.config.tile_xaxis,
                y="metric",
                err=["metric_low", "metric_high"],
                config=self.config,
            )
            grid.add_legend()
```

### PlotGridConfig Reference

`PlotGridConfig` fields that control tiling and axis mapping all accept `ColRef` strings.
A `ColRef` is a column name wrapped in angle brackets (e.g. `<my_param>`). This syntax
decouples the config key from any display label that `derived_columns` may assign:

| Field | Purpose |
|---|---|
| `tile_row` | DataFrame column used to create plot rows |
| `tile_col` | DataFrame column used to create plot columns |
| `tile_xaxis` | Column mapped to the x-axis within each tile |
| `tile_yaxis` | Column mapped to the y-axis (default y for `grid.map`) |
| `tile_hue` | Column mapped to bar/line colour |
| `derived_columns` | List of `DerivedColumnConfig` — renames, remaps, unit conversions |

The `_metric_type` column (`"absolute"`, `"delta"`, `"overhead"`) is a synthetic axis
that can be referenced as `<_metric_type>` in `tile_col` or `tile_row` to split plots by
metric kind.

---

## Testing

Tests live in `tests/` and are run with `pytest`.

Guidelines:

- **Unit-test configuration classes**: verify that `config_field` defaults are correct,
  that `Config.REQUIRED` fields raise a validation error when absent, and that marshmallow
  validators catch invalid values. See `tests/test_config.py` for patterns.
- **Unit-test loaders**: construct a `Path` to a fixture file in `tests/assets/` and call
  `_load_one` directly; assert the resulting DataFrame schema and values.
- **Unit-test analysis logic**: build a synthetic Polars DataFrame, call
  `compute_overhead`, and assert that the `_metric_type` column contains the expected
  rows and that confidence intervals are present.
- **Integration tests**: use helper utilities in `tests/util/` to construct a minimal
  session, run the scheduler against a mocked `ScriptContext`, and verify that `run()`
  produces the expected context variables and template name.

---

## Common Pitfalls

- **Forgetting `__init__.py` imports**: task classes are only registered when their module
  is imported. If `from . import exec` is absent, the task will not appear in
  `benchplot-cli info task` and cannot be referenced in config files.

- **Using `dataclasses.field()` instead of `config_field()`**: this bypasses marshmallow
  serialization, breaks JSON loading, and hides the field from `benchplot-cli info`.

- **Not calling `super().run()`**: `ExecutionTask.run()` dispatches hooks registered by
  auxiliary generators. Skipping it silently disables sysctl collection, pmc setup, and
  any other hook-based add-on.

- **Calling `set_template()` from an auxiliary generator**: only the primary benchmark
  task may call `set_template()`. An auxiliary generator must only call
  `extend_context()` and `register_global()`, otherwise it will override the template
  chosen by the primary task.

- **Hardcoding output paths**: always obtain paths through `Target.iter_paths()` (Python
  side) or `shell_path_builder()` (template side). Hardcoded paths break when the session
  directory changes or when multiple parameterisation combinations produce overlapping names.

- **Using pandas instead of Polars**: the framework exclusively uses Polars. Mixing in
  pandas will cause type errors at concatenation boundaries.
