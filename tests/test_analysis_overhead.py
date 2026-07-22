import numpy as np
import polars as pl
import pytest

from pycheribenchplot.core.analysis import BootstrapAnalysisTask
from pycheribenchplot.core.config import AnalysisConfig

from .util.session import TaskFactory, task_factory  # noqa: F401 (task_factory used indirectly)


# Tests for compute_overhead() — the median/bootstrap path (_do_median_bootstrap).
#
# The function computes three statistics for each parameterisation group relative
# to a designated baseline group:
#   - "absolute"  — median of the raw metric values
#   - "delta"     — median(group) - median(baseline)
#   - "overhead"  — (median(group) / median(baseline) - 1) * overhead_scale
#
# Confidence intervals for each statistic are produced via BCa bootstrap
# (scipy.stats.bootstrap with vectorized=True).

# ---------------------------------------------------------------------------
# Helper concrete subclass — no real analysis logic needed
# ---------------------------------------------------------------------------


class MinimalAnalysisTask(BootstrapAnalysisTask):
    """Concrete AnalysisTask subclass used purely to exercise compute_overhead."""

    task_namespace = "analysis.test"
    task_name = "overhead"

    def run(self):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_target_factory(tmp_path):
    """
    TaskFactory with two benchmark configurations:
      - target="default"     (built-in first config, used as baseline)
      - target="comparison"  (added via add_configuration)
    """
    factory = TaskFactory(tmp_path)
    factory.add_configuration("comparison", {"target": "comparison"}, iterations=3)
    factory.build()
    return factory


@pytest.fixture
def analysis_task(two_target_factory):
    """
    MinimalAnalysisTask on the two-target session with target="default" as baseline.
    """
    analysis_config = AnalysisConfig(baseline={"target": "default"})
    return two_target_factory.build_task(
        MinimalAnalysisTask, analysis_config=analysis_config
    )


# ---------------------------------------------------------------------------
# Data-construction helpers
# ---------------------------------------------------------------------------


def _make_df(baseline_values, comparison_values):
    """
    Build a polars DataFrame with columns [target, iteration, value] from two
    lists of per-iteration values.  Iteration numbers are 1-based.
    """
    rows = [("default", i + 1, float(v)) for i, v in enumerate(baseline_values)] + [
        ("comparison", i + 1, float(v)) for i, v in enumerate(comparison_values)
    ]
    targets, iterations, values = zip(*rows)
    return pl.DataFrame(
        {
            "target": list(targets),
            "iteration": list(iterations),
            "value": list(values),
        }
    )


def _get_row(result: pl.DataFrame, metric_type: str, target: str) -> dict:
    """Extract a single named row by _metric_type and target, asserting uniqueness."""
    rows = result.filter(
        (pl.col("_metric_type") == metric_type) & (pl.col("target") == target)
    )
    assert rows.shape[0] == 1, (
        f"Expected exactly 1 row for (metric_type={metric_type!r}, target={target!r}), "
        f"got {rows.shape[0]}"
    )
    return rows.row(0, named=True)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestMedianBootstrap:
    # --- Output structure ---------------------------------------------------

    def test_output_columns(self, analysis_task):
        """Output contains exactly the expected set of columns."""
        df = _make_df([1, 2, 3], [10, 20, 30])
        result = analysis_task.compute_overhead(df, "value", how="median")
        assert set(result.columns) == {
            "target",
            "value",
            "value_low",
            "value_high",
            "_metric_type",
            "_is_baseline",
        }

    def test_output_row_count(self, analysis_task):
        """3 metric types × 2 targets produces 6 output rows."""
        df = _make_df([1, 2, 3], [10, 20, 30])
        result = analysis_task.compute_overhead(df, "value", how="median")
        assert result.shape[0] == 6

    def test_metric_type_values(self, analysis_task):
        """_metric_type column contains exactly {'absolute', 'delta', 'overhead'}."""
        df = _make_df([1, 2, 3], [10, 20, 30])
        result = analysis_task.compute_overhead(df, "value", how="median")
        assert set(result["_metric_type"].unique().to_list()) == {
            "absolute",
            "delta",
            "overhead",
        }

    def test_baseline_flag(self, analysis_task):
        """_is_baseline is True only for rows belonging to the baseline target."""
        df = _make_df([1, 2, 3], [10, 20, 30])
        result = analysis_task.compute_overhead(df, "value", how="median")
        baseline_rows = result.filter(pl.col("target") == "default")
        comparison_rows = result.filter(pl.col("target") == "comparison")
        assert all(baseline_rows["_is_baseline"].to_list())
        assert not any(comparison_rows["_is_baseline"].to_list())

    # --- Point-estimate correctness -----------------------------------------

    def test_absolute_median_correct(self, analysis_task):
        """'absolute' rows hold the correct per-group medians."""
        df = _make_df([1, 2, 3], [10, 20, 30])
        result = analysis_task.compute_overhead(df, "value", how="median")
        baseline_row = _get_row(result, "absolute", "default")
        comparison_row = _get_row(result, "absolute", "comparison")
        assert baseline_row["value"] == pytest.approx(2.0, rel=1e-6)
        assert comparison_row["value"] == pytest.approx(20.0, rel=1e-6)

    def test_delta_baseline_is_zero(self, analysis_task):
        """Delta of the baseline group against itself must be exactly 0."""
        df = _make_df([10, 20, 30], [15, 25, 35])
        result = analysis_task.compute_overhead(df, "value", how="median")
        row = _get_row(result, "delta", "default")
        assert row["value"] == pytest.approx(0.0, abs=1e-10)

    def test_delta_comparison_correct(self, analysis_task):
        """Delta equals median(comparison) - median(baseline) for known data."""
        # baseline median = 20, comparison median = 25 → delta = 5
        df = _make_df([10, 20, 30], [15, 25, 35])
        result = analysis_task.compute_overhead(df, "value", how="median")
        row = _get_row(result, "delta", "comparison")
        assert row["value"] == pytest.approx(5.0, rel=1e-6)

    def test_overhead_baseline_is_zero(self, analysis_task):
        """Overhead of the baseline group against itself must be exactly 0."""
        df = _make_df([10, 20, 30], [15, 25, 35])
        result = analysis_task.compute_overhead(df, "value", how="median")
        row = _get_row(result, "overhead", "default")
        assert row["value"] == pytest.approx(0.0, abs=1e-10)

    def test_overhead_comparison_2x(self, analysis_task):
        """When comparison values are exactly 2× baseline, overhead == 1.0 (scale=1)."""
        # baseline median = 20, comparison median = 40 → (40/20 - 1) * 1 = 1.0
        df = _make_df([10, 20, 30], [20, 40, 60])
        result = analysis_task.compute_overhead(df, "value", how="median")
        row = _get_row(result, "overhead", "comparison")
        assert row["value"] == pytest.approx(1.0, rel=1e-6)

    def test_overhead_scale(self, analysis_task):
        """overhead_scale multiplies the overhead value linearly."""
        # Same 2× data, scale=100 → overhead = 100.0
        df = _make_df([10, 20, 30], [20, 40, 60])
        result = analysis_task.compute_overhead(
            df, "value", how="median", overhead_scale=100
        )
        row = _get_row(result, "overhead", "comparison")
        assert row["value"] == pytest.approx(100.0, rel=1e-6)

    def test_inverted_overhead(self, analysis_task):
        """inverted=True negates the overhead relative to inverted=False."""
        df = _make_df([10, 20, 30], [20, 40, 60])
        normal = analysis_task.compute_overhead(df, "value", how="median")
        inverted = analysis_task.compute_overhead(
            df, "value", how="median", inverted=True
        )
        normal_val = _get_row(normal, "overhead", "comparison")["value"]
        inverted_val = _get_row(inverted, "overhead", "comparison")["value"]
        assert inverted_val == pytest.approx(-normal_val, rel=1e-6)

    # --- Confidence interval presence --------------------------------------

    def test_single_iteration_ci_is_nan(self, analysis_task):
        """
        With only one observation per group there is nothing to bootstrap —
        both CI bounds must be NaN for every output row.
        """
        df = _make_df([5], [10])
        result = analysis_task.compute_overhead(df, "value", how="median")
        assert result["value_low"].is_nan().sum() == result.shape[0]
        assert result["value_high"].is_nan().sum() == result.shape[0]

    def test_multi_iteration_ci_not_null(self, analysis_task):
        """
        With multiple observations per group, BCa bootstrap must produce
        non-null CI bounds for every output row.
        """
        rng = np.random.default_rng(0)
        baseline_vals = rng.uniform(1, 10, 10).tolist()
        comparison_vals = rng.uniform(2, 20, 10).tolist()
        df = _make_df(baseline_vals, comparison_vals)
        result = analysis_task.compute_overhead(df, "value", how="median")
        assert result["value_low"].null_count() == 0
        assert result["value_high"].null_count() == 0

    def test_ci_ordering(self, analysis_task):
        """
        For every row with non-null CI bounds: value_low <= value <= value_high.
        BCa bootstrap should produce CIs that bracket the point estimate.
        """
        rng = np.random.default_rng(1)
        baseline_vals = rng.uniform(1, 10, 10).tolist()
        comparison_vals = rng.uniform(2, 20, 10).tolist()
        df = _make_df(baseline_vals, comparison_vals)
        result = analysis_task.compute_overhead(df, "value", how="median")
        for row in result.iter_rows(named=True):
            lo, val, hi = row["value_low"], row["value"], row["value_high"]
            if np.isnan(lo) or np.isnan(hi):
                continue
            assert lo <= val <= hi, (
                f"CI ordering violated: low={lo}, value={val}, high={hi} "
                f"(target={row['target']!r}, _metric_type={row['_metric_type']!r})"
            )

    # --- Edge cases --------------------------------------------------------

    def test_zero_baseline_overhead_nan(self, analysis_task):
        """When the baseline median is zero, overhead must be NaN (not an error)."""
        df = _make_df([0, 0, 0], [1, 2, 3])
        result = analysis_task.compute_overhead(df, "value", how="median")
        row = _get_row(result, "overhead", "comparison")
        val = row["value"]
        # Polars may return a Python float NaN or None for a missing/NaN cell
        is_nan = (val is None) or (isinstance(val, float) and np.isnan(val))
        assert is_nan, f"Expected NaN overhead for zero baseline, got {val!r}"

    def test_extra_groupby(self, analysis_task):
        """
        extra_groupby adds a further grouping dimension.
        2 targets × 2 scenarios × 3 metric types = 12 output rows, and both
        scenario labels must appear in the result.
        """
        rows = [
            (target, iteration, float(iteration * 10), scenario)
            for target in ("default", "comparison")
            for scenario in ("A", "B")
            for iteration in (1, 2, 3)
        ]
        df = pl.DataFrame(
            {
                "target": [r[0] for r in rows],
                "iteration": [r[1] for r in rows],
                "value": [r[2] for r in rows],
                "scenario": [r[3] for r in rows],
            }
        )
        result = analysis_task.compute_overhead(
            df, "value", how="median", extra_groupby=["scenario"]
        )
        assert result.shape[0] == 12
        assert set(result["scenario"].unique().to_list()) == {"A", "B"}

    # --- Ordering independence ---------------------------------------------

    def test_order_independence(self, analysis_task):
        """
        The result must be identical regardless of input row order.

        Alignment between each group and its paired baseline observations is
        performed via a key-based join on the iteration column, not positional
        alignment.  Additionally, the median statistic is inherently
        order-independent.  This test verifies both properties together.
        """
        df = _make_df([10, 20, 30, 15, 25], [20, 40, 60, 30, 50])

        result_natural = analysis_task.compute_overhead(df, "value", how="median")

        df_shuffled = df.sample(fraction=1.0, shuffle=True, seed=42)
        result_shuffled = analysis_task.compute_overhead(
            df_shuffled, "value", how="median"
        )

        # Normalise to a deterministic sort order before comparing
        sort_keys = ["target", "_metric_type"]
        r_nat = result_natural.sort(sort_keys)
        r_shuf = result_shuffled.sort(sort_keys)

        assert r_nat.equals(r_shuf), (
            "compute_overhead produced different results for naturally-ordered "
            "vs shuffled input — possible positional alignment bug.\n"
            f"Natural:\n{r_nat}\nShuffled:\n{r_shuf}"
        )
