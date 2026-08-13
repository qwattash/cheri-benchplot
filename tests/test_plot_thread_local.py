"""
Tests for ThreadLocalRcParams — the thread-local proxy that replaces
mpl.rcParams in pycheribenchplot.core.plot.

Each test exercises a different aspect of isolation:
  1. Two concurrent PlotTaskMixin tasks each see only their own rc_params
     override while both rc_context blocks are simultaneously active.
  2. A direct rcParams assignment in one thread is invisible to another thread.
  3. mpl.rc_context correctly restores the thread-local state after it exits.
  4. After two concurrent tasks both finish, neither thread's rcParams is
     polluted by the other's override.
  5. A freshly-spawned thread always starts from _global_rc_blueprint,
     regardless of what the spawning thread has written.
"""

import threading
from unittest.mock import MagicMock

import matplotlib as mpl
import pytest

from pycheribenchplot.core.plot import PlotTaskMixin, _global_rc_blueprint


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _run_thread(target, errors, thread_id):
    """
    Run *target()* inside a thread, storing any raised exception in *errors*
    under *thread_id*.  This lets the main thread re-raise thread failures so
    pytest reports them as test failures rather than silent hangs.
    """
    try:
        target()
    except Exception as exc:
        errors[thread_id] = exc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_concurrent_rc_params_isolation():
    """
    Two concurrent _run_with_plot_sandbox calls with different font.size
    overrides must each observe only their own value while both rc_context
    blocks are simultaneously active.

    Thread timeline:
        t1: [rc_context enter + update] ──BARRIER── [read rcParams]
        t2: [rc_context enter + update] ──BARRIER── [read rcParams]

    The barrier guarantees both threads are inside their respective contexts
    at the moment of the read, maximising the exposure window for any
    isolation failure.
    """
    barrier = threading.Barrier(2, timeout=5)
    results = {}
    errors = {}

    class TaskA(PlotTaskMixin):
        rc_params = {"font.size": 7.77}
        logger = MagicMock()

        def run_plot(self):
            barrier.wait()
            results["A"] = mpl.rcParams["font.size"]

    class TaskB(PlotTaskMixin):
        rc_params = {"font.size": 13.13}
        logger = MagicMock()

        def run_plot(self):
            barrier.wait()
            results["B"] = mpl.rcParams["font.size"]

    t1 = threading.Thread(
        target=_run_thread,
        args=(TaskA()._run_with_plot_sandbox, errors, "t1"),
    )
    t2 = threading.Thread(
        target=_run_thread,
        args=(TaskB()._run_with_plot_sandbox, errors, "t2"),
    )
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    for exc in errors.values():
        raise exc

    assert results["A"] == pytest.approx(7.77)
    assert results["B"] == pytest.approx(13.13)


def test_thread_local_baseline_independence():
    """
    A direct mpl.rcParams['font.size'] write in thread 1 must not affect
    thread 2's view of that key.  Thread 2 should see the value from
    _global_rc_blueprint, not thread 1's assignment.

    Thread timeline:
        t1: [write font.size=99.9] ──BARRIER──
        t2:                         ──BARRIER── [read font.size → blueprint value]
    """
    barrier = threading.Barrier(2, timeout=5)
    results = {}
    errors = {}

    def thread1():
        mpl.rcParams["font.size"] = 99.9
        barrier.wait()

    def thread2():
        barrier.wait()
        results["font.size"] = mpl.rcParams["font.size"]

    t1 = threading.Thread(target=_run_thread, args=(thread1, errors, "t1"))
    t2 = threading.Thread(target=_run_thread, args=(thread2, errors, "t2"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    for exc in errors.values():
        raise exc

    assert results["font.size"] == pytest.approx(_global_rc_blueprint["font.size"])


def test_rc_context_restores_within_thread():
    """
    Within a single thread, mpl.rc_context must restore the thread-local
    rcParams to their pre-context values when the block exits.  This verifies
    that ThreadLocalRcParams correctly delegates _update_raw (the restore
    call inside rc_context's finally clause) to the thread-local RcParams.
    """
    # Capture the thread-local baseline before entering the context.
    # (This also initialises the thread-local config if it hasn't been yet.)
    baseline = mpl.rcParams["font.size"]
    override = 99.9
    assert baseline != pytest.approx(override), (
        "Chosen override value must differ from the baseline for the test to be meaningful"
    )

    with mpl.rc_context({"font.size": override}):
        assert mpl.rcParams["font.size"] == pytest.approx(override)

    assert mpl.rcParams["font.size"] == pytest.approx(baseline)


def test_no_cross_contamination_after_concurrent_contexts():
    """
    After two concurrent _run_with_plot_sandbox calls both complete, each
    thread's rcParams must be restored to the thread-local baseline
    (_global_rc_blueprint values), not left at either task's override.

    The barrier ensures the tasks are concurrent (not sequential), so that
    any cleanup race would be visible.
    """
    barrier = threading.Barrier(2, timeout=5)
    post_results = {}
    errors = {}

    class TaskA(PlotTaskMixin):
        rc_params = {"font.size": 7.77}
        logger = MagicMock()

        def run_plot(self):
            barrier.wait()

    class TaskB(PlotTaskMixin):
        rc_params = {"font.size": 13.13}
        logger = MagicMock()

        def run_plot(self):
            barrier.wait()

    def run_a():
        TaskA()._run_with_plot_sandbox()
        post_results["A"] = mpl.rcParams["font.size"]

    def run_b():
        TaskB()._run_with_plot_sandbox()
        post_results["B"] = mpl.rcParams["font.size"]

    t1 = threading.Thread(target=_run_thread, args=(run_a, errors, "t1"))
    t2 = threading.Thread(target=_run_thread, args=(run_b, errors, "t2"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    for exc in errors.values():
        raise exc

    blueprint_font_size = _global_rc_blueprint["font.size"]
    assert post_results["A"] == pytest.approx(blueprint_font_size)
    assert post_results["B"] == pytest.approx(blueprint_font_size)


def test_thread_local_fresh_init_from_blueprint():
    """
    A freshly-spawned thread must initialise its thread-local rcParams from
    _global_rc_blueprint, regardless of any writes the spawning thread has
    made to its own thread-local copy.
    """
    # Modify the calling thread's local copy.
    mpl.rcParams["font.size"] = 55.5

    results = {}
    errors = {}

    def new_thread():
        results["font.size"] = mpl.rcParams["font.size"]

    t = threading.Thread(target=_run_thread, args=(new_thread, errors, "t"))
    t.start()
    t.join()

    for exc in errors.values():
        raise exc

    assert results["font.size"] == pytest.approx(_global_rc_blueprint["font.size"])
