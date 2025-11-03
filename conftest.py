from pathlib import Path

import pytest

from pycheribenchplot.core.util import setup_logging


def pytest_addoption(parser):
    parser.addoption("--benchplot-user-config", type=Path, default=None)
    parser.addoption("--run-qemu-trace", action="store_true", default=False, help="run slow qemu tests")
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "qemu_trace: mark qemu tracing test")
    config.addinivalue_line("markers", "slow: mark slow test")
    config.addinivalue_line("markers", "user_config: mark test requiring a benchplot user config")
    if config.getoption("-v"):
        setup_logging(verbose=True, debug_config=True)


def pytest_collection_modifyitems(config, items):
    skip_qemu = pytest.mark.skip(reason="need --run-qemu-trace option to run")
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    skip_user_conf = pytest.mark.skip(reason="need --benchplot-user-config option to run")
    for item in items:
        if not config.getoption("--run-qemu-trace") and "qemu_trace" in item.keywords:
            item.add_marker(skip_qemu)
        if not config.getoption("--run-slow") and "slow" in item.keywords:
            item.add_marker(skip_slow)
        if not config.getoption("--benchplot-user-config") and "user_config" in item.keywords:
            item.add_marker(skip_user_conf)
