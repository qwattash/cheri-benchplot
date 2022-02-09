from pathlib import Path

import pytest

def pytest_addoption(parser):
    parser.addoption("--benchplot-user-config", type=Path)
    parser.addoption("--run-qemu-trace", action="store_true", default=False, help="run slow qemu tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "qemu_trace: mark qemu tracing test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-qemu-trace"):
        # do not skip qemu tests
        return
    skip = pytest.mark.skip(reason="need --run-qemu-trace option to run")
    for item in items:
        if "qemu_trace" in item.keywords:
            item.add_marker(skip)


