import asyncio as aio
import io
import uuid
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pycheribenchplot.core.benchmark import Benchmark
from pycheribenchplot.core.config import (BenchmarkRunConfig, BenchplotUserConfig, DatasetName, SessionRunConfig)
from pycheribenchplot.core.dataset import DatasetArtefact, DataSetContainer
from pycheribenchplot.core.pipeline import PipelineManager
from pycheribenchplot.core.session import PipelineSession

fake_session = """
{
    "uuid": "17856370-2fd1-4597-937a-42b1277da44f",
    "name": "benchplot-selftest",
    "configurations": []
}
"""

fake_benchmark = """
{
    "name": "selftest0",
    "iterations": 2,
    "benchmark": {
        "handler": "test-fake"
    },
    "uuid": "8bc941a3-f6d6-4d37-b193-4738f1da3dae",
    "g_uuid": "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
    "instance": {
        "kernel": "selftest-kernel",
        "baseline": true,
        "name": "selftest-instance",
        "cheribuild_kernel": false
    }
}
"""


class FakeDataset(DataSetContainer):
    dataset_config_name = DatasetName.TEST_FAKE
    dataset_source_id = DatasetArtefact.TEST_FAKE


@pytest.fixture
def benchplot_user_config(pytestconfig):
    user_config = BenchplotUserConfig.load_json(pytestconfig.getoption("--benchplot-user-config"))
    return user_config


@pytest.fixture
def fake_simple_benchmark(pytestconfig, tmp_path, mocker):
    """
    A fake benchmark instance with the default user configuration
    """
    mgr = PipelineManager(BenchplotUserConfig())
    config = SessionRunConfig.from_json(fake_session)
    bench_config = BenchmarkRunConfig.from_json(fake_benchmark)
    session = PipelineSession(mgr, config, session_path=tmp_path)
    session.clean()
    return Benchmark(session, bench_config)


@pytest.fixture
def fake_existing_path():
    path = Path("fake/output/file")
    mock_path = MagicMock(wraps=path)
    mock_path.exists.return_value = True
    mock_path.is_file.return_value = True
    mock_path.is_dir.return_value = False
    return mock_path
