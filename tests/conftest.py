import uuid
from enum import Enum
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pycheribenchplot.core.dataset import DatasetRegistry, DatasetName, DataSetContainer
from pycheribenchplot.core.manager import BenchmarkManager, BenchmarkSessionConfig, BenchplotUserConfig
from pycheribenchplot.core.benchmark import BenchmarkBase, BenchmarkRunConfig, BenchmarkDataSetConfig
from pycheribenchplot.core.instance import InstanceConfig, InstancePlatform, InstanceCheriBSD, InstanceKernelABI


def pytest_addoption(parser):
    parser.addoption("--benchplot-user-config", type=Path)


@pytest.fixture
def fake_simple_benchmark(pytestconfig, tmp_path, mocker):
    # name_items = dict(DatasetName.__members__.items())
    # name_items.update(FAKE="__fake__")
    # FakeDatasetName = Enum("DatasetName", names=name_items)
    # mocker.patch("pycheribenchplot.core.dataset.DatasetName", FakeDatasetName)
    # fake_enum = mocker.patch("pycheribenchplot.core.dataset.DatasetName")

    ds_config = BenchmarkDataSetConfig(type="__fake__")
    b_config = BenchmarkRunConfig(name="fake-test", iterations=1,
                                  benchmark_dataset=ds_config,
                                  datasets={})
    i_config = InstanceConfig(
        kernel="CHERI-QEMU",
        baseline=True,
        platform=InstancePlatform.QEMU,
        cheri_target=InstanceCheriBSD.RISCV64_PURECAP,
        kernelabi=InstanceKernelABI.HYBRID)
    s_config = BenchmarkSessionConfig(benchmarks=[b_config], instances=[i_config])
    s_config.output_path = tmp_path

    user_config_path = pytestconfig.getoption("--benchplot-user-config", default=None)
    if user_config_path:
        user_config = BenchplotUserConfig.load_json(user_config_path)
    else:
        user_config = BenchplotUserConfig()
    manager = BenchmarkManager(user_config, s_config)

    fake_id = uuid.UUID(bytes=b"\x00" * 16)
    fake_dsname = mocker.patch("pycheribenchplot.core.benchmark.DatasetName")
    fake_registry = mocker.patch("pycheribenchplot.core.benchmark.DatasetRegistry")
    yield BenchmarkBase(manager, b_config, i_config, fake_id)


@pytest.fixture
def fake_existing_path():
    path = Path("fake/output/file")
    mock_path = MagicMock(wraps=path)
    mock_path.exists.return_value = True
    mock_path.is_file.return_value = True
    mock_path.is_dir.return_value = False
    return mock_path


