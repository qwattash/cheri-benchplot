import asyncio as aio
import logging
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import pytest_asyncio
from pypika import Query

from pycheribenchplot.core.benchmark import BenchmarkBase
from pycheribenchplot.core.instance import (InstanceCheriBSD, InstanceConfig, InstanceKernelABI, InstanceManager,
                                            InstancePlatform, InstanceStatus, PlatformOptions)
from pycheribenchplot.core.manager import (BenchmarkManager, BenchmarkManagerConfig, BenchmarkSessionConfig,
                                           BenchplotUserConfig)
from pycheribenchplot.core.perfetto import PerfettoDataSetContainer
from pycheribenchplot.core.util import setup_logging


@pytest.fixture(scope="session", autouse=True)
def logsetup():
    setup_logging(verbose=True)


@pytest.fixture(scope="session")
def qemu_outdir(tmp_path_factory):
    path = tmp_path_factory.mktemp("test-qemu-trace-output")
    return path


@pytest.fixture(scope="session")
def asset_path():
    return Path("tests/assets")


@pytest.fixture
def manager_config(pytestconfig, qemu_outdir):
    user_config = BenchplotUserConfig.load_json(pytestconfig.getoption("--benchplot-user-config"))
    session_config = BenchmarkSessionConfig()
    session_config.ssh_key = Path("extra_files/freebsd_test_id_rsa")
    session_config.output_path = qemu_outdir
    session_config.verbose = True
    with patch.object(BenchmarkManagerConfig, "__post_init__") as fake_post:
        # Skip sanity checks on the session config
        manager_config = BenchmarkManagerConfig.merge(user_config, session_config)
    return manager_config


@pytest.fixture
def instance_manager(manager_config, event_loop):
    return InstanceManager(event_loop, manager_config)


@pytest.fixture
def qemu_riscv_config(qemu_outdir):
    instance_config = InstanceConfig(kernel="CHERI-QEMU-NODEBUG", name="test")
    instance_config.platform = InstancePlatform.QEMU
    instance_config.cheri_target = InstanceCheriBSD.RISCV64_PURECAP
    instance_config.kernel_abi = InstanceKernelABI.HYBRID
    opts = PlatformOptions()
    opts.qemu_trace = True
    opts.qemu_trace_file = qemu_outdir / "qemu_trace.pb"
    opts.qemu_trace_categories = []  # To be determined by the test
    instance_config.platform_options = opts
    return instance_config


@pytest.fixture
def perfetto_dataset(manager_config):
    manager = Mock(BenchmarkManager)
    manager.cleanup_callbacks = []
    manager.config = manager_config
    benchmark = Mock(BenchmarkBase)
    benchmark.manager = manager
    benchmark.logger = logging.getLogger("cheri-benchplot-test")
    return PerfettoDataSetContainer(benchmark, "qemu_gen_test", Mock())


@pytest.mark.qemu_trace
@pytest.mark.asyncio
async def test_qemu_gen_ctrl(asset_path, instance_manager, qemu_riscv_config, perfetto_dataset):
    # Test setup
    qemu_riscv_config.platform_options.qemu_trace_categories = ["ctrl"]
    owner = uuid.uuid4()
    instance = await instance_manager.request_instance(owner, qemu_riscv_config)
    try:
        await instance.connect()
        await instance.import_file(asset_path / "riscv_qemu_trace_gen_ctrl", "test")
        result = await instance.run_cmd("./test")
    finally:
        await instance_manager.release_instance(owner)
        await instance_manager.shutdown()

    assert result.returncode == 0
    # Load qemu trace, use the protected interface of the perfetto dataset
    tp = perfetto_dataset._get_trace_processor(qemu_riscv_config.platform_options.qemu_trace_file)
    t_slice = perfetto_dataset.t_slice
    q = Query.from_(t_slice).select(t_slice.star).where(t_slice.category == "ctrl")
    df = perfetto_dataset._query_to_df(tp, q.get_sql(quote_char=None))
    assert len(df) == 1
    assert (df["dur"] > 0).all()
    assert (df["ts"] > 0).all()


@pytest.mark.qemu_trace
@pytest.mark.asyncio
async def test_qemu_gen_counters(asset_path, instance_manager, qemu_riscv_config, perfetto_dataset):
    # Test setup
    qemu_riscv_config.platform_options.qemu_trace_categories = ["ctrl", "counter"]
    owner = uuid.uuid4()
    instance = await instance_manager.request_instance(owner, qemu_riscv_config)
    try:
        await instance.connect()
        await instance.import_file(asset_path / "riscv_qemu_trace_gen_counter", "test")
        result = await instance.run_cmd("./test")
    finally:
        await instance_manager.release_instance(owner)
        await instance_manager.shutdown()

    assert result.returncode == 0
    # Load qemu trace and check the counter
    tp = perfetto_dataset._get_trace_processor(qemu_riscv_config.platform_options.qemu_trace_file)
    t_track = perfetto_dataset.t_track
    t_counter = perfetto_dataset.t_counter
    q = Query.from_(t_counter).join(t_track).on(t_counter.track_id == t_track.id).select(
        t_counter.star).where(t_track.name == "test-counter")
    df = perfetto_dataset._query_to_df(tp, q.get_sql(quote_char=None))
    assert len(df) == 50
    ts_sort = df.sort_values("ts", ascending=True)
    assert (ts_sort["value"] == np.arange(1, 51)).all()
