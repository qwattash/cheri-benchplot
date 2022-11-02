import asyncio as aio
import logging
import re
import subprocess
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import pytest_asyncio
from pypika import Query

from pycheribenchplot.core.benchmark import BenchmarkBase, BenchmarkRunConfig
from pycheribenchplot.core.instance import (InstanceCheriBSD, InstanceConfig, InstanceKernelABI, InstanceManager,
                                            InstancePlatform, InstanceStatus, PlatformOptions)
from pycheribenchplot.core.manager import (BenchmarkManager, BenchmarkManagerConfig, BenchmarkSessionConfig,
                                           BenchplotUserConfig)
from pycheribenchplot.core.perfetto import PerfettoDataSetContainer
from pycheribenchplot.core.util import setup_logging
from pycheribenchplot.qemu.dataset import QEMUStatsBBHistogramDataset


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


@pytest.fixture
def qemu_bb_dataset(mocker, manager_config, qemu_outdir):
    manager = Mock(BenchmarkManager)
    manager.cleanup_callbacks = []
    manager.config = manager_config
    benchmark = Mock(BenchmarkBase)
    benchmark.config = BenchmarkRunConfig("testbench", 1, [], [])
    benchmark.uuid = "bench-uuid"
    benchmark.g_uuid = "bench-guuid"
    benchmark.manager = manager
    benchmark.logger = logging.getLogger("cheri-benchplot-test")
    get_output = mocker.patch.object(QEMUStatsBBHistogramDataset, "output_file")
    get_output.return_value = qemu_outdir / "qemu_trace.pb"
    return QEMUStatsBBHistogramDataset(benchmark, "qemu_gen_test", Mock())


@pytest.fixture
def qemu_bb_expect(pytestconfig, asset_path):
    config = BenchplotUserConfig.load_json(pytestconfig.getoption("--benchplot-user-config"))
    objdump_bin = config.sdk_path / "sdk" / "bin" / "llvm-objdump"
    objdump = subprocess.run([objdump_bin, "-t", asset_path / "riscv_qemu_trace_gen_bb"], capture_output=True)
    syms = subprocess.run(["grep", "test_bb"], input=objdump.stdout, capture_output=True)
    match = re.match("([0-9a-f]+)[a-zA-Z.\s]+([0-9a-f]+)\s+test_bb", syms.stdout.decode("UTF-8"))
    assert match
    addr = int(match.group(1), 16)
    size = int(match.group(2), 16)
    return (addr, size)


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
    df = perfetto_dataset._query_to_df(tp, q)
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
    df = perfetto_dataset._query_to_df(tp, q)
    assert len(df) == 50
    ts_sort = df.sort_values("ts", ascending=True)
    assert (ts_sort["value"] == np.arange(1, 51)).all()


@pytest.mark.qemu_trace
@pytest.mark.asyncio
async def test_qemu_gen_bb(asset_path, instance_manager, qemu_riscv_config, qemu_bb_dataset, qemu_bb_expect):
    # Test setup
    qemu_riscv_config.platform_options.qemu_trace_categories = ["ctrl", "stats"]
    owner = uuid.uuid4()
    instance = await instance_manager.request_instance(owner, qemu_riscv_config)
    try:
        await instance.connect()
        await instance.import_file(asset_path / "riscv_qemu_trace_gen_bb", "test")
        result = await instance.run_cmd("./test")
    finally:
        await instance_manager.release_instance(owner)
        await instance_manager.shutdown()
    assert result.returncode == 0

    # Load qemu trace and check the dataset
    qemu_bb_dataset.load()
    df = qemu_bb_dataset.df

    addr, size = qemu_bb_expect
    # To find the relocation base, use the heuristic that the base is always page aligned
    test_base = df["start"] - addr
    base = df[(test_base & 0x0fff) == 0]
    assert len(base) == 1
    test_bb_addr = base["start"][0]
    test_bb_end = test_bb_addr + size

    intervals = df[(df["start"] >= test_bb_addr) & (df["end"] <= test_bb_end)]
    for _, i in intervals.iterrows():
        print(f"{int(i['start']):x}-{int(i['end']):x} {i['hit_count']}")
    # Minimal smoke test for now, do not check the counts
    assert len(intervals) == 3
