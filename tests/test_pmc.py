import json
import shutil
import pytest
from pathlib import Path

from pycheribenchplot.pmc.ingest import IngestPMCStatStacks, IngestPMCStatStacksConfig

@pytest.fixture
def pmc_stacks_input(tmp_path):
    input_dir = tmp_path / "test_pmc"
    input_dir.mkdir()

    # Take a sample stacks file for a single iteration
    shutil.copy("tests/assets/test_pmc_stacks.txt", input_dir / "test_pmc.1.stacks")
    return input_dir

@pytest.fixture
def pmc_benchmark(fake_simple_benchmark):
    # Ensure we only have 1 iteration
    fake_simple_benchmark.config.iterations = 1
    fake_simple_benchmark.get_benchmark_data_path().mkdir(exist_ok=True)
    fake_simple_benchmark.get_benchmark_iter_data_path(0).mkdir(exist_ok=True)

    return fake_simple_benchmark

def test_pmc_stacks_folding(pmc_stacks_input, pmc_benchmark):
    config = IngestPMCStatStacksConfig(path=pmc_stacks_input, stack_file_pattern="test_pmc.{iteration}.stacks")
    task = IngestPMCStatStacks(pmc_benchmark, None, config)

    task.run()

    # Verify the result against stackcollapse-pmc.pl?
    with open(task.data.paths()[0], "r") as fd:
        data = json.load(fd)

    folded_stacks = data["stacks"]
    assert len(folded_stacks) == 5

    assert folded_stacks.get("sousrsend;sosend_generic;tcp_usr_send;tcp_default_output") == 45
    assert folded_stacks.get("soreceive;soreceive_generic;tcp_usr_rcvd;tcp_default_output") == 65
    assert folded_stacks.get(".L_ZN9grpc_core22ParseBackendMetricDataENSt3__117basic_string_viewIcNS0_11char_traitsIcEEEEPNS_31BackendMetricAllocatorInterfaceE$eh_alias;bar;foo") == 50
    assert folded_stacks.get("baz;bar2;foo") == 20
    assert folded_stacks.get("baz2;bar2;foo") == 20
