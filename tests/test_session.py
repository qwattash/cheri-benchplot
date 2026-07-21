import json
from pathlib import Path

import pytest

from pycheribenchplot.core.config import (
    AssetConfig,
    AssetImportAction,
    BenchplotUserConfig,
    InstanceConfig,
    PipelineBenchmarkConfig,
    PipelineConfig,
    SystemConfig,
)
from pycheribenchplot.core.session import Session, SESSION_RUN_FILE
from .util.session import TaskFactory


@pytest.fixture
def user_config(tmp_path):
    config = BenchplotUserConfig()
    config.session_path = tmp_path
    return config


# ==============================================================================
# Test Helpers for Session Merge Tests
# ==============================================================================


def create_session_with_params(tmp_path, name, param_configs):
    """
    Create a session with specified parameterization.

    Args:
        tmp_path: pytest tmp_path fixture
        name: session name (used as subdirectory)
        param_configs: list of parameter dicts, e.g., [{"arch": "aarch64", "variant": "base"}]

    Returns:
        Session object with configurations matching param_configs
    """
    session_path = tmp_path / name
    session_path.mkdir(parents=True, exist_ok=True)
    factory = TaskFactory(session_path)

    if not param_configs:
        # Empty session - just build with defaults
        return factory.build()

    # Collect all parameter keys to ensure consistency
    all_keys = set()
    for params in param_configs:
        all_keys.update(params.keys())

    # Add 'target' if not explicitly provided (TaskFactory adds it by default)
    if "target" not in all_keys:
        all_keys.add("target")

    # First config: update the default configuration
    first_params = param_configs[0].copy()
    if "target" not in first_params:
        first_params["target"] = "default"
    factory._unrolled_config["configurations"][0]["parameters"].update(first_params)

    # Additional configs: add via add_configuration
    # Ensure all configs have all parameter keys
    for i, params in enumerate(param_configs[1:], start=1):
        complete_params = params.copy()
        # Fill in missing keys with default values from first config
        for key in all_keys:
            if key not in complete_params:
                if key == "target":
                    complete_params["target"] = "default"
                # Note: other missing keys should not happen in well-formed tests
        factory.add_configuration(f"config-{i}", complete_params)

    return factory.build()


def assert_merge_succeeds(dst, src, ext_params, allow_override=False):
    """
    Verify that merge completes without raising an exception.

    Args:
        dst: destination Session
        src: source Session
        ext_params: external parameters dict
        allow_override: whether to allow parameter override
    """
    try:
        dst.merge_by_key(src, ext_params, allow_override=allow_override)
    except Exception as e:
        pytest.fail(f"Merge should succeed but raised {type(e).__name__}: {e}")


def assert_merge_fails(dst, src, ext_params, expected_error, expected_msg_fragment):
    """
    Verify that merge raises the expected error with expected message fragment.

    Args:
        dst: destination Session
        src: source Session
        ext_params: external parameters dict
        expected_error: expected exception class
        expected_msg_fragment: substring that should appear in error message
    """
    with pytest.raises(expected_error) as exc_info:
        dst.merge_by_key(src, ext_params)

    error_msg = str(exc_info.value)
    assert expected_msg_fragment in error_msg, (
        f"Expected error message to contain '{expected_msg_fragment}', "
        f"but got: {error_msg}"
    )


# ==============================================================================
# Test Group 1: Subset Validation
# ==============================================================================


def test_merge_by_key_valid_subset(tmp_path):
    """Verify successful merge when source axes are a proper subset of destination."""
    # Destination: 4 configurations (arch × variant)
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [
            {"arch": "aarch64", "variant": "base"},
            {"arch": "aarch64", "variant": "cap"},
            {"arch": "riscv64", "variant": "base"},
            {"arch": "riscv64", "variant": "cap"},
        ],
    )

    # Source: 1 configuration (only arch axis)
    src = create_session_with_params(
        tmp_path,
        "src",
        [
            {"arch": "aarch64"},
        ],
    )

    # ext_params completes the space
    ext_params = {"variant": "base"}
    assert_merge_succeeds(dst, src, ext_params)


def test_merge_by_key_equal_parameterization(tmp_path):
    """Verify merge when source and destination have identical axes."""
    # Both have identical axes and values
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [
            {"arch": "aarch64", "variant": "base"},
        ],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [
            {"arch": "aarch64", "variant": "base"},
        ],
    )

    ext_params = {}
    assert_merge_succeeds(dst, src, ext_params)


def test_merge_by_key_fails_not_subset(tmp_path):
    """Verify rejection when source has extra axes not in destination."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [
            {"arch": "aarch64"},
        ],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [
            {"arch": "aarch64", "variant": "base", "opt_level": "O2"},
        ],
    )

    ext_params = {}
    assert_merge_fails(
        dst,
        src,
        ext_params,
        ValueError,
        "Invalid merge source",
    )


def test_merge_by_key_fails_disjoint_axes(tmp_path):
    """Verify rejection when source has completely different axes."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [
            {"arch": "aarch64", "variant": "base"},
        ],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [
            {"platform": "morello", "optimization": "speed"},
        ],
    )

    ext_params = {}
    assert_merge_fails(
        dst,
        src,
        ext_params,
        ValueError,
        "Invalid merge source",
    )


# ==============================================================================
# Test Group 2: External Parameters Validation
# ==============================================================================


def test_merge_by_key_ext_params_completes_space(tmp_path):
    """Verify ext_params successfully completes source parameter space."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [
            {"arch": "aarch64", "variant": "base", "opt_level": "O2"},
        ],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [
            {"arch": "aarch64"},
        ],
    )

    ext_params = {"variant": "base", "opt_level": "O2"}
    assert_merge_succeeds(dst, src, ext_params)


def test_merge_by_key_fails_incomplete_ext_params(tmp_path):
    """Verify rejection when ext_params leaves axes incomplete."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [
            {"arch": "aarch64", "variant": "base", "opt_level": "O2"},
        ],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [
            {"arch": "aarch64"},
        ],
    )

    ext_params = {"variant": "base"}  # Missing opt_level
    assert_merge_fails(
        dst,
        src,
        ext_params,
        ValueError,
        "Invalid merge extended parameters",
    )


def test_merge_by_key_fails_ext_params_duplicate_keys(tmp_path):
    """Verify rejection when ext_params contains keys already in source."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [
            {"arch": "aarch64", "variant": "cap"},
        ],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [
            {"arch": "aarch64", "variant": "base"},
        ],
    )

    ext_params = {"variant": "cap"}  # Tries to override
    assert_merge_fails(
        dst,
        src,
        ext_params,
        ValueError,
        "Invalid merge parameter extension",
    )


def test_merge_by_key_ext_params_override_allowed(tmp_path):
    """Verify ext_params can override source parameters when allow_override=True."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [
            {"arch": "aarch64", "variant": "cap"},
        ],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [
            {"arch": "aarch64", "variant": "base"},
        ],
    )

    ext_params = {"variant": "cap"}
    assert_merge_succeeds(dst, src, ext_params, allow_override=True)


def test_merge_by_key_fails_ext_params_excess_keys(tmp_path):
    """Verify rejection when ext_params introduces unknown axes."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [
            {"arch": "aarch64"},
        ],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [
            {"arch": "aarch64"},
        ],
    )

    ext_params = {"variant": "base", "extra_axis": "value"}
    assert_merge_fails(
        dst,
        src,
        ext_params,
        ValueError,
        "Invalid merge extended parameters",
    )


# ==============================================================================
# Test Group 3: Parameterization Value Validation
# ==============================================================================


def test_merge_by_key_fails_value_not_in_destination(tmp_path):
    """Verify rejection when source values do not exist in destination."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [
            {"arch": "aarch64", "variant": "base"},
            {"arch": "aarch64", "variant": "cap"},
        ],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [
            {"arch": "riscv64", "variant": "base"},
        ],
    )

    ext_params = {}
    assert_merge_fails(
        dst,
        src,
        ext_params,
        ValueError,
        "Invalid merge source values",
    )


def test_merge_by_key_value_rename_with_override(tmp_path):
    """Verify renaming source values via ext_params with allow_override."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [
            {"arch": "aarch64-new", "variant": "base"},
        ],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [
            {"arch": "aarch64-old", "variant": "base"},
        ],
    )

    ext_params = {"arch": "aarch64-new"}
    assert_merge_succeeds(dst, src, ext_params, allow_override=True)


def test_merge_by_key_fails_no_value_overlap(tmp_path):
    """Verify rejection when source and destination have no value overlap."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [
            {"arch": "aarch64", "compiler": "clang"},
        ],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [
            {"arch": "aarch64", "compiler": "gcc"},
        ],
    )

    ext_params = {}
    assert_merge_fails(
        dst,
        src,
        ext_params,
        ValueError,
        "Invalid merge source values",
    )


def test_merge_by_key_partial_value_overlap(tmp_path):
    """Verify merge succeeds with partial value overlap."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [
            {"arch": "aarch64", "opt": "O0"},
            {"arch": "aarch64", "opt": "O2"},
            {"arch": "aarch64", "opt": "O3"},
        ],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [
            {"arch": "aarch64", "opt": "O2"},
            {"arch": "aarch64", "opt": "O3"},
        ],
    )

    ext_params = {}
    assert_merge_succeeds(dst, src, ext_params)


# ==============================================================================
# Data Testing Infrastructure
# ==============================================================================


@pytest.fixture
def session_with_data(tmp_path):
    """
    Create a session with actual data files for merge testing.

    Returns a session with one benchmark context containing:
    - results.csv
    - nested/file.json
    """
    session = create_session_with_params(
        tmp_path,
        "session_with_data",
        [
            {"arch": "aarch64"},
        ],
    )

    # Add test data files to the first benchmark
    benchmark = session.all_benchmarks()[0]
    data_path = benchmark.get_benchmark_data_path()
    data_path.mkdir(parents=True, exist_ok=True)

    # Create simple data file
    (data_path / "results.csv").write_text("test,data\n1,2\n")

    # Create nested directory structure
    nested_dir = data_path / "nested"
    nested_dir.mkdir(parents=True, exist_ok=True)
    (nested_dir / "file.json").write_text('{"key": "value"}')

    return session


def assert_data_merged(dst_benchmark, expected_files):
    """
    Verify that expected files exist in destination benchmark.

    Args:
        dst_benchmark: destination Benchmark object
        expected_files: list of relative file paths to check
    """
    data_path = dst_benchmark.get_benchmark_data_path()
    for file in expected_files:
        file_path = data_path / file
        assert file_path.exists(), f"Expected {file} in merged data at {file_path}"


def assert_data_not_merged(dst_benchmark, unexpected_files):
    """
    Verify that files do NOT exist in destination.

    Args:
        dst_benchmark: destination Benchmark object
        unexpected_files: list of relative file paths that should NOT exist
    """
    data_path = dst_benchmark.get_benchmark_data_path()
    for file in unexpected_files:
        file_path = data_path / file
        assert not file_path.exists(), f"Unexpected {file} found in {file_path}"


# ==============================================================================
# Test Group 4: Benchmark Context Matching
# ==============================================================================


def test_merge_by_key_matches_correct_contexts(tmp_path):
    """Verify benchmarks are matched by parameterization, not by order."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [
            {"arch": "aarch64"},
            {"arch": "riscv64"},
            {"arch": "x86_64"},
        ],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [
            {"arch": "riscv64"},
        ],
    )

    # Add data to source
    src_benchmark = src.all_benchmarks()[0]
    src_data_path = src_benchmark.get_benchmark_data_path()
    src_data_path.mkdir(parents=True, exist_ok=True)
    (src_data_path / "test.txt").write_text("riscv64 data")

    ext_params = {}

    # Perform merge
    assert_merge_succeeds(dst, src, ext_params)

    # Verify: only middle benchmark (riscv64) should have data
    dst_benchmarks = dst.all_benchmarks()

    # First benchmark (aarch64) - no data
    assert_data_not_merged(dst_benchmarks[0], ["test.txt"])

    # Second benchmark (riscv64) - has data
    assert_data_merged(dst_benchmarks[1], ["test.txt"])

    # Third benchmark (x86_64) - no data
    assert_data_not_merged(dst_benchmarks[2], ["test.txt"])


def test_merge_by_key_duplicate_source_contexts_with_warning(tmp_path):
    """Verify multiple source contexts with same parameters generate warning."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [
            {"arch": "aarch64", "variant": "base"},
        ],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [
            {"arch": "aarch64"},
            {"arch": "aarch64"},  # Duplicate!
        ],
    )

    # Add different data to each source context
    src_benchmarks = src.all_benchmarks()

    src_data_1 = src_benchmarks[0].get_benchmark_data_path()
    src_data_1.mkdir(parents=True, exist_ok=True)
    (src_data_1 / "data.txt").write_text("first")

    src_data_2 = src_benchmarks[1].get_benchmark_data_path()
    src_data_2.mkdir(parents=True, exist_ok=True)
    (src_data_2 / "data.txt").write_text("second")

    ext_params = {"variant": "base"}

    # Perform merge - should succeed (warning will be logged to stderr)
    assert_merge_succeeds(dst, src, ext_params)

    # Verify data was merged (last one wins)
    dst_benchmark = dst.all_benchmarks()[0]
    dst_data_path = dst_benchmark.get_benchmark_data_path()
    data_file = dst_data_path / "data.txt"
    assert data_file.exists()
    # Second source context's data should be present (last duplicate wins)
    assert data_file.read_text() == "second"


def test_merge_by_key_cartesian_join(tmp_path):
    """Verify cartesian join behavior when ext_params creates multiple matches."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [
            {"arch": "aarch64", "variant": "base", "opt": "O2"},
            {"arch": "aarch64", "variant": "base", "opt": "O3"},
            {"arch": "aarch64", "variant": "cap", "opt": "O2"},
            {"arch": "aarch64", "variant": "cap", "opt": "O3"},
        ],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [
            {"arch": "aarch64", "variant": "base"},
        ],
    )

    # Add data to source
    src_benchmark = src.all_benchmarks()[0]
    src_data_path = src_benchmark.get_benchmark_data_path()
    src_data_path.mkdir(parents=True, exist_ok=True)
    (src_data_path / "marker.txt").write_text("merged")

    ext_params = {"opt": "O2"}

    # Perform merge
    assert_merge_succeeds(dst, src, ext_params)

    # Verify: only first context should have data
    dst_benchmarks = dst.all_benchmarks()

    assert_data_merged(dst_benchmarks[0], ["marker.txt"])  # Match!
    assert_data_not_merged(dst_benchmarks[1], ["marker.txt"])
    assert_data_not_merged(dst_benchmarks[2], ["marker.txt"])
    assert_data_not_merged(dst_benchmarks[3], ["marker.txt"])


def test_merge_by_key_multiple_source_matches_destination(tmp_path):
    """Verify multiple source contexts can match different destination contexts."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [
            {"arch": "aarch64", "variant": "base"},
            {"arch": "riscv64", "variant": "base"},
        ],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [
            {"arch": "aarch64"},
            {"arch": "riscv64"},
        ],
    )

    # Add different data to each source
    src_benchmarks = src.all_benchmarks()

    src_data_aarch64 = src_benchmarks[0].get_benchmark_data_path()
    src_data_aarch64.mkdir(parents=True, exist_ok=True)
    (src_data_aarch64 / "aarch64.txt").write_text("aarch64 data")

    src_data_riscv64 = src_benchmarks[1].get_benchmark_data_path()
    src_data_riscv64.mkdir(parents=True, exist_ok=True)
    (src_data_riscv64 / "riscv64.txt").write_text("riscv64 data")

    ext_params = {"variant": "base"}

    # Perform merge
    assert_merge_succeeds(dst, src, ext_params)

    # Verify both destinations received their respective data
    dst_benchmarks = dst.all_benchmarks()

    assert_data_merged(dst_benchmarks[0], ["aarch64.txt"])
    assert_data_not_merged(dst_benchmarks[0], ["riscv64.txt"])

    assert_data_merged(dst_benchmarks[1], ["riscv64.txt"])
    assert_data_not_merged(dst_benchmarks[1], ["aarch64.txt"])


# ==============================================================================
# Test Group 5: Data File Merging
# ==============================================================================


def test_merge_by_key_merges_data_files(tmp_path):
    """Verify data files are correctly merged from source to destination."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [{"arch": "aarch64", "variant": "base"}],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [{"arch": "aarch64"}],
    )

    # Add data files to source
    src_benchmark = src.all_benchmarks()[0]
    src_data_path = src_benchmark.get_benchmark_data_path()
    src_data_path.mkdir(parents=True, exist_ok=True)
    (src_data_path / "data.csv").write_text("a,b,c\n1,2,3\n")
    (src_data_path / "metrics.json").write_text('{"score": 100}')

    ext_params = {"variant": "base"}

    # Perform merge
    assert_merge_succeeds(dst, src, ext_params)

    # Verify data files were merged
    dst_benchmark = dst.all_benchmarks()[0]
    assert_data_merged(dst_benchmark, ["data.csv", "metrics.json"])


def test_merge_by_key_overwrites_existing_data(tmp_path):
    """Verify merging overwrites existing data files in destination."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [{"arch": "aarch64", "variant": "base"}],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [{"arch": "aarch64"}],
    )

    # Add data to both dst and src
    dst_benchmark = dst.all_benchmarks()[0]
    dst_data_path = dst_benchmark.get_benchmark_data_path()
    dst_data_path.mkdir(parents=True, exist_ok=True)
    (dst_data_path / "data.txt").write_text("old data")

    src_benchmark = src.all_benchmarks()[0]
    src_data_path = src_benchmark.get_benchmark_data_path()
    src_data_path.mkdir(parents=True, exist_ok=True)
    (src_data_path / "data.txt").write_text("new data")

    ext_params = {"variant": "base"}

    # Perform merge
    assert_merge_succeeds(dst, src, ext_params)

    # Verify data was overwritten
    merged_file = dst_data_path / "data.txt"
    assert merged_file.exists()
    assert merged_file.read_text() == "new data"


def test_merge_by_key_preserves_nested_directories(tmp_path):
    """Verify nested directory structures are preserved during merge."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [{"arch": "aarch64", "variant": "base"}],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [{"arch": "aarch64"}],
    )

    # Add nested data structure to source
    src_benchmark = src.all_benchmarks()[0]
    src_data_path = src_benchmark.get_benchmark_data_path()
    src_data_path.mkdir(parents=True, exist_ok=True)

    (src_data_path / "subdir1").mkdir()
    (src_data_path / "subdir1" / "file1.txt").write_text("data1")
    (src_data_path / "subdir2").mkdir()
    (src_data_path / "subdir2" / "file2.txt").write_text("data2")
    (src_data_path / "subdir2" / "nested").mkdir()
    (src_data_path / "subdir2" / "nested" / "file3.txt").write_text("data3")

    ext_params = {"variant": "base"}

    # Perform merge
    assert_merge_succeeds(dst, src, ext_params)

    # Verify nested structure preserved
    dst_benchmark = dst.all_benchmarks()[0]
    dst_data_path = dst_benchmark.get_benchmark_data_path()

    assert (dst_data_path / "subdir1" / "file1.txt").read_text() == "data1"
    assert (dst_data_path / "subdir2" / "file2.txt").read_text() == "data2"
    assert (dst_data_path / "subdir2" / "nested" / "file3.txt").read_text() == "data3"


def test_merge_by_key_skips_runner_scripts(tmp_path):
    """Verify runner scripts are NOT merged from source to destination."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [{"arch": "aarch64", "variant": "base"}],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [{"arch": "aarch64"}],
    )

    # Get benchmark instances
    dst_benchmark = dst.all_benchmarks()[0]
    src_benchmark = src.all_benchmarks()[0]

    # Add runner script to source data directory
    src_data_path = src_benchmark.get_benchmark_data_path()
    src_data_path.mkdir(parents=True, exist_ok=True)
    (src_data_path / "runner-script.sh").write_text("#!/bin/bash\necho source")

    # Add different runner script to destination data directory
    dst_data_path = dst_benchmark.get_benchmark_data_path()
    dst_data_path.mkdir(parents=True, exist_ok=True)
    (dst_data_path / "runner-script.sh").write_text("#!/bin/bash\necho destination")

    # Add some regular data to source (should be merged)
    (src_data_path / "data.txt").write_text("data")

    ext_params = {"variant": "base"}

    # Perform merge
    assert_merge_succeeds(dst, src, ext_params)

    # Verify runner script NOT overwritten in destination
    dst_runner = dst_data_path / "runner-script.sh"
    assert dst_runner.exists()
    assert dst_runner.read_text() == "#!/bin/bash\necho destination"

    # Verify data was merged
    assert_data_merged(dst_benchmark, ["data.txt"])


# ==============================================================================
# Test Group 6: Integration & Edge Cases
# ==============================================================================


def test_merge_by_key_empty_source_session(tmp_path):
    """Verify behavior when source session has no data to merge."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [{"arch": "aarch64", "variant": "base"}],
    )

    # Create source with matching parameterization but no data
    src = create_session_with_params(
        tmp_path,
        "src",
        [{"arch": "aarch64"}],  # Same params, no data added
    )

    ext_params = {"variant": "base"}
    assert_merge_succeeds(dst, src, ext_params)

    # Destination should have no data merged (source had none)
    dst_benchmark = dst.all_benchmarks()[0]
    data_path = dst_benchmark.get_benchmark_data_path()

    # Data directory might exist but should be empty or only have session files
    if data_path.exists():
        # Check that no actual data files were created (only directories if any)
        data_files = [f for f in data_path.rglob("*") if f.is_file()]
        # Runner script might exist, but no other data files
        non_runner_files = [f for f in data_files if f.name != "runner-script.sh"]
        assert len(non_runner_files) == 0


def test_merge_by_key_complex_multi_axis_scenario(tmp_path):
    """Integration test with complex multi-axis parameterization."""
    dst = create_session_with_params(
        tmp_path,
        "dst",
        [
            {"arch": "aarch64", "variant": "base"},
            {"arch": "aarch64", "variant": "purecap"},
            {"arch": "riscv64", "variant": "base"},
            {"arch": "riscv64", "variant": "purecap"},
        ],
    )

    src = create_session_with_params(
        tmp_path,
        "src",
        [
            {"arch": "aarch64"},
            {"arch": "riscv64"},
        ],
    )

    # Add data to each source context
    src_benchmarks = src.all_benchmarks()

    for i, bench in enumerate(src_benchmarks):
        data_path = bench.get_benchmark_data_path()
        data_path.mkdir(parents=True, exist_ok=True)
        arch = bench.config.parameters["arch"]
        (data_path / f"{arch}.txt").write_text(f"{arch} data")

    ext_params = {"variant": "base"}

    # Perform merge
    assert_merge_succeeds(dst, src, ext_params)

    # Verify data merged to correct destinations (base variants only)
    dst_benchmarks = dst.all_benchmarks()

    # Find the base variant contexts
    aarch64_base = next(
        b
        for b in dst_benchmarks
        if b.config.parameters["arch"] == "aarch64"
        and b.config.parameters["variant"] == "base"
    )
    riscv64_base = next(
        b
        for b in dst_benchmarks
        if b.config.parameters["arch"] == "riscv64"
        and b.config.parameters["variant"] == "base"
    )

    # Verify data merged
    assert_data_merged(aarch64_base, ["aarch64.txt"])
    assert_data_merged(riscv64_base, ["riscv64.txt"])

    # Verify purecap variants NOT modified
    aarch64_purecap = next(
        b
        for b in dst_benchmarks
        if b.config.parameters["arch"] == "aarch64"
        and b.config.parameters["variant"] == "purecap"
    )
    riscv64_purecap = next(
        b
        for b in dst_benchmarks
        if b.config.parameters["arch"] == "riscv64"
        and b.config.parameters["variant"] == "purecap"
    )

    assert_data_not_merged(aarch64_purecap, ["aarch64.txt"])
    assert_data_not_merged(riscv64_purecap, ["riscv64.txt"])
