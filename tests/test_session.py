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


@pytest.fixture
def user_config(tmp_path):
    config = BenchplotUserConfig()
    config.session_path = tmp_path
    return config


def create_pipeline_config(name="test_bench", parameterize=None):
    if parameterize is None:
        parameterize = {"param1": [1, 2]}

    return PipelineConfig(
        benchmark_config=PipelineBenchmarkConfig(
            name=name,
            parameterize=parameterize,
            system=[SystemConfig(matches={}, host_system=InstanceConfig.native())],
        )
    )


def test_session_extend_success(user_config, tmp_path):
    """
    Test that extending a session with a strict superset successfully appends
    new configurations while keeping existing ones (and their UUIDs) intact.
    """
    session_path = tmp_path / "test_session"

    # Create initial session
    config_1 = create_pipeline_config(parameterize={"param1": [1, 2]})
    session = Session.make_new(user_config, config_1, session_path)

    # Store the original UUIDs
    original_uuids = {
        bench.parameters["param1"]: str(bench.uuid)
        for bench in session.config.configurations
    }

    assert len(original_uuids) == 2
    assert set(original_uuids.keys()) == {1, 2}

    # Extend the session
    config_2 = create_pipeline_config(parameterize={"param1": [1, 2, 3]})
    session.extend(config_2, workdir=tmp_path)

    # Verify the updated session object
    assert len(session.config.configurations) == 3
    new_uuids = {
        bench.parameters["param1"]: str(bench.uuid)
        for bench in session.config.configurations
    }

    # Original UUIDs must remain unchanged
    assert new_uuids[1] == original_uuids[1]
    assert new_uuids[2] == original_uuids[2]
    # New UUID must be created for param1=3
    assert new_uuids[3] is not None

    # Verify the saved runfile contains the new configurations
    runfile_path = session_path / SESSION_RUN_FILE
    with open(runfile_path, "r") as f:
        saved_data = json.load(f)

    assert len(saved_data["configurations"]) == 3
    saved_param1_values = {
        conf["parameters"]["param1"] for conf in saved_data["configurations"]
    }
    assert saved_param1_values == {1, 2, 3}


def test_session_extend_fail_missing_axis_value(user_config, tmp_path):
    """
    Test that extending a session with a configuration that misses an existing
    parameter value fails cleanly.
    """
    session_path = tmp_path / "test_session"

    # Create initial session
    config_1 = create_pipeline_config(parameterize={"param1": [1, 2]})
    session = Session.make_new(user_config, config_1, session_path)

    # Attempt to extend with missing value '1'
    config_2 = create_pipeline_config(parameterize={"param1": [2, 3]})

    with pytest.raises(ValueError, match="missing values"):
        session.extend(config_2, workdir=tmp_path)

    # Verify session was not modified
    assert len(session.config.configurations) == 2


def test_session_extend_fail_missing_axis(user_config, tmp_path):
    """
    Test that extending a session with a configuration that entirely misses
    an existing axis fails cleanly.
    """
    session_path = tmp_path / "test_session"

    # Create initial session
    config_1 = create_pipeline_config(parameterize={"param1": [1, 2], "param2": ["a"]})
    session = Session.make_new(user_config, config_1, session_path)

    # Attempt to extend with missing axis 'param2'
    config_2 = create_pipeline_config(parameterize={"param1": [1, 2, 3]})

    with pytest.raises(ValueError, match="missing 'param2'"):
        session.extend(config_2, workdir=tmp_path)


def test_session_extend_fail_orphaned_combination(user_config, tmp_path):
    """
    Test that extending a session with a configuration that adds a new axis
    but fails to provide the existing combinations fails cleanly.
    """
    session_path = tmp_path / "test_session"

    # Create initial session
    config_1 = create_pipeline_config(parameterize={"param1": [1, 2]})
    session = Session.make_new(user_config, config_1, session_path)

    # Attempt to extend by adding a new axis. Because the cartesian product
    # of the new configuration produces (1, "a") and (2, "a"), the original
    # parameter configurations (param1=1) and (param1=2) are effectively omitted.
    config_2 = create_pipeline_config(
        parameterize={"param1": [1, 2, 3], "param2": ["a"]}
    )

    with pytest.raises(ValueError, match="not a superset"):
        session.extend(config_2, workdir=tmp_path)
