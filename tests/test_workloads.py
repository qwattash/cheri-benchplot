from pathlib import Path
import shutil

import polars as pl
import pytest

from pycheribenchplot.core.config import BenchplotUserConfig, PipelineConfig
from pycheribenchplot.core.session import Session


ASSETS_DIR = Path(__file__).parent.parent / "workloads"

# Note: rely on the convention that analysis configurations will be named foo.analysis.json
config_files = [
    c
    for c in ASSETS_DIR.rglob("*.json")
    if not c.name.endswith(".analysis.json") and not c.name.endswith(".template.json")
]


def generate_id(path):
    return path.name


@pytest.fixture
def session_path(tmp_path):
    shutil.rmtree(tmp_path)
    return tmp_path


@pytest.mark.parametrize("config_file", config_files, ids=generate_id)
def test_configuration(config_file, session_path):
    """
    Attempt to parse every workload configuration an initialize the
    corresponding session.
    This ensures that all workloads configurations are properly maintained.
    """
    user_config = BenchplotUserConfig()
    workload = PipelineConfig.load_json(config_file)
    session = Session.make_new(
        user_config, workload, session_path, workdir=config_file.parent
    )
    session.generate()

    # TODO Generate mock data for analysis tests

    session.clean_all()
