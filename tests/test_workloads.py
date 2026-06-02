from dataclasses import dataclass
from pathlib import Path
import shutil

import polars as pl
import pytest

from pycheribenchplot.core.config import (
    AnalysisConfig,
    BenchplotUserConfig,
    PipelineConfig,
)
from pycheribenchplot.core.session import Session


WORKLOADS_ROOT = Path(__file__).parent.parent / "workloads"


@dataclass
class WorkloadInfo:
    session_name: str
    workload_config: Path
    data_path: Path | None = None
    analysis_config: Path | None = None

    @property
    def has_smoketest(self):
        return self.data_path is not None and self.analysis_config is not None


# Note: expect the following directory layout for workloads
#
# workloads
#   - <group> (e.g. iperf)
#     - data (sessions and collected data)
#       - <session-name> (e.g. iperf.smoketest)
#       - ...
#     - analysis (analysis configurations)
#       - <session-name>.json
#       - ...
#     - <session-name>.json (workload configuration e.g. iperf.smoketest.json)
#
workloads = []

for group in WORKLOADS_ROOT.iterdir():
    if not group.is_dir():
        continue
    for element in group.iterdir():
        if element.name.endswith(".json"):
            name = element.stem
            wki = WorkloadInfo(name, element.absolute())
            analysis_conf = group / "analysis" / element.name
            if analysis_conf.exists():
                wki.analysis_config = analysis_conf.absolute()
            data = group / "data" / name
            if data.exists() and data.is_dir():
                wki.data_path = data.absolute()
            workloads.append(wki)


def generate_id(wkinfo):
    return wkinfo.session_name


@pytest.fixture
def session_path(tmp_path):
    shutil.rmtree(tmp_path)
    return tmp_path


@pytest.mark.parametrize("wkinfo", workloads, ids=generate_id)
def test_configuration(wkinfo, session_path):
    """
    Attempt to parse every workload configuration an initialize the
    corresponding session.
    This ensures that all workloads configurations are properly maintained.
    """
    user_config = BenchplotUserConfig()
    workload = PipelineConfig.load_json(wkinfo.workload_config)
    session = Session.make_new(
        user_config, workload, session_path, workdir=wkinfo.workload_config.parent
    )
    session.generate()
    session.clean_all()


@pytest.mark.parametrize(
    "wkinfo", filter(lambda w: w.has_smoketest, workloads), ids=generate_id
)
def test_analysis(wkinfo, tmp_path):
    """
    Run analysis passes for smoketest data.
    Smoketest data must be maintained in sync with the configurations.
    """
    scratch = tmp_path / "data"
    shutil.copytree(wkinfo.data_path, scratch)
    user_config = BenchplotUserConfig()
    session = Session.from_path(user_config, scratch)
    analysis_config = AnalysisConfig.load_json(wkinfo.analysis_config)

    # Run the analysis pass
    session.analyse(analysis_config)
