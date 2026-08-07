from dataclasses import dataclass, replace
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
    analysis_config: list[Path] | None = None

    @property
    def can_test_analysis(self):
        return self.data_path is not None and self.analysis_config is not None


# Note: expect the following directory layout for workloads
#
# workloads
#   - fragments (imported workload fragments)
#   - <group> (e.g. iperf)
#     - data (sessions and collected data)
#       - <session-name> (e.g. iperf.smoketest)
#       - ...
#     - analysis (analysis configurations)
#       - <session-name>.json
#       - <session-name> (alternative to a single <session-name>.json)
#         - pass_1.json
#         - pass_2.json
#         - ...
#       - ...
#     - <session-name>.json (workload configuration e.g. iperf.smoketest.json)
#
workloads = []

for group in WORKLOADS_ROOT.iterdir():
    if not group.is_dir():
        continue
    if group.name == "fragments":
        continue
    for element in group.iterdir():
        if element.name.endswith(".json"):
            name = element.stem
            wki = WorkloadInfo(name, element.absolute())
            analysis_conf = group / "analysis" / element.name
            analysis_conf_dir = group / "analysis" / element.stem
            if analysis_conf.exists():
                wki.analysis_config = [analysis_conf.absolute()]
            elif analysis_conf_dir.is_dir():
                pass_configs = analysis_conf_dir.glob("*.json")
                wki.analysis_config = [p.absolute() for p in pass_configs]
            data = group / "data" / name
            if data.exists() and data.is_dir():
                wki.data_path = data.absolute()
            workloads.append(wki)


def analysis_workloads() -> list[WorkloadInfo]:
    unrolled = []
    for wki in workloads:
        if not wki.can_test_analysis:
            continue
        if len(wki.analysis_config) == 1:
            unrolled.append(wki)
        else:
            for pass_config in wki.analysis_config:
                inst = replace(wki, analysis_config=[pass_config])
                unrolled.append(inst)
    return unrolled


def generate_id(wkinfo):
    return wkinfo.session_name


def generate_analysis_id(wkinfo):
    pass_name = wkinfo.analysis_config[0].stem
    return f"{wkinfo.session_name}:{pass_name}"


@pytest.fixture
def session_path(tmp_path):
    shutil.rmtree(tmp_path)
    return tmp_path


@pytest.mark.timeout(30)
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
    assert session is not None
    session.generate()
    # Verify no errors
    assert not session.scheduler.failed_tasks


@pytest.mark.timeout(300)
@pytest.mark.parametrize("wkinfo", analysis_workloads(), ids=generate_analysis_id)
def test_analysis(wkinfo, tmp_path):
    """
    Run analysis passes for smoketest data.
    Smoketest data must be maintained in sync with the configurations.
    """
    scratch = tmp_path / "data"
    shutil.copytree(wkinfo.data_path, scratch)
    user_config = BenchplotUserConfig()
    session = Session.from_path(user_config, scratch)
    assert session is not None
    analysis_config = AnalysisConfig.load_json(wkinfo.analysis_config[0])

    # Run the analysis pass
    session.analyse(analysis_config)

    # Verify no errors
    assert not session.scheduler.failed_tasks
