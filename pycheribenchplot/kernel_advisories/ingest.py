import shutil
from dataclasses import MISSING, dataclass

import polars as pl

from ..core.artefact import PLDataFrameSessionLoadTask, Target
from ..core.config import Config, ConfigPath, config_field
from ..core.task import SessionDataGenTask, dependency, output


@dataclass
class IngestAdvisoriesConfig(Config):
    data_path: ConfigPath = config_field(MISSING, desc="Path to the input data file")


class IngestAdvisoriesTask(SessionDataGenTask):
    """
    Freeze the advisories file in the session for later processing.
    """
    public = True
    task_namespace = "kernel-advisories"
    task_name = "ingest"
    task_config_class = IngestAdvisoriesConfig

    @output
    def advisories(self):
        return Target(self, "advisories", loader=PLDataFrameSessionLoadTask, ext="json")

    def run(self):
        shutil.copyfile(self.config.data_path, self.advisories.single_path())
