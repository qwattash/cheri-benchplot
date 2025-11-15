import shutil
from dataclasses import dataclass

from ..core.artefact import PLDataFrameSessionLoadTask, Target
from ..core.config import Config, ConfigPath, config_field
from ..core.task import ExecutionTask, output


@dataclass
class IngestAdvisoriesConfig(Config):
    data_path: ConfigPath = config_field(
        Config.REQUIRED, desc="Path to the input data file"
    )


class IngestAdvisoriesTask(ExecutionTask):
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
        raise NotImplementedError("TODO: Needs to be ported to new system")
        shutil.copyfile(self.config.data_path, self.advisories.single_path())
