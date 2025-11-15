#
# Utility for standard session and benchmark fixtures
#
import typing
import uuid
from pathlib import Path

import pytest

from pycheribenchplot.core.analysis import ParamSliceInfo, SliceAnalysisTask
from pycheribenchplot.core.artefact import Target
from pycheribenchplot.core.config import (
    AnalysisConfig,
    BenchplotUserConfig,
    Config,
    SessionRunConfig,
)
from pycheribenchplot.core.session import Session
from pycheribenchplot.core.shellgen import ScriptContext
from pycheribenchplot.core.task import DatasetTask, ExecutionTask, SessionTask, Task


class TaskFactory:
    """
    Helper that provides an interface to setup test task configurations.
    """

    def __init__(self, session_root: Path, session_id: uuid.UUID = None):
        if not session_id:
            session_id = uuid.uuid4()
        dataset_id = uuid.uuid3(session_id, "dataset")

        self._session_root = session_root
        self._unrolled_config = {
            "uuid": str(session_id),
            "name": "test-session",
            "configurations": [
                {
                    "name": "test-configuration",
                    "iterations": 1,
                    "parameters": {
                        "target": "default",
                    },
                    "generators": [],
                    "uuid": str(dataset_id),
                    "instance": {
                        "kernel": "TEST-KERNEL-CONFIG",
                        "name": "test-target-instance",
                        "cheri_target": "morello-purecap",
                        "kernelabi": "purecap",
                        "userabi": "purecap",
                    },
                }
            ],
        }
        self._session = None

    def get_session(self) -> Session:
        assert self._session is not None
        return self._session

    def add_gen_task(self, task_type: typing.Type[Task], task_config: Config = None):
        handler = task_type.task_namespace + "." + task_type.task_name
        serialized_task_config = {}
        if task_config:
            # re-serialize task configuration to ensure that the session agrees
            serialized_task_config = task_config.schema().dump(task_config)

        generators = self._unrolled_config["configurations"][0]["generators"]
        generators.append({"handler": handler, "task_options": serialized_task_config})

    def quick_with_session(
        self, task_type: typing.Type[Task], task_config: Config = None
    ) -> Session:
        self.add_gen_task(task_type, task_config)
        self.build()
        return self.build_task(task_type, task_config)

    def build(self) -> Session:
        assert self._session is None, "Session already built"
        sconfig = SessionRunConfig.schema().load(self._unrolled_config)
        user_config = BenchplotUserConfig()
        session = Session(user_config, sconfig, session_path=self._session_root)
        self._session = session
        return session

    def build_task(
        self,
        task_type: typing.Type[Task],
        config: Config = None,
        analysis_config: AnalysisConfig = None,
    ) -> Task:
        """
        Helper to build a generic task for this session.
        If this is a generator task, it should have been added to the session configuration
        before creating the session, for consistency.
        """
        if task_type.task_config_class:
            assert isinstance(config, task_type.task_config_class), (
                "Unexpected configuration in test"
            )
        assert self._session is not None, "Can not build a task without a session"
        if analysis_config is None:
            analysis_config = AnalysisConfig()

        if issubclass(task_type, SliceAnalysisTask):
            return task_type(
                self._session,
                ParamSliceInfo(fixed_params={}, free_axes=["target"], rank=1),
                analysis_config,
                config,
            )
        elif issubclass(task_type, SessionTask):
            # No such thing as a session execution task
            assert not issubclass(task_type, ExecutionTask), (
                "Session-wide exec tasks are not supported"
            )
            return task_type(self._session, analysis_config, task_config=config)
        elif issubclass(task_type, DatasetTask):
            context = self._session.all_benchmarks()[0]
            if issubclass(task_type, ExecutionTask):
                return task_type(
                    context, script=ScriptContext(context), task_config=config
                )
            else:
                return task_type(context, analysis_config, task_config=config)
        else:
            # Plain task
            return task_type(task_config=config)


@pytest.fixture
def task_factory(tmp_path) -> TaskFactory:
    f = TaskFactory(tmp_path)
    return f


def helper_run_task(task: Task):
    """
    Run a task and all its dependencies recursively
    """
    for dep in task.dependencies():
        helper_run_task(dep)
    task.run()


def helper_hook_output_location(task: Task, **kwargs):
    """
    Helper to hook a task @output decorated accessor and replace the
    target file with a test-controlled value.
    """
    for attr_name, mock_value in kwargs.items():
        output = getattr(task, attr_name)
        assert isinstance(output, Target)
        output.iter_paths = lambda: [Path(mock_value)]
