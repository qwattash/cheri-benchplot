import asyncio as aio
import shutil
import subprocess
import typing
import uuid
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .analysis import BenchmarkAnalysis, BenchmarkAnalysisRegistry
from .config import AnalysisConfig, BenchplotUserConfig, PipelineConfig
from .session import PipelineSession
from .util import new_logger


class PipelineManager:
    """
    Main API for the benchplot framework.
    The pipeline manager handles all actions for a specific user setup.

    :param user_config: User configuration parameters
    """
    def __init__(self, user_config: BenchplotUserConfig):
        self.user_config = user_config
        self.logger = new_logger("manager")

    def make_session(self, name: Path, pipeline_config: PipelineConfig) -> uuid.UUID:
        """
        Create a new session. The pipeline configuration specifies the path.

        :param name: The new session name
        :param pipeline_config: Pipeline session configuration parameters.
        This is used to generate the session instructions file.
        :return: The UUID of the newly created session
        """
        session = PipelineSession.make_new(self, name, pipeline_config)
        return session.uuid

    def delete_session(self, session: PipelineSession):
        """
        Delete an existing session

        :param session: A session instance
        """
        self.logger.info("Remove session %s (%s)", session.name, session.uuid)
        shutil.rmtree(session.session_root_path)

    def resolve_session(self, path: Path) -> typing.Optional[PipelineSession]:
        """
        Find a matching session and load it. One of name or uuid must be given.

        :param name: Name or path of the session
        """
        self.logger.debug("Scanning %s for valid session", path)
        found = None
        if path.exists() and PipelineSession.is_session(path):
            found = PipelineSession.from_path(self, path)

        if found:
            self.logger.debug("Resolved session %s => %s", path, found)
        else:
            self.logger.debug("Session for %s not found", path)
        return found

    def run_session(self, session: PipelineSession, shellgen_only: bool = False):
        """
        Run the given session.
        This triggers the data generation phase for all dataset handlers
        that have been configured.
        Note that the session runs asyncronously using asyncio.

        :param session: A valid session object
        :param shellgen_only: If True, only generate run scripts and do not run anything
        """
        assert session is not None
        session.clean()

        run_mode = "full"
        if shellgen_only:
            run_mode = "shellgen"
        self.logger.debug("Session %s (%s) start run", session.name, session.uuid)
        session.run(mode=run_mode)
        self.logger.info("Session %s (%s) run finished", session.name, session.uuid)

    def run_analysis(self, session: PipelineSession, config: AnalysisConfig, mode: str = None):
        """
        Run the analysis step for the given session.
        Note that the session must have been run before the
        analysis.

        :param session: A valid session object
        """
        assert session is not None

        loop = aio.get_event_loop()
        self.logger.debug("Session %s (%s) start analysis", session.name, session.uuid)
        with session.analyse(config, mode) as ctx:
            loop.run_until_complete(ctx.main())
        loop.close()
        self.logger.info("Session %s (%s) analysis finished", session.name, session.uuid)

    def get_analysis_handlers(self, session: PipelineSession | None) -> list[typing.Type[BenchmarkAnalysis]]:
        """
        Return the analysis handlers suitable for a given session.
        If not session is given, return all analysis handlers.

        :param session: Optional session to filter compatible handlers
        :return: A list of :class:`BenchmarkAnalysis` classes.
        """
        handlers = [h for h in BenchmarkAnalysisRegistry.analysis_steps.values() if h.name]
        if session:
            available = set()
            for c in session.config.configurations:
                available.add(c.benchmark.handler)
                for d in c.aux_dataset_handlers:
                    available.add(d.handler)
            handlers = [h for h in handlers if h in available]
        return handlers

    def bundle(self, session: PipelineSession):
        """
        Produce a compressed archive of a session
        """
        bundle_file = session.session_root_path.with_suffix(".tar.xz")
        self.logger.info("Generate %s bundle", session.session_root_path)
        if bundle_file.exists():
            self.logger.info("Replacing old bundle %s", bundle_file)
            bundle_file.unlink()
        result = subprocess.run(["tar", "-J", "-c", "-f", bundle_file, session.session_root_path])
        if result.returncode:
            self.logger.error("Failed to produce bundle")
        self.logger.info("Archive created at %s", bundle_file)
