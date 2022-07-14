import asyncio as aio
import shutil
import typing
import uuid
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

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

    def make_session(self, name: str, pipeline_config: PipelineConfig) -> uuid.UUID:
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
        shutil.rmtree(self.user_config.session_path / session.name)

    def resolve_session(self, name=None, uuid=None) -> typing.Optional[PipelineSession]:
        """
        Find a matching session and load it. One of name or uuid must be given.

        :param name: Name of the session
        :param uuid: Session UUID
        """
        found = None
        if name:
            path = self.user_config.session_path / name
            self.logger.debug("Scanning %s for valid session", path)
            if path.exists() and PipelineSession.is_session(path):
                found = PipelineSession.from_path(self, path)
        elif uuid:
            for dirname in self.user_config.session_path.iterdir():
                path = self.user_config.session_path / dirname
                self.logger.debug("Scanning %s for valid session", path)
                if not PipelineSession.is_session(path):
                    continue
                session = PipelineSession.from_path(self, path)
                if session.uuid == uuid:
                    found = session
                    break
        if found:
            self.logger.debug("Resolved session %s => %s", path, found)
        else:
            self.logger.debug("Session for %s not found", path)
        return found

    def run_session(self, session: PipelineSession):
        """
        Run the given session.
        This triggers the data generation phase for all dataset handlers
        that have been configured.
        Note that the session runs asyncronously using asyncio.

        :param session: A valid session object
        """
        assert session is not None
        session.clean()

        loop = aio.get_event_loop()
        self.logger.debug("Session %s (%s) start run", session.name, session.uuid)
        with session.run() as ctx:
            loop.run_until_complete(ctx.main())
        loop.close()
        self.logger.info("Session %s (%s) run finished", session.name, session.uuid)

    def run_analysis(self, session: PipelineSession, config: AnalysisConfig, interactive: typing.Optional[str] = None):
        """
        Run the analysis step for the given session.
        Note that the session must have been run before the
        analysis.

        :param session: A valid session object
        """
        assert session is not None

        loop = aio.get_event_loop()
        self.logger.debug("Session %s (%s) start analysis", session.name, session.uuid)
        with session.analyse(config, interactive) as ctx:
            loop.run_until_complete(ctx.main())
        loop.close()
        self.logger.info("Session %s (%s) analysis finished", session.name, session.uuid)
