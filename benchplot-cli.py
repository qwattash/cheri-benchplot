#!/usr/bin/env python

import argparse as ap
import logging
import traceback
import uuid
from json.decoder import JSONDecodeError
from pathlib import Path

from marshmallow.exceptions import ValidationError

from pycheribenchplot.core.config import (AnalysisConfig, BenchplotUserConfig, PipelineConfig, TaskTargetConfig)
from pycheribenchplot.core.error import ToolArgparseError
from pycheribenchplot.core.session import Session
from pycheribenchplot.core.task import TaskRegistry
from pycheribenchplot.core.tool import CommandLineTool, SubCommand


class SessionSubCommand(SubCommand):
    """
    Manage and ispect sessions.
    """
    name = "session"

    def _register_session_arg(self, parser):
        """
        Helper to add the session ID argument to the given parser.
        """
        parser.add_argument("target",
                            type=Path,
                            default=Path.cwd(),
                            nargs="?",
                            help="Path to the target session, defaults to the current working directory.")

    def register_options(self, parser):
        super().register_options(parser)

        session_subparsers = parser.add_subparsers(help="Action", dest="session_action")

        sub_create = session_subparsers.add_parser("create", help="Create a new session from the given configuration.")
        sub_create.add_argument("pipeline_config", type=Path, help="Path to the pipeline configuration file.")
        self._register_session_arg(sub_create)
        sub_create.add_argument("-f",
                                "--force",
                                action="store_true",
                                help="Force re-create the session if it already exists. Use with caution.")

        sub_run = session_subparsers.add_parser("run",
                                                help="Run the session datagen tasks, this must be done "
                                                "before data can be analysed.")
        self._register_session_arg(sub_run)
        sub_run.add_argument("--shellgen-only",
                             action="store_true",
                             help="Only perform shell script generation and stop before running anything.")

        sub_clean = session_subparsers.add_parser("clean", help="Clean session data and plots.")
        self._register_session_arg(sub_clean)

        sub_analyse = session_subparsers.add_parser("analyse", help="Process data and generate plots.")
        self._register_session_arg(sub_analyse)
        sub_analyse.add_argument("-a",
                                 "--analysis-config",
                                 type=Path,
                                 default=None,
                                 help="Analysis configuration file.")
        sub_analyse.add_argument("-t",
                                 "--task",
                                 type=str,
                                 action="append",
                                 help="Task names to run for the analysis. This is a convenience shorthand "
                                 "for the full analysis configuration")
        sub_analyse.add_argument("--clean", action="store_true", help="Wipe analysis outputs before running.")

        sub_bundle = session_subparsers.add_parser("bundle",
                                                   help="Create a session archive with all the generated content.")
        self._register_session_arg(sub_bundle)

    def handle_create(self, user_config, args):
        """
        Hook to handle the create subcommand.
        """
        config = self._parse_config(args.pipeline_config, PipelineConfig)

        session = self._get_session(user_config, args, missing_ok=True)
        if not session:
            session = Session.make_new(user_config, config, args.target)
        elif args.force:
            session.delete()
            session = Session.make_new(user_config, config, args.target)
        else:
            self.logger.error("Session %s already exists", args.target)
            raise FileExistsError(f"Session {args.target} already exists")

    def handle_run(self, user_config, args):
        """
        Hook to handle the run subcommand.
        """
        session = self._get_session(user_config, args)
        session.run("shellgen" if args.shellgen_only else "full")

    def handle_clean(self, user_config, args):
        # XXX add safety net question?
        session = self._get_session(user_config, args)
        session.clean_all()

    def handle_analyse(self, user_config, args):
        session = self._get_session(user_config, args)

        if args.clean:
            session.clean_analysis()

        if args.analysis_config:
            analysis_config = self._parse_config(args.analysis_config, AnalysisConfig)
        else:
            analysis_config = AnalysisConfig()
            for task in args.task:
                analysis_config.tasks.append(TaskTargetConfig(handler=task))
        if not analysis_config.tasks:
            self.logger.error("Invalid analysis configuration, one of the -a or "
                              "-t options must be used")
            raise RuntimeError("Malformed configuration")
        session.analyse(analysis_config)

    def handle_bundle(self, user_config, args):
        session = self._get_session(user_config, args)
        session.bundle()

    def handle(self, user_config, args):
        if args.session_action is None:
            raise ToolArgparseError("Missing session action")
        handler_name = f"handle_{args.session_action}"
        handler = getattr(self, handler_name)
        handler(user_config, args)


class TaskInfoSubCommand(SubCommand):
    """
    Display information about tasks.
    """
    name = "info"

    def register_options(self, parser):
        super().register_options(parser)

        parser.add_argument("what",
                            choices=["session", "task", "tasks", "generators", "config"],
                            help="Display information about something.")
        parser.add_argument("-t",
                            "--task",
                            type=str,
                            action="append",
                            required=False,
                            help="Name of the task for which information is requested, when relevant")
        parser.add_argument("-u",
                            type=Path,
                            default=False,
                            required=False,
                            help="Generate a default user configuration file")
        parser.add_argument("target", type=Path, default=Path.cwd(), nargs="?", help="Path of the target session.")

    def handle_session(self, user_config, args):
        """
        Display information about the tasks and the data within a session.
        """
        session = self._get_session(user_config, args)
        configurations = session.config.configurations
        print(f"Session {session.config.name} ({session.config.uuid}):")
        for c in session.config.configurations:
            print("\t", c)

    def handle_tasks(self, user_config, args):
        """
        Display information about compatible analysis tasks for a session.
        """
        session = self._get_session(user_config, args)
        print("Analysis targets:")
        for task in session.get_public_tasks():
            spec_line = f"{task.task_namespace}.{task.task_name} ({task.__name__})"
            print("\t", spec_line)

    def handle_generators(self, user_config, args):
        """
        List all available data generation tasks.

        These are the tasks that may be used to produce data during a session run.
        """
        print("Generator tasks:")
        for task_class in TaskRegistry.iter_public():
            if not task_class.is_exec_task():
                continue
            spec_line = f"{task_class.task_namespace}.{task_class.task_name} ({task_class.__name__})"
            print("\t", spec_line)

    def handle_spec(self, user_config, args):
        """
        Display detailed information about a task and how it can be used.
        """
        if not args.task:
            self.logger.warning("No -t,--task specified, you should indicate at least "
                                "one task to show information about")
        for task_name in args.task:
            task_types = TaskRegistry.resolve_task(task_name)
            if not task_types:
                self.logger.error(
                    "Invalid task name %s. The task name may be "
                    "'<namespace>.<task_name>' or '<namespace>.*'.", task_name)
            print("TODO not implemented")

    def handle_config_info(self, user_config, args):
        """
        Help information about configurations.

        Without any arguments, this will dump the existing user config,
        or a default one if none is found.
        """
        if args.u:
            if args.u.exists():
                self.logger.warning("User configuration file already exists: %s", args.u)
            with open(args.u, "w+") as outconfig:
                outconfig.write(BenchplotUserConfig().emit_json())
        else:
            print(user_config.emit_json())

    def handle(self, user_config, args):
        if args.what == "session":
            self.handle_session(user_config, args)
        elif args.what == "tasks":
            self.handle_tasks(user_config, args)
        elif args.what == "generators":
            self.handle_generators(user_config, args)
        elif args.what == "task":
            self.handle_spec(user_config, args)
        elif args.what == "config":
            self.handle_config_info(user_config, args)
        else:
            raise ValueError(f"Invalid key {args.what}")


def main():
    cli = CommandLineTool("benchplot-cli")
    cli.add_subcommand(SessionSubCommand())
    cli.add_subcommand(TaskInfoSubCommand())
    cli.setup()


if __name__ == "__main__":
    main()
