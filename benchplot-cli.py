#!/usr/bin/env python

import argparse as ap
import logging
import re
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

        sub_gen = session_subparsers.add_parser("generate", help="Generate run scripts")
        self._register_session_arg(sub_gen)

        sub_run = session_subparsers.add_parser("run", help="Run the benchmark session")
        self._register_session_arg(sub_run)

        sub_clean = session_subparsers.add_parser("clean", help="Clean session data and plots.")
        self._register_session_arg(sub_clean)

        sub_analyse = session_subparsers.add_parser("analyse", help="Process data and generate plots.")
        self._register_session_arg(sub_analyse)
        sub_analyse.add_argument("-a",
                                 "--analysis-config",
                                 type=Path,
                                 default=None,
                                 help="Analysis configuration file.")
        sub_analyse.add_argument("--set-default",
                                 action="store_true",
                                 help="Set the current analysis configuration as the default analysis configuration")
        sub_analyse.add_argument("-t",
                                 "--task",
                                 type=str,
                                 action="append",
                                 help="Task names to run for the analysis. This is a convenience shorthand "
                                 "for the full analysis configuration")
        sub_analyse.add_argument("--clean", action="store_true", help="Wipe analysis outputs before running.")

        sub_bundle = session_subparsers.add_parser("bundle",
                                                   help="Create a session archive with all the generated content.")
        sub_bundle.add_argument("-o",
                                "--output",
                                default=None,
                                type=Path,
                                help="Output directory, defaults to the directory containing the session")
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

    def handle_generate(self, user_config, args):
        """
        Hook to handle script generation subcommand.
        """
        session = self._get_session(user_config, args)
        session.generate()

    def handle_run(self, user_config, args):
        """
        Hook to handle the run subcommand.
        """
        session = self._get_session(user_config, args)
        session.run()

    def handle_clean(self, user_config, args):
        # XXX add safety net question?
        session = self._get_session(user_config, args)
        session.clean_all()

    def handle_analyse(self, user_config, args):
        session = self._get_session(user_config, args)

        if args.clean:
            session.clean_analysis()

        # Use the analysis configuration from the session by default
        analysis_config = None
        if args.analysis_config:
            analysis_config = self._parse_config(args.analysis_config, AnalysisConfig)
        elif args.task:
            analysis_config = AnalysisConfig()
            for task in args.task:
                analysis_config.tasks.append(TaskTargetConfig(handler=task))
        session.analyse(analysis_config, set_default=args.set_default)

    def handle_bundle(self, user_config, args):
        session = self._get_session(user_config, args)
        session.bundle(path=args.output)

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

        info_subparsers = parser.add_subparsers(help="Action", dest="info_action")

        sub_session = info_subparsers.add_parser("session", help="Display information about an existing session")
        self._register_session_arg(sub_session)
        sub_session.add_argument("-a",
                                 "--show-analysis-tasks",
                                 action="store_true",
                                 help="Show compatible analysis tasks")

        sub_task = info_subparsers.add_parser("task", help="Display information about a task")
        sub_task.add_argument("task_spec", nargs="+", help="Name(s) of task(s) to describe")

        sub_config = info_subparsers.add_parser("config", help="Display configuration information")
        sub_config.add_argument("-u",
                                "--user",
                                type=Path,
                                required=False,
                                default=None,
                                help="Generate a default user configuration file")

    def handle_session(self, user_config, args):
        """
        Display information about the tasks and the data within a session.
        """
        session = self._get_session(user_config, args)
        configurations = session.config.configurations
        print(f"Session {session.config.name} ({session.config.uuid}):")
        for c in session.config.configurations:
            print(f"\t{c.name} ({c.uuid}) on {c.instance}")
            print("\t\tParameterization:")
            for pk, pv in c.parameters.items():
                print(f"\t\t - {pk} = {pv}")
            print("\t\tGenerators:")
            for g in c.generators:
                print(f"\t\t - {g.handler}")

        if args.show_analysis_tasks:
            print("\tAvailable analysis tasks:\n")
            for task in session.get_public_tasks():
                spec_line = f"{task.task_namespace}.{task.task_name} ({task.__name__})"
                print("\t", spec_line)

    def handle_task(self, user_config, args):
        """
        Display task information.
        """
        for task_class in TaskRegistry.iter_public():
            match = False
            for matcher in args.task_spec:
                match = re.match(matcher, f"{task_class.task_namespace}.{task_class.task_name}")
                if match:
                    break
            if not match:
                continue
            # Write out the task description
            print(task_class.describe())

    def handle_config(self, user_config, args):
        """
        Help information about configurations.

        Without any arguments, this will dump the existing user config,
        or a default one if none is found.
        """
        if hasattr(args, "u"):
            if args.u.exists():
                self.logger.warning("User configuration file already exists: %s", args.u)
            with open(args.u, "w+") as outconfig:
                outconfig.write(BenchplotUserConfig().emit_json())
        else:
            print(user_config.emit_json())

    def handle(self, user_config, args):
        if args.info_action is None:
            raise ToolArgparseError("Missing info action")
        handler_name = f"handle_{args.info_action}"
        handler = getattr(self, handler_name)
        handler(user_config, args)


def main():
    cli = CommandLineTool("benchplot-cli")
    cli.add_subcommand(SessionSubCommand())
    cli.add_subcommand(TaskInfoSubCommand())
    cli.setup()


if __name__ == "__main__":
    main()
