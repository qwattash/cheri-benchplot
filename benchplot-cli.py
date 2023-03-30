#!/usr/bin/env python

import argparse as ap
import logging
import traceback
import uuid
from json.decoder import JSONDecodeError
from pathlib import Path

from marshmallow.exceptions import ValidationError

from pycheribenchplot.core.config import (AnalysisConfig, BenchplotUserConfig, PipelineConfig, TaskTargetConfig)
from pycheribenchplot.core.session import Session
from pycheribenchplot.core.util import setup_logging

# Global logger from the logging setup
logger = None


def add_session_spec_options(parser):
    """
    Helper to add options to identify a session
    """
    session_name = parser.add_argument("session_path", type=Path, help="Path or name of the target session")


def resolve_session(user_config, session_path):
    session = Session.from_path(user_config, session_path)
    if not session:
        logger.error("Session %s does not exist", session_path)
        exit(1)
    return session


def list_session(session):
    configurations = session.config.configurations
    print(f"Session {session.config.name} ({session.config.uuid}):")
    for c in session.config.configurations:
        print("\t", c)


def list_tasks(session):
    """
    Helper to display public tasks for a given session
    """
    print("Public analysis targets:")
    for task in session.get_public_tasks():
        spec_line = f"{task.task_namespace}.{task.task_name} ({task.__name__})"
        print("\t", spec_line)


def handle_command(user_config: BenchplotUserConfig, args):
    if args.command == "session":
        try:
            config = PipelineConfig.load_json(args.pipeline_config)
            session = Session.from_path(user_config, args.session_path)
        except JSONDecodeError as ex:
            logger.error("Malformed pipeline configuration %s: %s", args.pipeline_config, ex)
            raise
        except ValidationError as ex:
            logger.error("Invalid pipeline configuration %s: %s", args.pipeline_config, ex)
            raise
        if not session:
            session = Session.make_new(user_config, config, args.session_path)
        elif args.force:
            session.delete()
            session = Session.make_new(user_config, config, args.session_path)
        else:
            logger.error("Session %s already exists", args.session_path)
            exit(1)
    elif args.command == "run":
        session = resolve_session(user_config, args.session_path)
        session.run("shellgen" if args.shellgen_only else "full")
    elif args.command == "analyse":
        session = resolve_session(user_config, args.session_path)
        if args.clean:
            session.clean_analysis()
        if args.analysis_config:
            analysis_config = AnalysisConfig.load_json(args.analysis_config)
        else:
            analysis_config = AnalysisConfig()
            for task in args.task:
                analysis_config.handlers.append(TaskTargetConfig(handler=task))
        session.analyse(analysis_config)
    elif args.command == "clean":
        # XXX add safety net question?
        session = resolve_session(user_config, args.session_path)
        session.clean_all()
    elif args.command == "show":
        if args.session_path:
            session = resolve_session(user_config, args.session_path)
        else:
            session = None
        if args.what == "info":
            list_session(session)
        elif args.what == "tasks":
            list_tasks(session)
    elif args.command == "bundle":
        session = resolve_session(user_config, args.session_path)
        session.bundle()
    else:
        # No command
        parser.print_help()
        exit(1)


def main():
    parser = ap.ArgumentParser(description="Benchmark run and plot tool")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-l", "--logfile", type=Path, help="logfile", default=None)
    parser.add_argument("-w",
                        "--workers",
                        type=int,
                        help="Override max number of workers from configuration file",
                        default=None)
    parser.add_argument("-c",
                        "--config",
                        type=Path,
                        help="User environment configuration file. Defaults to ~/.config/cheri-benchplot.json",
                        default=Path("~/.config/cheri-benchplot.json").expanduser())
    sub = parser.add_subparsers(help="command", dest="command")

    sub_session = sub.add_parser("session", help="create a new session from the given configuration")
    add_session_spec_options(sub_session)
    sub_session.add_argument("pipeline_config", type=Path, help="New analysis pipeline configuration file")
    sub_session.add_argument("-f", "--force", action="store_true", help="Force rebuild if existing")

    sub_run = sub.add_parser("run", help="run benchmarks in configuration")
    add_session_spec_options(sub_run)
    sub_run.add_argument("--shellgen-only",
                         action="store_true",
                         help="Only perform shell script generation and stop before running any instance")

    sub_analyse = sub.add_parser("analyse", help="process benchmarks and generate plots")
    add_session_spec_options(sub_analyse)
    sub_analyse.add_argument("-a", "--analysis-config", type=Path, help="Analysis configuration file", default=None)
    sub_analyse.add_argument(
        "-t",
        "--task",
        type=str,
        nargs="+",
        help="Task names to run for the analysis. This is a convenience shorthand for the full analysis configuration")
    sub_analyse.add_argument("--clean", action="store_true", help="Wipe analysis outputs before running")

    sub_clean = sub.add_parser("clean", help="clean output directory")
    add_session_spec_options(sub_clean)

    sub_list = sub.add_parser("show", help="Show session contents and other information")
    sub_list.add_argument("what", choices=["info", "tasks"], help="What to show")
    sub_list.add_argument("session_path", type=Path, help="Path of the target session", nargs='?')

    sub_bundle = sub.add_parser("bundle", help="create a session archive")
    sub_bundle.add_argument("session_path", type=Path, help="Path of the target session")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        exit(1)

    global logger
    logger = setup_logging(args.verbose, args.logfile)
    logger.debug("Loading user config %s", args.config)
    if not args.config.exists():
        logger.error("Missing user configuration file %s", args.config)
        exit(1)

    try:
        if args.config.exists():
            user_config = BenchplotUserConfig.load_json(args.config)
        else:
            logger.debug("Missing user config, using defaults")
        user_config.verbose = args.verbose
    except JSONDecodeError as ex:
        logger.error("Malformed user configuration %s: %s", args.config, ex)
        if args.verbose:
            traceback.print_exception(ex)
            exit(1)

    if args.workers:
        # The argument takes precedence over everything
        user_config.concurrent_workers = args.workers

    # Adjust log level
    if user_config.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        handle_command(user_config, args)
    except Exception as ex:
        logger.error("Failed to run command '%s'", args.command)
        if args.verbose:
            traceback.print_exception(ex)
        exit(1)


if __name__ == "__main__":
    main()
