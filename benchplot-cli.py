#!/bin/python

import logging
import argparse as ap
import traceback
import uuid
from json.decoder import JSONDecodeError
from pathlib import Path

from marshmallow.exceptions import ValidationError

from pycheribenchplot.core.config import PipelineConfig, BenchplotUserConfig, AnalysisConfig
from pycheribenchplot.core.session import SessionAnalysisMode
from pycheribenchplot.core.pipeline import PipelineManager
from pycheribenchplot.core.util import setup_logging

def add_session_spec_options(parser):
    """
    Helper to add options to identify a session
    """
    session_name = parser.add_argument("session_path", type=Path, help="Path or name of the target session")


def resolve_session(args, manager, logger):
    """
    Helper to resolve a session depending on the argument passed
    """
    session = manager.resolve_session(args.session_path)
    if session is None:
        logger.error("Session %s does not exist", args.session_path)
        exit(1)
    return session

def list_session(manager, session, logger):
    if session is None:
        logger.error("No session %s", args.session_path)
        return

    configurations = session.config.configurations
    print(f"Session {session.config.name} ({session.config.uuid}):")
    for c in session.config.configurations:
        print("\t", c)

def list_analysis(manager, session, logger):
    """
    Helper to display analysis handlers for a given manager
    """
    handlers = manager.get_analysis_handlers(session)
    print("Analysis handlers:")
    for h in handlers:
        if h.cross_analysis:
            continue
        print("\t", h)

    print("Cross-parameter analysis handlers:")
    for h in handlers:
        if h.cross_analysis:
            print("\t", h)


def main():
    parser = ap.ArgumentParser(description="Benchmark run and plot tool")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-l", "--logfile", type=Path, help="logfile", default=None)
    parser.add_argument("-c", "--config", type=Path, help="User environment configuration file",
                        default=Path("~/.config/cheri-benchplot.json").expanduser())
    sub = parser.add_subparsers(help="command", dest="command")

    sub_session = sub.add_parser("session", help="create a new session from the given configuration")
    add_session_spec_options(sub_session)
    sub_session.add_argument("pipeline_config", type=Path, help="New analysis pipeline configuration file")
    sub_session.add_argument("-f", "--force", action="store_true", help="Force rebuild if existing")

    sub_run = sub.add_parser("run", help="run benchmarks in configuration")
    add_session_spec_options(sub_run)
    sub_run.add_argument("--shellgen-only", action="store_true",
                         help="Only perform shell script generation and stop before running any instance")

    sub_analyse = sub.add_parser("analyse", help="process benchmarks and generate plots")
    add_session_spec_options(sub_analyse)
    sub_analyse.add_argument("--mode", choices=[m.value for m in SessionAnalysisMode], help="Analysis mode",
                             default=SessionAnalysisMode.ASYNC_LOAD)
    sub_analyse.add_argument("-a", "--analysis-config", type=Path, help="Analysis configuration file",
                             default=None)

    sub_clean = sub.add_parser("clean", help="clean output directory")
    add_session_spec_options(sub_clean)

    sub_list = sub.add_parser("list", help="List sessions and other information")
    sub_list.add_argument("what", choices=["session", "analysis"], help="What to show")
    sub_list.add_argument("session_path", type=Path, help="Path of the target session",
                          nargs='?')

    sub_bundle = sub.add_parser("bundle", help="create a session archive")
    sub_bundle.add_argument("session_path", type=Path, help="Path of the target session")

    args = parser.parse_args()

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

    if user_config.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        manager = PipelineManager(user_config)

        if args.command == "session":
            try:
                config = PipelineConfig.load_json(args.pipeline_config)
            except JSONDecodeError as ex:
                logger.error("Malformed pipeline configuration %s: %s", args.pipeline_config, ex)
                raise
            except ValidationError as ex:
                logger.error("Invalid pipeline configuration %s: %s", args.pipeline_config, ex)
                raise
            existing = manager.resolve_session(args.session_path)
            if existing:
                if args.force:
                    manager.delete_session(existing)
                else:
                    logger.error("Session %s already exists", args.session_path)
                    exit(1)
            manager.make_session(args.session_path, config)
        elif args.command == "run":
            session = resolve_session(args, manager, logger)
            manager.run_session(session, shellgen_only=args.shellgen_only)
        elif args.command == "analyse":
            session = resolve_session(args, manager, logger)
            if args.analysis_config:
                analysis_config = AnalysisConfig.load_json(args.analysis_config)
            else:
                analysis_config = AnalysisConfig()
            manager.run_analysis(session, analysis_config, mode=args.mode)
        elif args.command == "clean":
            raise NotImplementedError("TODO")
        elif args.command == "list":
            if args.session_path:
                session = resolve_session(args, manager, logger)
            else:
                session = None
            if args.what == "session":
                list_session(manager, session, logger)
            elif args.what == "analysis":
                list_analysis(manager, session, logger)
        elif args.command == "bundle":
            session = resolve_session(args, manager, logger)
            manager.bundle(session)
        else:
            # No command
            parser.print_help()
            exit(1)
    except Exception as ex:
        logger.error("Failed to run command '%s'", args.command)
        if args.verbose:
            traceback.print_exception(ex)
        exit(1)

if __name__ == "__main__":
    main()
