#!/bin/python

import logging
import argparse as ap
import traceback
import uuid
from json.decoder import JSONDecodeError
from pathlib import Path

from pycheribenchplot.core.config import PipelineConfig, BenchplotUserConfig, AnalysisConfig
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
    sub_analyse.add_argument("--interactive",
                             choices=["load", "pre-merge", "merge", "aggregate"],
                             help="Interact with live datasets")
    sub_analyse.add_argument("-a", "--analysis-config", type=Path, help="Analysis configuration file",
                             default=None)

    sub_clean = sub.add_parser("clean", help="clean output directory")
    add_session_spec_options(sub_clean)

    sub_list = sub.add_parser("list", help="list sessions in the output directory")

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
            manager.run_analysis(session, analysis_config, interactive=args.interactive)
        elif args.command == "clean":
            raise NotImplementedError("TODO")
        elif args.command == "list":
            raise NotImplementedError("TODO")
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
