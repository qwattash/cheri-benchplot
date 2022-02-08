#!/bin/python

import logging
import argparse as ap
import uuid

from pathlib import Path

from pycheribenchplot.core.util import setup_logging
from pycheribenchplot.core.manager import BenchmarkManager, BenchmarkSessionConfig, BenchplotUserConfig

def main():
    parser = ap.ArgumentParser(description="Benchmark run and plot tool")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-l", "--logfile", type=Path, help="logfile", default=None)
    parser.add_argument("-c", "--config", type=Path, help="User environment configuration file",
                        default=Path("~/.config/cheri-benchplot.json").expanduser())
    parser.add_argument("session_config", type=Path, help="Session configuration file")
    sub = parser.add_subparsers(help="command", dest="command")
    sub_run = sub.add_parser("run", help="run benchmarks in configuration")
    sub_analyse = sub.add_parser("analyse", help="process benchmarks and generate plots")
    sub_analyse.add_argument("session", type=uuid.UUID,
                          help="session ID to plot for, defaults to the latest session recorded",
                          nargs="?", default=None)
    sub_analyse.add_argument("--interactive",
                             choices=["load", "pre-merge", "merge", "aggregate"],
                             help="Interact with live datasets")
    sub_analyse.add_argument("-a", "--analysis-config", type=Path, help="Analysis configuration file",
                             default=None)
    sub_clean = sub.add_parser("clean", help="clean output directory")
    sub_list = sub.add_parser("list", help="list sessions in the output directory")

    args = parser.parse_args()

    logger = setup_logging(args.verbose, args.logfile)
    logger.debug("Loading user config %s", args.config)
    if not args.config.exists():
        logger.error("Missing user configuration file %s", args.config)
        exit(1)
    user_config = BenchplotUserConfig.load_json(args.config)
    logger.debug("Loading session config %s", args.session_config)
    if not args.session_config.exists():
        logger.error("Session config does not exist %s", args.session_config)
        exit(1)
    config = BenchmarkSessionConfig.load_json(args.session_config)
    if args.verbose:
        # Override from command line
        config.verbose = True
    benchmark_manager = BenchmarkManager(user_config, config)
    benchmark_manager.run(args)

if __name__ == "__main__":
    main()
