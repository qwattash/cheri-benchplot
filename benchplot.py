#!/bin/python

import logging
import argparse as ap
import uuid

from pathlib import Path

from pycheribenchplot.core.util import setup_logging
from pycheribenchplot.core.manager import BenchmarkManager, BenchmarkManagerConfig

def main():
    parser = ap.ArgumentParser(description="Benchmark run and plot tool")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-l", type=Path, help="logfile", default=None)
    parser.add_argument("json_config", type=Path, help="Configuration file")
    sub = parser.add_subparsers(help="command", dest="command")
    sub_run = sub.add_parser("run", help="run benchmarks in configuration")
    sub_plot = sub.add_parser("plot", help="process benchmarks and generate plots")
    sub_plot.add_argument("session", type=uuid.UUID,
                          help="session ID to plot for, defaults to the latest session recorded",
                          nargs="?", default=None)
    sub_clean = sub.add_parser("clean", help="clean output directory")
    sub_list = sub.add_parser("list", help="list sessions in the output directory")

    args = parser.parse_args()

    logger = setup_logging(args.verbose, args.l)
    logger.debug("Loading config %s", args.json_config)
    config = BenchmarkManagerConfig.load_json(args.json_config)
    if args.verbose:
        # Override from command line
        config.verbose = True
    benchmark_manager = BenchmarkManager(config)
    benchmark_manager.run(args)

if __name__ == "__main__":
    main()
