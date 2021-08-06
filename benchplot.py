#!/bin/python

import logging
import argparse as ap

from pathlib import Path

from pycheribenchplot.core.manager import BenchmarkManager, BenchmarkManagerConfig

def main():
    parser = ap.ArgumentParser(description="Benchmark run and plot tool")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-l", type=Path, help="logfile", default=None)
    parser.add_argument("json_config", type=Path, help="Configuration file")
    sub = parser.add_subparsers(help="command", dest="command")
    sub_run = sub.add_parser("run", help="run benchmarks in configuration")
    sub_plot = sub.add_parser("plot", help="process benchmarks and generate plots")

    args = parser.parse_args()

    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    logging.basicConfig(filename=args.l, level=level, filemode="w")
    logging.debug("Loading config %s", args.json_config)
    config = BenchmarkManagerConfig.load_json(args.json_config)
    if args.verbose:
        # Override from command line
        config.verbose = True
    benchmark_manager = BenchmarkManager(config)
    benchmark_manager.run(args.command)

if __name__ == "__main__":
    main()
