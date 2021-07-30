#!/bin/python

import logging
import argparse as ap

from pathlib import Path

from pycheribenchplot.core.manager import BenchmarkManager, BenchmarkManagerConfig

def main():
    root_logger = logging.getLogger(None)
    root_logger.setLevel(logging.INFO)

    parser = ap.ArgumentParser(description="Benchmark run and plot tool")
    parser.add_argument("json_config", type=Path, help="Configuration file")
    parser.add_argument("command", choices=["run", "plot"])
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        root_logger.setLevel(logging.DEBUG)
    logging.debug("Loading config %s", args.json_config)
    config = BenchmarkManagerConfig.from_json(args.json_config)
    if args.verbose:
        # Override from command line
        config.verbose = True
    benchmark_manager = BenchmarkManager(config)
    if args.command == "run":
        benchmark_manager.run()
    elif args.command == "plot":
        benchmark_manager.plot()

if __name__ == "__main__":
    main()
