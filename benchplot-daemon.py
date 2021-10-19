#!/bin/python

import logging
import argparse as ap
from pathlib import Path

from pycheribenchplot.core.util import setup_logging
from pycheribenchplot.core.instanced import InstanceDaemon, InstanceDaemonConfig

def main():
    parser = ap.ArgumentParser(description="Benchmark runner instance daemon")
    parser.add_argument("json_config", type=Path, help="Configuration file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-l", type=Path, help="Log file", default=None)

    args = parser.parse_args()

    logger = setup_logging(args.verbose, args.l)
    logger.debug("Loading config %s", args.json_config)
    config = InstanceDaemonConfig.load_json(args.json_config)
    if args.v:
        # Override from command line
        config.verbose = True
    daemon = InstanceDaemon(config)
    daemon.start()

if __name__ == "__main__":
    main()
