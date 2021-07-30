#!/bin/python

import logging
import argparse as ap
from pathlib import Path

from pycheribenchplot.core.instanced import InstanceDaemon, InstanceDaemonConfig

def main():
    root_logger = logging.getLogger(None)
    root_logger.setLevel(logging.INFO)

    parser = ap.ArgumentParser(description="Benchmark runner instance daemon")
    parser.add_argument("json_config", type=Path, help="Configuration file")
    parser.add_argument("-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.v:
        root_logger.setLevel(logging.DEBUG)
    logging.debug("Loading config %s", args.json_config)
    config = InstanceDaemonConfig.from_json(args.json_config)
    if args.v:
        # Override from command line
        config.verbose = True
    daemon = InstanceDaemon(config)
    daemon.start()

if __name__ == "__main__":
    main()
