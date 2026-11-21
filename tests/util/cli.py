"""
Helpers for testing benchplot-cli subcommands without subprocess invocation.
"""

import argparse as ap
from types import SimpleNamespace

from pycheribenchplot.core.tool import SubCommand
from pycheribenchplot.core.util import new_logger


def make_subcommand(cmd: SubCommand) -> SubCommand:
    """
    Attach a stub parent so a subcommand can be used standalone in tests.
    """
    cmd.parent = SimpleNamespace(logger=new_logger("test-cli"))
    return cmd


def parse_args(cmd: SubCommand, argv: list[str]) -> ap.Namespace:
    """
    Parse CLI arguments for a subcommand through its real option registrations.
    """
    parser = ap.ArgumentParser()
    cmd.register_options(parser)
    return parser.parse_args(argv)
