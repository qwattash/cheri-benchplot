import logging
import time
from contextlib import contextmanager
from enum import Enum
from pathlib import Path

import termcolor


class LogColorFormatter(logging.Formatter):
    colors = {
        logging.DEBUG: "white",
        logging.INFO: "blue",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "red"
    }

    def __init__(self, *args, use_colors=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_colors = use_colors

    def format(self, record):
        msg = super().format(record)
        if self.use_colors:
            color = self.colors[record.levelno]
            return termcolor.colored(msg, color)
        return msg


def setup_logging(verbose: bool = False, logfile: Path = None):
    log_fmt = "[%(levelname)s] %(name)s: %(message)s"
    date_fmt = None
    default_level = logging.INFO
    if verbose:
        default_level = logging.DEBUG
    logger = logging.getLogger("cheri-benchplot")
    logger.setLevel(default_level)
    logger.propagate = False
    # Console logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(LogColorFormatter(fmt=log_fmt))
    logger.addHandler(console_handler)
    # File logging
    if logfile:
        file_handler = logging.FileHandler(logfile, mode="w")
        file_handler.setFormatter(logging.Formatter(fmt=log_fmt))
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    return logger


def new_logger(name, parent=None):
    if parent is None:
        parent = logging.getLogger("cheri-benchplot")
    return parent.getChild(name)


@contextmanager
def timing(name, level=logging.INFO, logger=None):
    if logger is None:
        logger = logging
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        logger.log(level, "%s in %.2fs", name, end - start)
