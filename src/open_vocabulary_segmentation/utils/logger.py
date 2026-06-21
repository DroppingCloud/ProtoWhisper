# ------------------------------------------------------------------------------
# FreeDA / Talk2DINO
# ------------------------------------------------------------------------------

import logging
import os.path as osp

from mmcv.utils import get_logger as get_root_logger

try:
    from termcolor import colored
except ImportError:  # keep the project runnable even when termcolor is missing
    def colored(text, *args, **kwargs):
        return text


logger_name = None


class FriendlyStreamFormatter(logging.Formatter):
    """Colorize terminal logs while keeping file logs clean."""

    LEVEL_COLORS = {
        "DEBUG": "blue",
        "INFO": "cyan",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red",
    }

    def format(self, record):
        original_levelname = record.levelname
        color = self.LEVEL_COLORS.get(original_levelname, "white")
        attrs = ["bold"] if record.levelno >= logging.ERROR else None
        record.levelname = colored(f"{original_levelname:<8}", color, attrs=attrs)
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


def get_logger(cfg=None, log_level=logging.INFO):
    global logger_name
    if cfg is None:
        return get_root_logger(logger_name)

    name = cfg.model_name
    output = cfg.output
    logger_name = name

    logger = get_root_logger(
        name,
        osp.join(output, "log.txt"),
        log_level=log_level,
        file_mode="a",
    )
    logger.propagate = False

    # Terminal: compact, colored, readable.
    stream_fmt = (
        colored("[%(asctime)s]", "green")
        + " %(levelname)s "
        + colored("%(name)s", "magenta")
        + " | %(message)s"
    )

    # File: keep filename/line number for debugging, but no ANSI colors.
    file_fmt = "[%(asctime)s %(name)s] (%(filename)s:%(lineno)d): %(levelname)s %(message)s"

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setFormatter(logging.Formatter(fmt=file_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        elif isinstance(handler, logging.StreamHandler):
            handler.setFormatter(FriendlyStreamFormatter(fmt=stream_fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    return logger
