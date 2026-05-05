"""
src/utils/logger.py
World Development Clustering — Logging Utility

Single logger factory used by every module in the pipeline.
All log output goes to both console (INFO+) and logs/pipeline.log (DEBUG+).
"""

import logging
import os
import sys

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_LOGS_DIR     = os.path.join(_PROJECT_ROOT, "logs")

_ROOT_LOGGER_CONFIGURED = False


def _configure_root_logger() -> None:
    """One-time setup of the root logger with file + stream handlers."""
    global _ROOT_LOGGER_CONFIGURED
    if _ROOT_LOGGER_CONFIGURED:
        return

    os.makedirs(_LOGS_DIR, exist_ok=True)
    log_path = os.path.join(_LOGS_DIR, "pipeline.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File — DEBUG and above (full trace)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    _ROOT_LOGGER_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger. Call once per module:
        logger = get_logger(__name__)
    """
    _configure_root_logger()
    return logging.getLogger(name)