"""
============================================================
Logger Utility
============================================================

Purpose
-------
Provide a consistent, reusable logger for all modules.

Usage
-----
    from src.utils.logger import get_logger
    logger = get_logger(__name__)

Output
------
- Console  — INFO level and above
- Log file — DEBUG level and above (logs/pipeline.log)
"""

import logging
import sys
from pathlib import Path
from config.config import LOG_FILE


# ==========================================================
# Logger Factory
# ==========================================================

def get_logger(name: str) -> logging.Logger:
    """
    Create and return a named logger with console + file handlers.

    Handlers are added only once — safe for repeated imports.

    Parameters
    ----------
    name : str
        Logger name (use __name__ in each module).

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(name)

    # -------------------------------------------------
    # Avoid duplicate handlers on re-import
    # -------------------------------------------------

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # -------------------------------------------------
    # Formatter 1: Detailed (For the Log File)
    # -------------------------------------------------
    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # -------------------------------------------------
    # Formatter 2: Clean & Simple (For the Terminal)
    # -------------------------------------------------
    console_fmt = logging.Formatter(
        fmt="%(asctime)s | %(message)s", 
        datefmt="%H:%M:%S"
    )

    # -------------------------------------------------
    # Console handler — INFO and above
    # -------------------------------------------------

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_fmt)

    # -------------------------------------------------
    # File handler — DEBUG and above
    # -------------------------------------------------
    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
