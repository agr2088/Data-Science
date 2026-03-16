"""
============================================================
Central Logging System
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This module provides a centralized logging system used across
the entire machine learning pipeline.

Logging allows us to monitor pipeline execution, debug errors,
and track important events during model training and deployment.

Responsibilities
----------------
• Configure a global project logger
• Write logs to both console and file
• Maintain structured log formatting
• Prevent duplicate log handlers
• Ensure UTF-8 safe logging

Why Logging Matters
-------------------
Machine learning pipelines can involve multiple stages such as:

Data Ingestion
      ↓
Data Validation
      ↓
Data Transformation
      ↓
Model Training
      ↓
Model Evaluation

Without logging, it becomes extremely difficult to:

• identify failures
• debug pipeline issues
• monitor pipeline execution
• audit system behavior

Therefore, this logger acts as the **central monitoring system**
for the entire project.
"""

# ==========================================================
# 1. IMPORT REQUIRED LIBRARIES
# ==========================================================

import logging
import os
from datetime import datetime


# ==========================================================
# 2. CONFIGURE LOG DIRECTORY
# ==========================================================
"""
Logs are stored inside a configurable directory.

The directory can be changed using the environment variable
LOG_DIR. This makes the logging system compatible with:

• local development environments
• Docker containers
• cloud deployments
"""

LOG_DIR = os.getenv("LOG_DIR", "logs")

os.makedirs(LOG_DIR, exist_ok=True)


# ==========================================================
# 3. GENERATE DAILY LOG FILE
# ==========================================================
"""
A new log file is created each day to keep logs organized.
"""

LOG_FILE = f"bankruptcy_{datetime.now().strftime('%Y_%m_%d')}.log"

LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)


# ==========================================================
# 4. CREATE PROJECT LOGGER
# ==========================================================

logger = logging.getLogger("bankruptcyLogger")

logger.setLevel(logging.INFO)

# Prevent logs from propagating to the root logger
logger.propagate = False


# ==========================================================
# 5. PREVENT DUPLICATE HANDLERS
# ==========================================================
"""
This condition ensures that handlers are added only once.

Without this safeguard, importing the logger multiple times
could cause duplicate log messages.
"""

if not logger.handlers:

    # ------------------------------------------------------
    # FILE HANDLER
    # ------------------------------------------------------
    """
    Writes logs to a persistent log file.
    """

    file_handler = logging.FileHandler(
        LOG_FILE_PATH,
        mode="a",
        encoding="utf-8"
    )

    file_handler.setLevel(logging.INFO)


    # ------------------------------------------------------
    # CONSOLE HANDLER
    # ------------------------------------------------------
    """
    Prints logs to the terminal during pipeline execution.
    """

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)


    # ------------------------------------------------------
    # LOG FORMATTER
    # ------------------------------------------------------
    """
    Structured log format for better readability.

    Example log format:

    [2026-03-15 10:30:00] [INFO] [bankruptcyLogger]
    model_trainer.py:120 - Model Training Started
    """

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] "
        "%(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


    # Apply formatter
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)


    # Attach handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)