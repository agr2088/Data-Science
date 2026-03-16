"""
============================================================
Logger Tests
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This test module verifies that the centralized logging
system works correctly.

The logger is used across the ML pipeline to record
important events such as:

• pipeline stage execution
• warnings
• errors
• debugging information

Why This Test Matters
---------------------
Logging is essential for monitoring and debugging ML
pipelines in production environments.

This test ensures that:

• logger handlers are configured
• log file is created
• log messages are written successfully
"""

# ==========================================================
# IMPORT REQUIRED MODULES
# ==========================================================

import os
import logging

from bankruptcy.utils.logger import logger


# ==========================================================
# TEST: LOGGER FUNCTIONALITY
# ==========================================================

def test_logger_creates_log_file_and_logs_message():
    """
    Verify that the logger:

    • contains active handlers
    • includes a file handler
    • creates a log file
    • writes log messages successfully
    """

    # ======================================================
    # 1️⃣ VERIFY LOGGER HAS HANDLERS
    # ======================================================

    assert len(logger.handlers) >= 1


    # ======================================================
    # 2️⃣ FIND FILE HANDLER
    # ======================================================

    file_handlers = [
        handler for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
    ]

    assert len(file_handlers) == 1

    file_handler = file_handlers[0]

    log_file_path = file_handler.baseFilename


    # ======================================================
    # 3️⃣ WRITE TEST LOG MESSAGE
    # ======================================================

    test_message = "Logger test message"

    logger.info(test_message)


    # ======================================================
    # 4️⃣ VERIFY LOG FILE EXISTS
    # ======================================================

    assert os.path.exists(log_file_path)


    # ======================================================
    # 5️⃣ VERIFY MESSAGE WRITTEN TO LOG FILE
    # ======================================================

    with open(log_file_path, "r") as f:

        content = f.read()

    assert test_message in content