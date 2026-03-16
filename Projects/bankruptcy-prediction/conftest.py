"""
============================================================
Pytest Configuration
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This file configures the Python path so that pytest can
import modules from the project source directory.

Without this configuration, tests may fail with:

    ModuleNotFoundError: No module named 'bankruptcy'

Bug Fix
-------
Original code:

    PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

Problem:
    conftest.py lives at the project root:
        <project>/conftest.py

    os.path.dirname(__file__) already returns the project
    root directory.  Appending ".." therefore navigates ONE
    LEVEL UP to the *parent* of the project, which has no
    "src/" subdirectory.

    The resulting SRC_PATH pointed at a non-existent folder
    and was silently ignored by Python.

    The tests appeared to pass in development only because
    `pip install -e .` registers the package directly with
    site-packages, making the conftest.py path manipulation
    unnecessary.  In a fresh CI environment that skips the
    editable install, every test would fail with:

        ModuleNotFoundError: No module named 'bankruptcy'

Fix:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    os.path.abspath(__file__) resolves any symlinks and
    relative components in __file__ before calling dirname,
    giving the real project root directory regardless of
    how pytest was invoked (cwd, symlink, subprocess, etc.).
"""

import sys
import os


# ==========================================================
# ADD PROJECT SRC DIRECTORY TO PYTHON PATH
# ==========================================================

# __file__ is conftest.py, which lives at the project root.
# dirname(__file__) IS the project root — do not append "..".
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)