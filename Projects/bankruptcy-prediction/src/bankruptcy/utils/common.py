"""
============================================================
Shared Utilities
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This module provides utility functions used across all
pipeline components.

Centralising these helpers means every component gets
consistent behaviour for file I/O, directory creation,
and serialisation — with no duplicated try/except blocks
scattered through the codebase.

Bug Fix
-------
Original read_yaml used:

    open(path, "r", encoding="utf-8", errors="ignore")

errors="ignore" silently drops any byte that cannot be
decoded as UTF-8.  If a config file is saved with a BOM
(as requirements.txt was) or any non-ASCII character, the
offending bytes are silently removed and yaml.safe_load
receives a corrupted string.  The result is wrong config
values with no error raised.

Fix: use errors="strict" (the Python default).  Config
files must be valid UTF-8.  If they are not, an explicit
UnicodeDecodeError is raised immediately, pointing directly
at the bad file — far better than a downstream KeyError
or mysterious model behaviour caused by a silently mangled
YAML value.
"""

# ==========================================================
# IMPORTS
# ==========================================================

import json
import os
import yaml
from pathlib import Path


# ==========================================================
# YAML  I/O
# ==========================================================

def read_yaml(path: str) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : str
        Path to the .yaml file.

    Returns
    -------
    dict
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    UnicodeDecodeError
        If the file is not valid UTF-8.
        (Original code used errors='ignore' which silently
        dropped bad bytes and returned corrupted config.)
    yaml.YAMLError
        If the file content is not valid YAML.
    """
    path = str(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML file not found: '{path}'")

    with open(path, "r", encoding="utf-8") as fh:   # strict by default
        return yaml.safe_load(fh)


# ==========================================================
# JSON  I/O
# ==========================================================

def save_json(data: dict, path: str) -> None:
    """
    Serialise a dictionary to a JSON file.

    Handles numpy scalar types (int64, float64, bool_)
    that json.dump() cannot serialise by default.

    Parameters
    ----------
    data : dict
        Data to serialise.
    path : str
        Destination file path. Parent directories are
        created automatically.
    """
    import numpy as np

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, cls=_NumpyEncoder)


def load_json(path: str) -> dict:
    """
    Load a JSON file into a dictionary.

    Parameters
    ----------
    path : str
        Path to the .json file.

    Returns
    -------
    dict

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = str(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: '{path}'")

    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ==========================================================
# DIRECTORY HELPERS
# ==========================================================

def ensure_dir(path: str) -> None:
    """
    Create a directory (and all parents) if it does not exist.

    Equivalent to os.makedirs(path, exist_ok=True) but
    accepts empty strings and None gracefully.

    Parameters
    ----------
    path : str
        Directory path to create.
    """
    if path:
        os.makedirs(path, exist_ok=True)


# ==========================================================
# FILE INFO
# ==========================================================

def get_size_mb(path: str) -> float:
    """
    Return the size of a file in megabytes, rounded to 2 dp.

    Useful for MLflow artifact logging and sanity checks
    on large model or dataset files.

    Parameters
    ----------
    path : str
        Path to the file.

    Returns
    -------
    float
        File size in MB.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = str(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: '{path}'")

    return round(os.path.getsize(path) / (1024 * 1024), 2)