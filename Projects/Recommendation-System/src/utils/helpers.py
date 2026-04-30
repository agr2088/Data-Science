"""
============================================================
Helper Utilities
============================================================

Purpose
-------
Shared utility functions used across all pipeline modules.

Includes
--------
- save_pickle / load_pickle  — serialization helpers
- timer                      — function timing decorator
- encode_binary              — Yes/No → 1/0 encoding
- encode_ordinal             — ordinal category encoding
- clip_outliers              — value clipping
- compute_completion_rate    — derived feature helper
- summarize_df               — DataFrame logging summary
"""

import functools
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _patch_numpy_pickle_compat() -> None:
    """
    Provide backward/forward compatibility for NumPy internal module paths
    embedded in pickled artifacts.
    """

    import numpy.core.multiarray as np_multiarray
    import numpy.core.numeric as np_numeric
    import numpy.core.umath as np_umath

    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)
    sys.modules.setdefault("numpy._core.umath", np_umath)


def format_duration(seconds: float) -> str:
    """Render elapsed time in a compact human-readable form."""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rem:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {rem:.0f}s"


def log_banner(title: str, subtitle: str | None = None) -> None:
    line = "=" * 68
    logger.info("")
    logger.info(line)
    logger.info(title)
    if subtitle:
        logger.info(subtitle)
    logger.info(line)


def log_stage(number: int, total: int, title: str, objective: str | None = None) -> None:
    logger.info("")
    logger.info(f"STAGE {number}/{total} | {title}")
    logger.info("-" * 68)
    if objective:
        logger.info(objective)


def log_stage_result(title: str, elapsed: float, details: str | None = None) -> None:
    message = f"Stage complete | {title} | {format_duration(elapsed)}"
    if details:
        message += f" | {details}"
    logger.info(message)


# ==========================================================
# Serialization Utilities
# ==========================================================

def save_pickle(obj: Any, path: Path) -> None:
    """
    Serialize and save any Python object to disk as a pickle file.

    Parameters
    ----------
    obj  : Any   — object to serialize
    path : Path  — destination file path
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Saved artifact | {path.name}")


def load_pickle(path: Path) -> Any:
    """
    Load and return a serialized Python object from disk.

    Parameters
    ----------
    path : Path — path to the pickle file

    Returns
    -------
    Any — deserialized object
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Pickle not found: {path}")

    _patch_numpy_pickle_compat()

    with open(path, "rb") as f:
        obj = pickle.load(f)

    logger.info(f"Loaded artifact | {path.name}")
    return obj


# ==========================================================
# Timing Decorator
# ==========================================================

def timer(func):
    """
    Decorator that logs the execution time of any function.

    Usage
    -----
        @timer
        def my_function():
            ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start  = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        owner = ""
        if args and hasattr(args[0], "__class__"):
            owner = f"{args[0].__class__.__name__}."
        logger.info(f"Completed {owner}{func.__name__} | {format_duration(elapsed)}")
        return result

    return wrapper


# ==========================================================
# DataFrame Encoding Helpers
# ==========================================================

def encode_binary(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Convert Yes/No string columns to integer 1/0.

    Parameters
    ----------
    df      : DataFrame
    columns : list of column names to encode

    Returns
    -------
    DataFrame with encoded columns.
    """

    mapping = {"Yes": 1, "No": 0}

    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    return df


def encode_ordinal(df: pd.DataFrame, col: str, order: list) -> pd.DataFrame:
    """
    Encode an ordinal categorical column as integers based on a
    user-defined ordering.

    FIX: Previously used a bare .map() with no fallback, so any value
    not present in `order` (e.g. a typo in difficulty_level) would
    silently become NaN and propagate into training data undetected.
    Now logs a warning and fills unknown values with 0 (the lowest
    ordinal level) so the pipeline never produces silent NaN rows.

    Parameters
    ----------
    df    : DataFrame
    col   : column name
    order : list in ascending order (e.g. ["Beginner", "Intermediate", "Advanced"])

    Returns
    -------
    DataFrame with the encoded column (integer, no NaNs).
    """

    mapping = {v: i for i, v in enumerate(order)}

    # Count how many values are NOT in the known mapping before mapping
    n_before_nulls = int(df[col].isna().sum())
    df[col] = df[col].map(mapping)
    n_after_nulls  = int(df[col].isna().sum())

    n_unknown = n_after_nulls - n_before_nulls
    if n_unknown > 0:
        logger.warning(
            f"encode_ordinal: {n_unknown} unrecognised value(s) in column "
            f"'{col}' (not in {order}). Filling with 0 (lowest ordinal level)."
        )

    # Fill any NaN (including pre-existing ones) with 0 and cast to int
    df[col] = df[col].fillna(0).astype(int)

    return df


# ==========================================================
# Outlier Clipping
# ==========================================================

def clip_outliers(
    df: pd.DataFrame,
    col: str,
    lower: float,
    upper: float,
) -> pd.DataFrame:
    """
    Clip values in a column to the range [lower, upper].

    Parameters
    ----------
    df    : DataFrame
    col   : column name
    lower : minimum allowed value
    upper : maximum allowed value

    Returns
    -------
    DataFrame with clipped column.
    """

    before = df[col].copy()
    df[col] = df[col].clip(lower=lower, upper=upper)
    n_clipped = (before != df[col]).sum()

    if n_clipped:
        logger.debug(
            f"Clipped {n_clipped} outliers in '{col}' to [{lower}, {upper}]"
        )

    return df


# ==========================================================
# Feature Computation
# ==========================================================

def compute_completion_rate(df: pd.DataFrame) -> pd.Series:
    """
    Compute a proxy completion rate as:
        time_spent_hours / course_duration_hours

    Result is clipped to [0, 1] to handle edge cases
    where time_spent > duration.

    Parameters
    ----------
    df : DataFrame with 'time_spent_hours' and 'course_duration_hours'

    Returns
    -------
    pd.Series of completion rates in [0, 1].
    """

    rate = (df["time_spent_hours"] / df["course_duration_hours"]).clip(0, 1)
    return rate


# ==========================================================
# DataFrame Summary Logger
# ==========================================================

def summarize_df(df: pd.DataFrame, label: str = "") -> None:
    """
    Log shape, null counts, and dtypes for a DataFrame in a clean format.

    Parameters
    ----------
    df    : DataFrame to summarize
    label : optional tag shown in log output (e.g. "raw", "processed")
    """

    tag = f"[{label.upper()}] " if label else ""

    # 1. Clean Shape
    logger.info(f"{tag}Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 2. Clean Nulls
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        missing_cols = list(null_counts[null_counts > 0].index)
        logger.info(f"{tag}Missing values: detected in {missing_cols}")
    else:
        logger.info(f"{tag}Missing values: none")

    # 3. Clean Dtypes (Summarized instead of a massive list)
    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()
    dtype_str = ", ".join([f"{v} {k}" for k, v in dtype_counts.items()])
    logger.info(f"{tag}Dtypes: {dtype_str}")