"""
src/utils/helpers.py
World Development Clustering — Shared Utility Functions

All generic helper functions used across pipeline stages live here.
No business logic — only reusable I/O, parsing, and DataFrame utilities.
"""

import os
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── String parsers ────────────────────────────────────────────────────────────

def strip_currency(val) -> float:
    """
    Parse World Bank currency string to float.
    '$1,234,567' → 1234567.0
    None / NaN  → np.nan
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    try:
        return float(str(val).replace("$", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return np.nan


def strip_percent(val) -> float:
    """
    Parse percent string to float.
    '76.9%' → 76.9
    None / NaN → np.nan
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    try:
        return float(str(val).replace("%", "").strip())
    except (ValueError, TypeError):
        return np.nan


# ── DataFrame I/O ─────────────────────────────────────────────────────────────

def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved DataFrame ({df.shape}) → {path}")


def load_dataframe(path: str) -> pd.DataFrame:
    """Load CSV to DataFrame with basic validation."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected file not found: {path}\n"
            "Run the pipeline stages in order: ingestion → preprocessing → features."
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded DataFrame ({df.shape}) ← {path}")
    return df


# ── Diagnostic utilities ──────────────────────────────────────────────────────

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame summarising missing values per column.
    Columns: [column, n_missing, pct_missing]
    Sorted by pct_missing descending.
    """
    n_missing = df.isnull().sum()
    pct       = (n_missing / len(df) * 100).round(2)
    summary   = pd.DataFrame({
        "column"     : df.columns,
        "n_missing"  : n_missing.values,
        "pct_missing": pct.values,
    }).sort_values("pct_missing", ascending=False).reset_index(drop=True)
    return summary[summary["n_missing"] > 0]


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extended descriptive statistics for all numeric columns.
    Adds skewness and kurtosis to the standard describe() output.
    """
    desc              = df.describe().T
    desc["skewness"]  = df.skew(numeric_only=True)
    desc["kurtosis"]  = df.kurtosis(numeric_only=True)
    return desc.round(4)


def get_numeric_feature_cols(df: pd.DataFrame, exclude: list = None) -> list:
    """
    Return names of all numeric columns, optionally excluding a list of names.
    Used to safely extract the feature matrix without hardcoding column names.

    Default exclusions: ID column 'Country' is not numeric so is never returned;
    no extra exclusions are applied unless the caller passes them.
    """
    exclude = set(exclude) if exclude is not None else set()
    return [c for c in df.select_dtypes(include="number").columns if c not in exclude]