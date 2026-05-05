"""
src/features/feature_selection.py
World Development Clustering — Pipeline Stage 3b: Feature Selection

Responsibilities:
  - Remove low-variance features (threshold from config)
  - Save final_features.csv (input to model training)

Note: High-correlation removal is handled in feature_engineering.py.
      This module handles only variance-based selection.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

import numpy as np
import pandas as pd

from config.config import (
    PROCESSED_DIR,
    VARIANCE_FILTER_THRESHOLD,
    ID_COL,
)
from src.utils.logger import get_logger
from src.utils.helpers import save_dataframe, get_numeric_feature_cols

logger = get_logger(__name__)

FINAL_FEATURES_PATH = os.path.join(PROCESSED_DIR, "final_features.csv")


def remove_low_variance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop numeric columns whose variance falls below VARIANCE_FILTER_THRESHOLD.

    After StandardScaler, most features will have meaningful variance.
    Columns that are nearly constant after aggregation add noise to clustering
    distance computations without contributing signal.

    Never drops ID_COL.
    """
    feature_cols = get_numeric_feature_cols(df)
    variances    = df[feature_cols].var()
    low_var_cols = variances[variances < VARIANCE_FILTER_THRESHOLD].index.tolist()

    if low_var_cols:
        df = df.drop(columns=low_var_cols)
        logger.info(
            f"Removed {len(low_var_cols)} low-variance columns "
            f"(threshold={VARIANCE_FILTER_THRESHOLD}): {low_var_cols}"
        )
    else:
        logger.info(f"No low-variance columns to remove (threshold={VARIANCE_FILTER_THRESHOLD})")
    return df


def run_feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature selection pipeline.

    Steps:
      1. Remove low-variance features
      2. Save final_features.csv

    Args:
        df: feature-reduced DataFrame from feature_engineering.py

    Returns:
        df_selected: final feature set ready for model training
    """
    logger.info(f"Input shape: {df.shape}")
    feature_cols_before = get_numeric_feature_cols(df)

    df = remove_low_variance(df)

    feature_cols_after = get_numeric_feature_cols(df)
    removed = set(feature_cols_before) - set(feature_cols_after)

    logger.info(
        f"Feature selection complete — "
        f"{len(feature_cols_before)} → {len(feature_cols_after)} features "
        f"({len(removed)} removed)"
    )

    save_dataframe(df, FINAL_FEATURES_PATH)
    return df


if __name__ == "__main__":
    from src.features.feature_engineering import run_feature_engineering
    df_features, _ = run_feature_engineering()
    run_feature_selection(df_features)
