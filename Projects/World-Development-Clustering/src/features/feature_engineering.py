"""
src/features/feature_engineering.py
World Development Clustering — Pipeline Stage 3a: Feature Engineering

Responsibilities:
  - Drop multicollinear columns (defined in config)
  - Compute correlation matrix and drop features with |r| > 0.90
  - Save feature_reduced.csv (primary clustering input — NO PCA)
  - Apply PCA ONLY for 2D visualization → save pca_data.csv + pca_model.pkl

PCA RULE: PCA is NEVER applied before clustering. It is used ONLY to generate
          2D coordinates for scatter plot visualization.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

import numpy as np
import pandas as pd
import joblib

from sklearn.decomposition import PCA

from config.config import (
    SCALED_DATA,
    PROCESSED_DIR,
    PCA_MODEL_PATH,
    PCA_VARIANCE_THRESHOLD,
    MULTICOLLINEAR_DROP,
    CORRELATION_DROP_THRESHOLD,
    ID_COL,
    RANDOM_STATE,
)
from src.utils.logger import get_logger
from src.utils.helpers import save_dataframe, load_dataframe, get_numeric_feature_cols

logger = get_logger(__name__)

FEATURE_REDUCED_PATH = os.path.join(PROCESSED_DIR, "feature_reduced.csv")
PCA_DATA_PATH        = os.path.join(PROCESSED_DIR, "pca_data.csv")


def drop_multicollinear(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop pre-identified multicollinear columns from MULTICOLLINEAR_DROP config.
    These are known high-correlation pairs identified from domain knowledge:
      - CO2 Emissions × Energy Usage:   r = 0.993
      - Life Expectancy Female × Male:  r = 0.978
      - Population 15-64 is derived from the other two population bands
    """
    cols_to_drop = [c for c in MULTICOLLINEAR_DROP if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped {len(cols_to_drop)} known multicollinear columns: {cols_to_drop}")
    else:
        logger.info("No pre-identified multicollinear columns found")
    return df


def remove_high_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix and drop features with |r| > CORRELATION_DROP_THRESHOLD (0.90).

    Dynamic second-pass to catch any remaining correlated pairs after the
    pre-identified drops. Uses upper-triangle to avoid double-counting.
    Strategy: drop the column that appears later in column order (deterministic).
    """
    feature_cols = get_numeric_feature_cols(df)
    corr_matrix  = df[feature_cols].corr().abs()
    upper        = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    cols_to_drop = [
        col for col in upper.columns
        if any(upper[col] > CORRELATION_DROP_THRESHOLD)
    ]

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(
            f"Dropped {len(cols_to_drop)} high-correlation columns "
            f"(|r| > {CORRELATION_DROP_THRESHOLD}): {cols_to_drop} | shape: {df.shape}"
        )
    else:
        logger.info(f"No high-correlation columns to remove (|r| > {CORRELATION_DROP_THRESHOLD})")

    return df


def apply_pca_for_visualization(df: pd.DataFrame) -> tuple:
    """
    Apply PCA ONLY for 2D/3D visualization.

    CRITICAL: This function is called AFTER feature_reduced.csv is saved.
    The PCA output is NEVER used as input to any clustering model.
    It is used exclusively for scatter plot coordinates in notebooks/dashboard.

    Retains PCA_VARIANCE_THRESHOLD (90%) variance components.
    Saves pca_model.pkl + pca_data.csv.
    """
    feature_cols = get_numeric_feature_cols(df, exclude=[])
    X            = df[feature_cols].values

    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X)
    cumvar       = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cumvar >= PCA_VARIANCE_THRESHOLD) + 1)

    logger.info(
        f"PCA (visualization only): {n_components} components retain "
        f"{cumvar[n_components - 1] * 100:.1f}% variance "
        f"(threshold: {PCA_VARIANCE_THRESHOLD * 100:.0f}%)"
    )

    pca        = PCA(n_components=n_components, random_state=RANDOM_STATE)
    components = pca.fit_transform(X)

    comp_cols = [f"PC{i + 1}" for i in range(n_components)]
    df_pca    = pd.DataFrame(components, columns=comp_cols)
    if ID_COL in df.columns:
        df_pca.insert(0, ID_COL, df[ID_COL].values)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    joblib.dump(pca, PCA_MODEL_PATH)
    logger.info(f"PCA model saved → {PCA_MODEL_PATH} | output shape: {df_pca.shape}")

    return df_pca, pca


def run_feature_engineering() -> tuple:
    """
    Full feature engineering pipeline.

    Steps:
      1. Load scaled_data.csv
      2. Drop pre-identified multicollinear columns
      3. Drop remaining high-correlation pairs (|r| > 0.90)
      4. Save feature_reduced.csv  ← PRIMARY clustering input (NO PCA)
      5. Apply PCA for visualization only → save pca_data.csv + pca_model.pkl

    Returns:
        df_features: feature-reduced DataFrame ready for clustering
        df_pca:      PCA-projected DataFrame for visualization only
    """
    df = load_dataframe(SCALED_DATA)
    logger.info(f"Input shape: {df.shape}")

    df_features = drop_multicollinear(df)
    df_features = remove_high_correlation(df_features)

    save_dataframe(df_features, FEATURE_REDUCED_PATH)
    logger.info(f"Feature-reduced shape: {df_features.shape} | saved: {FEATURE_REDUCED_PATH}")

    df_pca, _ = apply_pca_for_visualization(df_features)
    save_dataframe(df_pca, PCA_DATA_PATH)
    logger.info(f"PCA visualization data saved | shape: {df_pca.shape} | path: {PCA_DATA_PATH}")

    return df_features, df_pca


if __name__ == "__main__":
    run_feature_engineering()
