"""
src/data/data_preprocessing.py
World Development Clustering - Pipeline Stage 2: Data Preprocessing

Preprocessing order (mandatory):
  1. Drop columns with > 40% missing values
  2. DROP rows where missing ratio > 0.5 (before imputation)
  3. Cap outliers (IQR x 3.0)
  4. SimpleImputer(strategy='median')   <- KNNImputer REMOVED
  5. log1p transform on GDP, Tourism Inbound, Tourism Outbound
  6. StandardScaler

Why this order:
  - Drop high-missing cols first: reduces dimensionality before imputation
  - Drop high-missing rows: rows >50% missing cannot be reliably imputed
  - Cap before imputation: bounds give better median estimates
  - Impute before transforms: log1p / StandardScaler require complete data
  - log1p before StandardScaler: normalise skewed distributions first
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

import numpy as np
import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config.config import (
    CLEANED_DATA,
    SCALED_DATA,
    PROCESSED_DIR,
    IMPUTER_PATH,
    SCALER_PATH,
    HIGH_MISSING_COL_THRESHOLD,
    HIGH_MISSING_ROW_THRESHOLD,
    IQR_CAP_MULTIPLIER,
    LOG1P_COLS,
    ID_COL,
)
from src.utils.logger import get_logger
from src.utils.helpers import save_dataframe, load_dataframe, get_numeric_feature_cols

logger = get_logger(__name__)

# -- Step 1: Drop high-missing columns ----------------------------------------

def drop_high_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with > HIGH_MISSING_COL_THRESHOLD (40%) missing values.
    Applied post-aggregation on the cleaned cross-sectional data.
    Columns that are >40% null across 208 countries cannot be reliably
    imputed and add noise to the clustering pipeline.
    """
    feature_cols  = get_numeric_feature_cols(df)
    missing_fracs = df[feature_cols].isnull().mean()
    cols_to_drop  = missing_fracs[missing_fracs > HIGH_MISSING_COL_THRESHOLD].index.tolist()

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped {len(cols_to_drop)} high-missing columns (>{HIGH_MISSING_COL_THRESHOLD:.0%}): {cols_to_drop} | shape: {df.shape}")
    else:
        logger.info(f"No columns exceed {HIGH_MISSING_COL_THRESHOLD:.0%} missing threshold")

    return df

# -- Step 2: Drop high-missing rows BEFORE imputation -------------------------

def drop_high_missing_rows(
    df: pd.DataFrame,
    threshold: float = HIGH_MISSING_ROW_THRESHOLD,
) -> pd.DataFrame:
    """
    Drop rows (countries) where missing ratio > threshold (0.5).
    Rows with >50% missing features cannot be reliably imputed and
    introduce noise into the clustering pipeline. Drop BEFORE imputation
    so the imputer's median estimates are not skewed by sparse rows.
    """
    feature_cols      = get_numeric_feature_cols(df, exclude=[])
    missing_fractions = df[feature_cols].isnull().mean(axis=1)
    drop_mask         = missing_fractions > threshold
    n_drop            = int(drop_mask.sum())

    if n_drop > 0:
        dropped_countries = df.loc[drop_mask, ID_COL].tolist() if ID_COL in df.columns else []
        logger.warning(f"Dropping {n_drop} rows with >{threshold:.0%} missing features | countries: {dropped_countries} | shape: {df[~drop_mask].shape}")
        df = df[~drop_mask].reset_index(drop=True)
    else:
        logger.info(f"No rows exceed {threshold:.0%} missing threshold")

    return df

# -- Step 3: IQR-based outlier capping ----------------------------------------

def cap_outliers_iqr(df: pd.DataFrame, multiplier: float = IQR_CAP_MULTIPLIER) -> pd.DataFrame:
    """
    Cap extreme values at Q1 - multiplier*IQR and Q3 + multiplier*IQR.
    Applied before imputation so median estimates are not distorted by extremes.
    """
    df           = df.copy()
    feature_cols = get_numeric_feature_cols(df)
    total_capped = 0

    for col in feature_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        q1  = series.quantile(0.25)
        q3  = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        n_cap = int((df[col] < lower).sum()) + int((df[col] > upper).sum())
        if n_cap > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            total_capped += n_cap

    logger.info(f"Outlier capping applied to {len(feature_cols)} features (multiplier={multiplier}) | total capped: {total_capped}")
    return df

# -- Step 4: SimpleImputer (median) -------------------------------------------

def impute_missing(df: pd.DataFrame) -> tuple:
    """
    Impute missing values using SimpleImputer(strategy='median').
    KNNImputer has been REMOVED. SimpleImputer(median) is used because:
    - It is robust to outliers (median, not mean).
    - It does not require pairwise distance computation -- safe for sparse data.
    - Consistent and reproducible: same result every run.
    Saves the fitted imputer artifact for deployment inference.
    """
    feature_cols = get_numeric_feature_cols(df)
    df           = df.copy()

    n_before             = int(df[feature_cols].isnull().sum().sum())
    imputer              = SimpleImputer(strategy="median")
    df[feature_cols]     = imputer.fit_transform(df[feature_cols])
    n_after              = int(df[feature_cols].isnull().sum().sum())

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    joblib.dump(imputer, IMPUTER_PATH)

    logger.info(f"Imputation complete | missing: {n_before} -> {n_after} | saved: {IMPUTER_PATH}")
    return df, imputer

# -- Step 5: log1p transform --------------------------------------------------

def apply_log1p_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply np.log1p to highly right-skewed columns: GDP, Tourism Inbound,
    Tourism Outbound.
    Why log1p (not PowerTransformer):
    - These specific columns span many orders of magnitude (GDP: $1B -> $17T).
    - log1p is interpretable, invertible, and standard for economic variables.
    - Applied AFTER imputation so no NaN values remain.
    - Applied BEFORE StandardScaler so scaler sees the transformed distribution.
    """
    df = df.copy()

    for col in LOG1P_COLS:
        if col in df.columns:
            skew_before = df[col].skew()
            df[col]     = np.log1p(df[col].clip(lower=0))
            skew_after  = df[col].skew()
            logger.info(f"log1p({col}): skew {skew_before:+.3f} -> {skew_after:+.3f}")
        else:
            logger.warning(f"log1p: column '{col}' not found - skipping")

    return df

# -- Step 6: StandardScaler ---------------------------------------------------

def scale_features(df: pd.DataFrame) -> tuple:
    """
    StandardScaler -- zero mean, unit variance.
    Applied as the final step after imputation and log1p transform.
    Saves the fitted scaler for deployment inference.
    """
    feature_cols     = get_numeric_feature_cols(df)
    df               = df.copy()
    scaler           = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"StandardScaler applied to {len(feature_cols)} features | saved: {SCALER_PATH}")
    
    return df, scaler

# -- Orchestrator -------------------------------------------------------------

def run_preprocessing() -> pd.DataFrame:
    """
    Full preprocessing pipeline.
    Mandatory order:
      1. Load cleaned_data.csv (output of ingestion)
      2. Drop columns with > 40% missing
      3. Drop rows with > 50% missing (BEFORE imputation)
      4. Cap outliers (IQR x 3.0)
      5. SimpleImputer(strategy='median')
      6. log1p transform (GDP, Tourism Inbound, Tourism Outbound)
      7. StandardScaler
      8. Save scaled_data.csv + 2 .pkl artifacts
    
    Returns:
        df_scaled: fully preprocessed DataFrame
    """
    df = load_dataframe(CLEANED_DATA)
    logger.info(f"Input shape: {df.shape}")

    df       = drop_high_missing_columns(df)
    df       = drop_high_missing_rows(df, threshold=HIGH_MISSING_ROW_THRESHOLD)
    df       = cap_outliers_iqr(df, multiplier=IQR_CAP_MULTIPLIER)
    df, _    = impute_missing(df)
    df       = apply_log1p_transform(df)
    df, _    = scale_features(df)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    save_dataframe(df, SCALED_DATA)
    logger.info(f"Output shape: {df.shape} | saved: {SCALED_DATA}")
    
    return df

if __name__ == "__main__":
    run_preprocessing()