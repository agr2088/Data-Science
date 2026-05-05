"""
src/data/data_ingestion.py
World Development Clustering - Pipeline Stage 1: Data Ingestion

Responsibilities:
  - Load raw Excel (208 countries * 13 years = 2704 rows)
  - Parse currency and percent string columns to float
  - Drop unusable columns (93%+ null, constant)
  - Assign year (2000-2012) from row order within each country group
  - Compute trend features on panel data BEFORE aggregation     <-- Fix 1
  - Aggregate panel -> cross-sectional (mean per country)
  - Merge trend slopes into aggregated table
  - Standardize country names for choropleth map compatibility  <-- Fix 9
  - Save cleaned_data.csv

Output: 208 rows * ~27 columns (22 mean features + 5 trend slopes)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

import numpy as np
import pandas as pd
from scipy import stats

from config.config import (
    RAW_FILE,
    SHEET_NAME,
    CLEANED_DATA,
    PROCESSED_DIR,
    CURRENCY_COLS,
    PERCENT_COLS,
    DROP_COLS,
    ID_COL,
    TREND_COLS,
    COUNTRY_NAME_MAP,
)
from src.utils.logger import get_logger
from src.utils.helpers import strip_currency, strip_percent, save_dataframe

logger = get_logger(__name__)


# -- Stage functions ----------------------------------------------------------

def load_raw() -> pd.DataFrame:
    """Load raw Excel file from config path."""
    if not os.path.exists(RAW_FILE):
        logger.error(f"Raw data file not found: {RAW_FILE}")
        raise FileNotFoundError(
            f"Raw data file not found: {RAW_FILE}\n"
            f"Place the Excel file at: {RAW_FILE}"
        )
    df = pd.read_excel(RAW_FILE, sheet_name=SHEET_NAME)
    logger.info(f"Raw data loaded | shape: {df.shape} | countries: {df[ID_COL].nunique()}")
    return df


def fix_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse string-encoded currency and percent columns to float.
    Only touches columns in CURRENCY_COLS and PERCENT_COLS (from config).
    """
    df = df.copy()

    for col in CURRENCY_COLS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].apply(strip_currency)

    for col in PERCENT_COLS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].apply(strip_percent)

    logger.info(f"Column types fixed | currency: {len(CURRENCY_COLS)} | percent: {len(PERCENT_COLS)}")
    return df


def drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns defined in DROP_COLS (config):
      - Ease of Business: 93.2% null -> unusable for any model
      - Number of Records: constant = 1 -> zero information
    """
    cols_present = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_present)
    logger.info(f"Dropped unusable columns: {cols_present}")
    return df


def assign_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    The dataset has no explicit Year column.
    Each country appears exactly 13 times (one per year, 2000-2012).
    Assign Year = cumulative row index within each country group + 2000.
    Required by compute_trend_features (Fix 1).
    Dropped after trend computation -> must not appear in aggregated output.
    """
    df = df.copy()
    df["Year"] = df.groupby(ID_COL).cumcount() + 2000
    return df


def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix 1 - Trend Features Computed BEFORE Aggregation.
    
    WHY before aggregation:
    - After mean aggregation all temporal information is lost.
    - Two countries with identical means but opposite trajectories
      (one growing, one declining) look identical to the clusterer.
    - GDP_trend, Internet_trend, etc. capture whether a country was on
      an ascending or descending development path 2000-2012.
      
    Method:
    - OLS linear regression of each TREND_COL against Year (2000-2012)
    - Minimum 3 non-null observations required for a reliable slope
    - Countries with fewer than 3 obs get NaN (handled later by KNN imputer)
    
    Columns produced: '{col}_trend' for each col in TREND_COLS (config)
    Returns:
        df_trends: 208 rows * (1 + len(TREND_COLS)) DataFrame
    """
    records = []
    
    for country, grp in df.groupby(ID_COL):
        row   = {ID_COL: country}
        years = grp["Year"].values.astype(float)
        
        for col in TREND_COLS:
            if col not in grp.columns:
                row[f"{col}_trend"] = np.nan
                continue
                
            vals = grp[col].values.astype(float)
            mask = ~np.isnan(vals)
            
            if mask.sum() >= 3:
                slope, _, _, _, _ = stats.linregress(years[mask], vals[mask])
                row[f"{col}_trend"] = round(float(slope), 6)
            else:
                row[f"{col}_trend"] = np.nan
                
        records.append(row)

    df_trends = pd.DataFrame(records)
    
    trend_feature_cols = [c for c in df_trends.columns if c != ID_COL]
    nulls = {col: int(df_trends[col].isnull().sum()) for col in trend_feature_cols if df_trends[col].isnull().any()}
    
    if nulls:
        logger.warning(f"Trend features with nulls (will be imputed): {nulls}")
        
    logger.info(f"Trend features computed: {len(trend_feature_cols)} | shape: {df_trends.shape}")
    return df_trends


def aggregate_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse panel data (2704 rows) to cross-sectional (208 rows) by taking
    the mean of each numeric column per country.
    Standard practice for development indicator clustering -> preserves the
    central tendency across 13 years without over-weighting any single year.
    'Year' column is dropped here (mean of years is not meaningful).
    """
    df_agg = df.groupby(ID_COL).mean(numeric_only=True).reset_index()
    if "Year" in df_agg.columns:
        df_agg = df_agg.drop(columns=["Year"])
    logger.info(f"Panel aggregated to cross-sectional | shape: {df_agg.shape}")
    return df_agg


def merge_trend_features(df_agg: pd.DataFrame, df_trends: pd.DataFrame) -> pd.DataFrame:
    """
    Join trend slopes (computed on panel) into the aggregated cross-sectional table.
    Trend features cannot be aggregated via mean -> they must be joined separately.
    """
    df_merged  = df_agg.merge(df_trends, on=ID_COL, how="left")
    trend_cols = [c for c in df_trends.columns if c != ID_COL]
    logger.info(f"Trend features merged: {len(trend_cols)} | shape: {df_merged.shape}")
    return df_merged


def standardize_country_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix 9 - Standardize Country Names for Plotly Choropleth Map.
    Plotly's locationmode='country names' uses ISO-standard country names.
    The World Bank dataset uses non-standard names for ~30 countries.
    Without this fix those countries render as grey/unmapped on the world map.
    COUNTRY_NAME_MAP is defined in config.py -> no hardcoding here.
    Logs every rename for audit trail.
    """
    df      = df.copy()
    renamed = 0
    
    for original, standardized in COUNTRY_NAME_MAP.items():
        mask = df[ID_COL] == original
        if mask.any():
            df.loc[mask, ID_COL] = standardized
            renamed += 1
            
    logger.info(f"Country names standardized: {renamed} renamed")
    return df


def run_ingestion() -> pd.DataFrame:
    """
    Full ingestion pipeline - runs all stages in correct order.
    
    Stage order:
      1. Load raw Excel
      2. Parse string columns (currency / percent)
      3. Drop unusable columns
      4. Assign Year (2000-2012)
      5. Compute trend features on panel    <-- Fix 1 (BEFORE aggregation)
      6. Aggregate to country level (mean)
      7. Merge trend features
      8. Standardize country names          <-- Fix 9
      9. Save cleaned_data.csv
      
    Returns:
        df_cleaned: 208 rows * ~27 columns
    """
    df         = load_raw()
    df         = fix_column_types(df)
    df         = drop_unusable_columns(df)
    df         = assign_years(df)
    df_trends  = compute_trend_features(df)
    df_agg     = aggregate_by_country(df)
    df_cleaned = merge_trend_features(df_agg, df_trends)
    df_cleaned = standardize_country_names(df_cleaned)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    save_dataframe(df_cleaned, CLEANED_DATA)

    logger.info(f"Ingestion complete | output shape: {df_cleaned.shape} | saved: {CLEANED_DATA}")
    return df_cleaned


if __name__ == "__main__":
    run_ingestion()