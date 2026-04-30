"""
============================================================
Data Ingestion Module
============================================================

Purpose
-------
Load the raw online course dataset from Excel and perform
basic integrity checks before passing it downstream.

Responsibilities
----------------
- Locate and load the raw Excel file
- Validate required columns are present
- Assert basic data quality constraints
- Return a clean DataFrame ready for preprocessing

Output
------
pandas.DataFrame with all 14 raw columns intact.

"""

import pandas as pd
from pathlib import Path
from pandas.api.types import is_integer_dtype

from config.config import RAW_DATA_PATH, EXPECTED_COLUMNS
from src.utils.logger import get_logger
from src.utils.helpers import timer, summarize_df

logger = get_logger(__name__)


# ==========================================================
# Data Ingestion Class
# ==========================================================

class DataIngestion:
    """
    Loads the raw Excel dataset, performs minimal integrity
    checks, and returns a clean DataFrame ready for preprocessing.
    """

    def __init__(self, path: Path = RAW_DATA_PATH):
        self.path = Path(path)

    @timer
    def load(self) -> pd.DataFrame:
        """
        Load dataset from Excel and validate its schema.

        Returns
        -------
        pandas.DataFrame
            Raw dataset with all original columns.
        """

        logger.info(f"Source dataset | {self.path}")

        if not self.path.exists():
            raise FileNotFoundError(
                f"Raw data not found at: {self.path}"
            )

        df = pd.read_excel(self.path, engine="openpyxl")
        logger.info(f"Raw dataset loaded | rows={len(df):,} | columns={len(df.columns)}")

        self._validate(df)
        summarize_df(df, label="raw")

        return df

    def _validate(self, df: pd.DataFrame) -> None:
        """
        Validate required columns are present and key columns
        fall within expected value ranges.

        Raises
        ------
        ValueError if schema or range checks fail.
        """

        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")

        # FIX (validate-raises): Use explicit ValueError instead of assert
        # so validation is never silently stripped in optimized (-O) runs.
        if not is_integer_dtype(df["user_id"]):
            raise ValueError(
                f"user_id must be integer, got {df['user_id'].dtype}"
            )
        if df["user_id"].isna().any():
            raise ValueError("user_id contains missing values")

        if not is_integer_dtype(df["course_id"]):
            raise ValueError(
                f"course_id must be integer, got {df['course_id'].dtype}"
            )
        if df["course_id"].isna().any():
            raise ValueError("course_id contains missing values")

        if not df["rating"].between(1.0, 5.0).all():
            bad = df.loc[~df["rating"].between(1.0, 5.0), "rating"]
            raise ValueError(
                f"rating must be in range [1, 5]. "
                f"Found {len(bad)} out-of-range values: {bad.unique()[:5]}"
            )

        if not df["feedback_score"].between(0.0, 1.0).all():
            bad = df.loc[~df["feedback_score"].between(0.0, 1.0), "feedback_score"]
            raise ValueError(
                f"feedback_score must be in range [0, 1]. "
                f"Found {len(bad)} out-of-range values: {bad.unique()[:5]}"
            )

        logger.info("Schema validation passed | required columns and numeric ranges verified")
