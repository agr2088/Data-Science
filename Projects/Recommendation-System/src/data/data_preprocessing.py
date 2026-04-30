"""
============================================================
Data Preprocessing Module
============================================================

Purpose
-------
Transform the raw dataset into a clean, model-ready format
and build the sparse interaction matrix and course features.

"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler

from config.config import (
    PROCESSED_DATA_PATH, USER_ITEM_MATRIX_PATH, COURSE_FEATURES_PATH,
    BINARY_COLUMNS, DIFFICULTY_ORDER, FLOAT_BOUNDS,
    SCALE_COLUMNS, ENGAGEMENT_WEIGHTS, SCALER_PATH,
    PRICE_TIER_BINS, PRICE_TIER_LABELS,
)
from src.utils.logger import get_logger
from src.utils.helpers import (
    timer, save_pickle,
    encode_binary, encode_ordinal, clip_outliers,
    compute_completion_rate, summarize_df,
)

logger = get_logger(__name__)


class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_source_ = None

    @timer
    def run(self, df: pd.DataFrame, persist: bool = True) -> pd.DataFrame:
        logger.info("Preparing model-ready interaction data")

        # FIX 1: Map duplicated IDs to a single Canonical ID before any processing
        df = self._apply_canonical_ids(df)
        
        df = self._drop_duplicates(df)
        
        # FIX 3: Increase sparsity threshold to 3 interactions to allow collaborative models to learn
        df = self._filter_sparse_interactions(df, min_interactions=3)
        
        df = self._clip_outliers(df)
        df = self._encode_categoricals(df)
        df = self._engineer_features(df)

        # Snapshot pre-normalisation — used to build the user-item matrix
        # and course feature store so they contain un-scaled engagement values.
        feature_source = df.copy()
        self.feature_source_ = feature_source.copy()

        df = self._normalise(df, persist=persist)

        if persist:
            summarize_df(df, "processed")
            df.to_csv(PROCESSED_DATA_PATH, index=False)
            logger.info(f"Saved processed CSV to {PROCESSED_DATA_PATH}")
            self._build_user_item_matrix(feature_source, persist=True)
            self._build_course_features(feature_source, persist=True)

        return df

    def _apply_canonical_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps arbitrary course IDs to a single Canonical ID based on the course name
        and instructor, resolving the duplicate ID issue that shattered the sparse matrix.
        """
        logger.info("Applying Canonical ID mapping to collapse redundant courses")
        
        df['canonical_course_id'] = df.groupby(['course_name', 'instructor']).ngroup()
        df = df.drop(columns=['course_id']).rename(columns={'canonical_course_id': 'course_id'})

        agg_dict = {}
        for col in df.columns:
            if col in ['user_id', 'course_id']:
                continue
            elif col == 'time_spent_hours':
                agg_dict[col] = 'sum'
            elif col in ['feedback_score', 'completion_rate', 'engagement_score', 'enrollment_numbers']:
                agg_dict[col] = 'max'
            elif col == 'rating':
                agg_dict[col] = 'mean'
            else:
                agg_dict[col] = 'first' 

        df = df.groupby(['user_id', 'course_id']).agg(agg_dict).reset_index()
        logger.info(f"Canonical mapping reduced catalogue to {df['course_id'].nunique()} distinct courses")
        return df

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(
            subset=["user_id", "course_id"], keep="first"
        ).reset_index(drop=True)
        logger.info(f"Deduplicated user-course interactions | removed={before - len(df)}")
        return df

    def _filter_sparse_interactions(self, df: pd.DataFrame, min_interactions: int = 1) -> pd.DataFrame:
        if min_interactions <= 1:
            logger.info("Sparsity filter | bypassed to preserve full collaborative signal")
            return df

        logger.info(f"Filtering users and courses with fewer than {min_interactions} interactions...")
        original_df  = df.copy()
        original_len = len(df)
        before_len   = len(df)
        iteration    = 1

        while True:
            user_counts   = df["user_id"].value_counts()
            valid_users   = user_counts[user_counts >= min_interactions].index
            df            = df[df["user_id"].isin(valid_users)]

            course_counts  = df["course_id"].value_counts()
            valid_courses  = course_counts[course_counts >= min_interactions].index
            df             = df[df["course_id"].isin(valid_courses)]

            if len(df) == before_len:
                break

            before_len = len(df)
            iteration += 1

        if len(df) < (original_len * 0.8):
            logger.warning(
                "Sparsity filter removed more than 20% of data. Aborting filter."
            )
            return original_df

        logger.info(
            f"Sparsity reduction complete in {iteration} passes. "
            f"New dataset size: {len(df)} interactions."
        )
        return df.reset_index(drop=True)

    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, (lo, hi) in FLOAT_BOUNDS.items():
            df = clip_outliers(df, col, lo, hi)
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = encode_binary(df, BINARY_COLUMNS)
        df = encode_ordinal(df, "difficulty_level", DIFFICULTY_ORDER)
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["completion_rate"] = compute_completion_rate(df)

        w = ENGAGEMENT_WEIGHTS
        df["engagement_score"] = (
            w["rating"]          * (df["rating"] / 5.0) +
            w["feedback_score"]  * df["feedback_score"] +
            w["completion_rate"] * df["completion_rate"]
        ).round(4)

        m = df["enrollment_numbers"].quantile(0.25)
        C = df["rating"].mean()
        v = df["enrollment_numbers"]
        df["weighted_score"] = (
            (v / (v + m)) * df["rating"] + (m / (v + m)) * C
        ).round(4)

        df["price_tier"] = pd.cut(
            df["course_price"],
            bins=PRICE_TIER_BINS,
            labels=PRICE_TIER_LABELS,
        ).astype(str)

        logger.info("Feature engineering complete | added completion_rate, engagement_score, price_tier")
        return df

    def _normalise(self, df: pd.DataFrame, persist: bool = True) -> pd.DataFrame:
        if len(df) == 0:
            logger.error("Dataset has 0 rows before normalization.")
            return df

        cols = [c for c in SCALE_COLUMNS if c in df.columns]
        df[cols] = self.scaler.fit_transform(df[cols])
        logger.info(f"Scaled continuous features with MinMax normalization | columns={len(cols)}")

        if persist:
            save_pickle(self.scaler, SCALER_PATH)
            logger.info(f"Saved fitted scaler to {SCALER_PATH}")

        return df

    def _build_user_item_matrix(self, df: pd.DataFrame, persist: bool = True) -> dict:
        logger.info("Building sparse user-item matrix for collaborative learning")

        user_ids   = df["user_id"].unique()
        course_ids = df["course_id"].unique()
        user_idx   = {u: i for i, u in enumerate(user_ids)}
        course_idx = {c: i for i, c in enumerate(course_ids)}

        rows = df["user_id"].map(user_idx).values
        cols = df["course_id"].map(course_idx).values
        data = df["engagement_score"].values.astype(np.float32)

        matrix = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(user_ids), len(course_ids)),
            dtype=np.float32,
        )

        payload = {
            "matrix":     matrix,
            "user_ids":   user_ids,
            "course_ids": course_ids,
            "user_idx":   user_idx,
            "course_idx": course_idx,
        }

        if persist:
            save_pickle(payload, USER_ITEM_MATRIX_PATH)
            logger.info(
                f"User-item matrix ready | shape={matrix.shape[0]:,} x {matrix.shape[1]:,} | nnz={matrix.nnz:,}"
            )

        return payload

    def _build_course_features(self, df: pd.DataFrame, persist: bool = True) -> pd.DataFrame:
        logger.info("Aggregating course-level feature store")

        def _mode_first(s):
            m = s.mode()
            return m.iloc[0] if not m.empty else s.iloc[0]

        agg = {
            "course_name":              "first",
            "instructor":               _mode_first,
            "course_duration_hours":    "mean",
            "certification_offered":    "first",
            "difficulty_level":         "first",
            "rating":                   "mean",
            "enrollment_numbers":       "mean",
            "course_price":             "mean",
            "feedback_score":           "mean",
            "study_material_available": "first",
            "time_spent_hours":         "mean",
            "previous_courses_taken":   "mean",
            "completion_rate":          "mean",
            "engagement_score":         "mean",
            "weighted_score":           "mean",
        }

        cf = df.groupby("course_id").agg(agg).reset_index()

        if persist:
            save_pickle(cf, COURSE_FEATURES_PATH)
            logger.info(
                f"Course feature store ready | courses={cf.shape[0]:,} | features={cf.shape[1]}"
            )

        return cf

    def build_user_item_payload(self, df: pd.DataFrame) -> dict:
        return self._build_user_item_matrix(df, persist=False)

    def build_course_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._build_course_features(df, persist=False)