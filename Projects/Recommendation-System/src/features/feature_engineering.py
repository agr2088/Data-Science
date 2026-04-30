"""
============================================================
Feature Engineering Module (Semantic Embeddings Edition)
============================================================

Purpose
-------
Build the numerical representations of courses and users
that power the content-based and hybrid recommendation arms.

"""

import contextlib
import io
import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, normalize

from config.config import NUMERIC_FEATURE_COLS, SEMANTIC_EMBEDDING_MODEL
from src.utils.helpers import timer
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.utils.loading_report").setLevel(logging.ERROR)

_LOCAL_ENCODER_FALLBACK_EXCEPTIONS = (
    TypeError,
    OSError,
    ValueError,
    RuntimeError,
)


class FeatureEngineer:
    def __init__(self):
        self.encoder = self._load_encoder()
        self.numeric_scaler = MinMaxScaler()
        self.course_matrix = None
        self.course_ids = None

    def _load_encoder(self) -> "SentenceTransformer":
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "sentence_transformers is required to build semantic course features. "
                "Install project dependencies before running feature engineering."
            ) from exc

        logger.info(
            f"Semantic encoder | model={SEMANTIC_EMBEDDING_MODEL} | strategy=local-cache-first"
        )

        quiet_out = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(quiet_out), contextlib.redirect_stderr(quiet_out):
                try:
                    encoder = SentenceTransformer(
                        SEMANTIC_EMBEDDING_MODEL,
                        local_files_only=True,
                    )
                    logger.info("Semantic encoder ready | source=local cache")
                    return encoder
                except _LOCAL_ENCODER_FALLBACK_EXCEPTIONS:
                    pass

        logger.info("Semantic encoder cache miss | fetching model artifacts")
        quiet_out = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(quiet_out), contextlib.redirect_stderr(quiet_out):
                encoder = SentenceTransformer(SEMANTIC_EMBEDDING_MODEL)

        logger.info("Semantic encoder ready | source=Hugging Face Hub")
        return encoder

    @timer
    def build_course_content_matrix(self, course_features: pd.DataFrame) -> np.ndarray:
        logger.info(
            f"Building multimodal course representation | courses={len(course_features):,}"
        )

        text_features = self._build_semantic_features(course_features)
        num_features = self._build_numeric_features(course_features)

        combined = np.hstack([text_features, num_features])
        combined = normalize(combined, norm="l2")

        self.course_matrix = combined
        self.course_ids = course_features["course_id"].values

        logger.info(
            f"Course content matrix ready | shape={combined.shape[0]:,} x {combined.shape[1]}"
        )
        return combined

    @timer
    def build_user_profiles(
        self,
        processed_df: pd.DataFrame,
        course_features: pd.DataFrame,
    ) -> pd.DataFrame:
        logger.info("Building semantic user profiles from observed course histories")
        if self.course_matrix is None:
            self.build_course_content_matrix(course_features)

        cid_to_idx = {cid: i for i, cid in enumerate(self.course_ids)}
        user_profiles = {}

        for uid, group in processed_df.groupby("user_id"):
            idxs    = []
            weights = []
            for cid, rating in zip(group["course_id"].values, group["rating"].values):
                idx = cid_to_idx.get(cid)
                if idx is None:
                    continue
                idxs.append(idx)
                weights.append(float(rating))

            if not idxs:
                continue

            vecs = self.course_matrix[idxs]
            w    = np.asarray(weights, dtype=np.float64)
            if w.sum() == 0:
                w = np.ones(len(idxs), dtype=np.float64)

            user_profiles[uid] = np.average(vecs, axis=0, weights=w)

        profile_df = pd.DataFrame.from_dict(
            user_profiles,
            orient="index",
            columns=[f"dim_{i}" for i in range(self.course_matrix.shape[1])],
        )
        profile_df.index.name = "user_id"

        logger.info(
            f"Semantic user profile matrix ready | users={len(profile_df):,} | dims={self.course_matrix.shape[1]}"
        )
        return profile_df

    def _build_semantic_features(self, cf: pd.DataFrame) -> np.ndarray:
        difficulty_map = {0: "beginner", 1: "intermediate", 2: "advanced"}
        if cf["difficulty_level"].dtype in ["int64", "int32", "float64"]:
            difficulty_tokens = cf["difficulty_level"].map(difficulty_map).fillna("unknown")
        else:
            difficulty_tokens = cf["difficulty_level"].str.lower().fillna("unknown")

        if "price_tier" in cf.columns:
            price_tokens = cf["price_tier"].astype(str).str.lower().replace("nan", "standard")
        else:
            price_tokens = pd.Series(["standard"] * len(cf))

        cert_map = {1: "offers a certification", 0: "does not offer certification"}
        if cf["certification_offered"].dtype in ["int64", "int32", "float64"]:
            cert_tokens = cf["certification_offered"].map(cert_map).fillna("")
        else:
            cert_tokens = cf["certification_offered"].str.lower().fillna("")

        # FIX (semantic-diversity): Add study_material tokens
        mat_map = {1: "includes study materials", 0: "does not include study materials"}
        if "study_material_available" in cf.columns:
            if cf["study_material_available"].dtype in ["int64", "int32", "float64"]:
                mat_tokens = cf["study_material_available"].map(mat_map).fillna("")
            else:
                mat_tokens = cf["study_material_available"].str.lower().fillna("")
        else:
            mat_tokens = pd.Series([""] * len(cf))

        # FIX (semantic-diversity): Enrollment tier token
        if "enrollment_numbers" in cf.columns:
            enroll_q33 = cf["enrollment_numbers"].quantile(0.33)
            enroll_q66 = cf["enrollment_numbers"].quantile(0.66)

            def _enroll_tier(v):
                if v <= enroll_q33:
                    return "low enrollment"
                elif v <= enroll_q66:
                    return "medium enrollment"
                return "high enrollment"

            enroll_tokens = cf["enrollment_numbers"].apply(_enroll_tier)
        else:
            enroll_tokens = pd.Series([""] * len(cf))

        # FIX (semantic-diversity): Price range token for numeric differentiation
        if "course_price" in cf.columns:
            price_q33 = cf["course_price"].quantile(0.33)
            price_q66 = cf["course_price"].quantile(0.66)

            def _price_range(v):
                if v <= price_q33:
                    return "budget priced"
                elif v <= price_q66:
                    return "moderately priced"
                return "premium priced"

            price_range_tokens = cf["course_price"].apply(_price_range)
        else:
            price_range_tokens = pd.Series([""] * len(cf))

        # FIX (semantic-diversity): Richer corpus — "Taught by instructor X"
        # emphasises the instructor more strongly than the original "Instructor: X"
        # phrasing, helping MiniLM distinguish same-title courses by teacher.
        corpus = (
            "Course Title: " + cf["course_name"].fillna("Unknown Title") + ". "
            + "Taught by instructor " + cf["instructor"].fillna("Unknown Instructor") + ". "
            + "Difficulty level: " + difficulty_tokens + ". "
            + "Price tier: " + price_tokens + ". "
            + "Price range: " + price_range_tokens + ". "
            + "Certification: " + cert_tokens + ". "
            + "Study materials: " + mat_tokens + ". "
            + "Popularity: " + enroll_tokens + "."
        ).tolist()

        logger.info(
            f"Encoding course narratives into dense semantic vectors | items={len(corpus):,}"
        )
        return self.encoder.encode(corpus, show_progress_bar=False)

    def _build_numeric_features(self, cf: pd.DataFrame) -> np.ndarray:
        cols = [c for c in NUMERIC_FEATURE_COLS if c in cf.columns]
        numeric = cf[cols].fillna(0).values.astype(float)
        if numeric.size == 0:
            return numeric
        logger.info(f"Scaling structured course signals | features={len(cols)}")
        return self.numeric_scaler.fit_transform(numeric)