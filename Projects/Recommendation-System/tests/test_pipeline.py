"""
============================================================
Test Suite — Online Course Recommendation System
============================================================

Purpose
-------
Validate each pipeline stage and model produces correct
output shapes, column names, and edge-case behaviour.

Run
---
    pytest tests/test_pipeline.py -v


Fixes carried forward
---------------------
FIX (brittle-row-count): test_preprocessing no longer hardcodes
  len(out) == 99_995.
FIX (dedup-composite-key-tests): Popularity and content model
  tests assert uniqueness on (course_name, instructor) pairs.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np

from src.data.data_ingestion import DataIngestion
from src.data.data_preprocessing import DataPreprocessor
from src.evaluation.recommendation_evaluator import (
    RecommendationEvaluator,
    sort_interactions_for_sequence,
)
from src.features.feature_engineering import FeatureEngineer
from src.models.hybrid_model import HybridModel
from src.models.knn_model import KNNRecommender
from src.models.user_interest_model import UserInterestModel
from src.utils.helpers import load_pickle
from config.config import (
    COURSE_FEATURES_PATH,
    USER_ITEM_MATRIX_PATH,
    HYBRID_MODEL_PATH,
    POPULARITY_MODEL_PATH,
    CONTENT_MODEL_PATH,
    COLLAB_MODEL_PATH,
)

_models_exist = (
    os.path.exists(HYBRID_MODEL_PATH)
    and os.path.exists(POPULARITY_MODEL_PATH)
    and os.path.exists(CONTENT_MODEL_PATH)
    and os.path.exists(COLLAB_MODEL_PATH)
)
_artifacts_exist = (
    os.path.exists(COURSE_FEATURES_PATH)
    and os.path.exists(USER_ITEM_MATRIX_PATH)
)

requires_models = pytest.mark.skipif(
    not _models_exist,
    reason="Trained model files not found on disk. Run 'python main.py train' first.",
)
requires_artifacts = pytest.mark.skipif(
    not _artifacts_exist,
    reason="Preprocessed artifacts not found on disk. Run 'python main.py train' first.",
)


# ==========================================================
# Data Pipeline Tests
# ==========================================================

def test_ingestion():
    """Raw dataset loads with correct shape and required columns."""

    df = DataIngestion().load()

    assert len(df) > 0
    assert "user_id"   in df.columns
    assert "course_id" in df.columns


def test_preprocessing():
    """Preprocessing drops duplicates and creates all engineered features."""

    df  = DataIngestion().load()
    pre = DataPreprocessor()
    out = pre.run(df, persist=False)

    assert "completion_rate"  in out.columns
    assert "engagement_score" in out.columns

    assert out.duplicated(subset=["user_id", "course_id"]).sum() == 0
    assert len(out) <= len(df)

    # Normalised output must be in [0, 1]
    assert out["rating"].max() <= 1.0
    assert out["course_price"].max() <= 1.0

    # feature_source_ is captured before scaling so raw values must exceed 1
    assert pre.feature_source_["rating"].max() > 1.0, (
        "Raw rating should be in original 1-5 scale before normalisation"
    )
    assert pre.feature_source_["course_price"].max() > 1.0, (
        "Raw course_price should be in original dollar scale before normalisation"
    )


@requires_artifacts
def test_feature_engineering():
    """Course content matrix has correct dimensions."""

    cf  = load_pickle(COURSE_FEATURES_PATH)
    fe  = FeatureEngineer()
    mat = fe.build_course_content_matrix(cf)

    assert mat.shape[0] == len(cf)
    assert mat.shape[1] > 0
    assert cf["rating"].max() > 1.0
    assert cf["course_price"].max() > 1.0


def test_user_interest_profiles():
    """Explicit user-interest profiles include vectors and readable summaries."""

    df = DataIngestion().load()
    pre = DataPreprocessor()
    pre.run(df, persist=False)
    feature_df = pre.feature_source_
    course_features = pre.build_course_features(feature_df)

    fe = FeatureEngineer()
    content_matrix = fe.build_course_content_matrix(course_features)

    model = UserInterestModel().fit(
        interactions_df=feature_df,
        course_features=course_features,
        content_matrix=content_matrix,
        feature_engineer=fe,
        persist=False,
    )

    uid = int(feature_df["user_id"].iloc[0])
    assert model.get_user_vector(uid) is not None
    summary = model.get_user_summary(uid)
    assert "top_keywords" in summary
    assert "top_instructors" in summary


# ==========================================================
# KNN Model Tests
# ==========================================================

@requires_artifacts
def test_knn_recommendations():
    """KNN model returns results with correct columns for a known user."""

    payload = load_pickle(USER_ITEM_MATRIX_PATH)
    model   = KNNRecommender().fit(payload, persist=False)
    uid     = payload["user_ids"][0]
    recs    = model.recommend(uid, n=5)

    assert "course_id" in recs.columns
    assert "knn_score" in recs.columns
    assert len(recs) <= 5


@requires_artifacts
def test_knn_cold_start():
    """Unknown user returns an empty DataFrame (cold-start)."""

    payload = load_pickle(USER_ITEM_MATRIX_PATH)
    model   = KNNRecommender().fit(payload, persist=False)
    recs    = model.recommend(user_id=9_999_999, n=5)

    assert recs.empty


@requires_artifacts
def test_knn_excludes_seen_courses():
    """KNN recommendations must not include courses the user has already seen."""

    payload = load_pickle(USER_ITEM_MATRIX_PATH)
    model   = KNNRecommender().fit(payload, persist=False)
    uid     = payload["user_ids"][0]
    seen    = model.get_seen_ids(uid)
    recs    = model.recommend(uid, n=10)

    if not recs.empty:
        rec_ids = set(recs["course_id"].tolist())
        assert len(rec_ids & set(seen)) == 0, "Seen course IDs must not appear in recs"


# ==========================================================
# Popularity Model Tests
# ==========================================================

@requires_models
def test_popularity_recommendations():
    """Popularity model returns n results with expected columns."""

    from src.models.popularity_model import PopularityModel

    pm   = PopularityModel.load()
    recs = pm.recommend(n=5)

    assert len(recs) == 5
    assert "course_id" in recs.columns


@requires_models
def test_popularity_filter_by_difficulty():
    """Difficulty filter returns at most n results."""

    from src.models.popularity_model import PopularityModel

    pm   = PopularityModel.load()
    recs = pm.recommend(n=5, difficulty="Beginner")

    assert len(recs) <= 5


@requires_models
def test_popularity_recommendations_have_unique_instructor_title_pairs():
    """Popularity results should not repeat the same (title, instructor) pair."""

    from src.models.popularity_model import PopularityModel

    pm   = PopularityModel.load()
    recs = pm.recommend(n=10)

    composite = recs[["course_name", "instructor"]].apply(tuple, axis=1)
    assert composite.nunique() == len(recs)


# ==========================================================
# Content-Based Model Tests
# ==========================================================

@requires_models
def test_content_recommendations():
    """Content model excludes seed course from results."""

    from src.models.content_based_model import ContentBasedModel

    cm   = ContentBasedModel.load()
    cid  = cm.course_ids[0]
    recs = cm.recommend_similar(cid, n=5)

    assert len(recs) <= 5
    assert "course_id" in recs.columns
    assert cid not in recs["course_id"].values


@requires_models
def test_content_recommendations_have_unique_instructor_title_pairs():
    """Semantic search should return distinct (course_name, instructor) pairs."""

    from src.models.content_based_model import ContentBasedModel

    cm   = ContentBasedModel.load()
    cid  = cm.course_ids[0]
    recs = cm.recommend_similar(cid, n=10)

    composite = recs[["course_name", "instructor"]].apply(tuple, axis=1)
    assert composite.nunique() == len(recs)


# ==========================================================
# Collaborative Filtering Tests
# ==========================================================

@requires_models
def test_collaborative_recommendations():
    """Collaborative model returns results for a known user."""

    from src.models.collaborative_model import CollaborativeModel

    cm   = CollaborativeModel.load()
    uid  = cm.user_ids[0]
    recs = cm.recommend(uid, n=5)

    assert "course_id" in recs.columns


@requires_models
def test_collaborative_cold_start():
    """Unknown user returns an empty DataFrame (cold-start)."""

    from src.models.collaborative_model import CollaborativeModel

    cm   = CollaborativeModel.load()
    recs = cm.recommend(user_id=9_999_999, n=5)

    assert recs.empty


# ==========================================================
# Hybrid Model Tests
# ==========================================================

@requires_models
def test_hybrid_recommendations():
    """Hybrid model returns correct columns and at most n results."""

    hm   = HybridModel.load()
    uid  = hm.collab.user_ids[0]
    recs = hm.recommend(uid, n=10)

    assert len(recs) <= 10
    assert "course_id"    in recs.columns
    assert "hybrid_score" in recs.columns
    assert "course_name"  in recs.columns

    composite = recs[["course_name", "instructor"]].apply(tuple, axis=1)
    assert composite.nunique() == len(recs)


@requires_models
def test_hybrid_cold_start():
    """Cold-start user still gets popularity-based results."""

    hm   = HybridModel.load()
    recs = hm.recommend(user_id=9_999_999, n=5)

    assert len(recs) > 0
    assert len(recs) <= 5
    assert "course_id"    in recs.columns
    assert "hybrid_score" in recs.columns


@requires_models
def test_hybrid_content_arm_active():
    """Content arm produces a non-None profile for a known user."""

    hm      = HybridModel.load()
    uid     = hm.collab.user_ids[0]
    profile = hm._build_user_profile(uid)

    assert profile is not None
    assert profile.shape[0] == hm.content.content_matrix.shape[1]


# ==========================================================
# Unit / Logic Tests (no disk required)
# ==========================================================

def test_sequence_sorting_uses_progression_proxy():
    """Sequence ordering should prioritize learner progression fields."""

    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1],
            "course_id": [40, 20, 30, 10],
            "previous_courses_taken": [2, 0, 1, 0],
            "difficulty_level": [2, 0, 1, 0],
            "enrollment_numbers": [100, 50, 75, 80],
        }
    )

    ordered = sort_interactions_for_sequence(df, log=False)

    assert ordered["course_id"].tolist() == [10, 20, 30, 40]


def test_holdout_uses_last_pseudo_chronological_interaction():
    """Evaluation holdout should align with next-item sequence modeling."""

    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2],
            "course_id": [300, 100, 200, 400],
            "course_name": ["c3", "c1", "c2", "other"],
            "difficulty_level": [2, 0, 1, 0],
            "course_price": [300, 100, 200, 50],
            "previous_courses_taken": [2, 0, 1, 0],
            "enrollment_numbers": [100, 300, 200, 10],
        }
    )

    evaluator = RecommendationEvaluator(max_users=1, random_state=42)
    train_df, test_df = evaluator._split_holdout(df)

    assert test_df["course_id"].tolist() == [300]
    assert sorted(train_df[train_df["user_id"] == 1]["course_id"].tolist()) == [100, 200]


def test_hybrid_active_weights_renormalize_over_available_arms():
    """Hybrid weights should renormalize cleanly over the arms that exist."""

    hm = HybridModel()
    weights = hm._resolve_active_weights(
        history_size=10,
        active_arms={"knn", "collaborative"},
    )

    assert set(weights) == {"knn", "collaborative"}
    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
