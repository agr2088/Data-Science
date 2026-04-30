"""
============================================================
Training Pipeline
============================================================

Purpose
-------
Orchestrate the complete end-to-end training workflow for the
Online Course Recommendation System.

"""

import os
import sys
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from config.config import COURSE_FEATURES_PATH, USER_ITEM_MATRIX_PATH
from src.data.data_ingestion import DataIngestion
from src.data.data_preprocessing import DataPreprocessor
from src.evaluation.recommendation_evaluator import RecommendationEvaluator
from src.features.feature_engineering import FeatureEngineer
from src.models.collaborative_model import CollaborativeModel
from src.models.content_based_model import ContentBasedModel
from src.models.hybrid_model import HybridModel
from src.models.knn_model import KNNRecommender
from src.models.popularity_model import PopularityModel
from src.models.user_interest_model import UserInterestModel
from src.utils.helpers import (
    format_duration,
    load_pickle,
    log_banner,
    log_stage,
    log_stage_result,
    timer,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """Execute all training stages and return the fitted hybrid model."""

    _TOTAL_STAGES = 6

    @timer
    def run(self) -> HybridModel:
        pipeline_start = time.perf_counter()
        log_banner(
            "ONLINE COURSE RECOMMENDATION TRAINING",
            "Objective | build a hybrid recommender from learner history, course semantics, and engagement signals",
        )

        stage_start = time.perf_counter()
        log_stage(1, self._TOTAL_STAGES, "Data Ingestion", "Load the raw interaction dataset and validate schema integrity")
        df_raw = DataIngestion().load()
        log_stage_result(
            "Data Ingestion",
            time.perf_counter() - stage_start,
            f"rows={len(df_raw):,} | columns={len(df_raw.columns)}",
        )

        stage_start = time.perf_counter()
        log_stage(2, self._TOTAL_STAGES, "Data Preparation", "Clean records, engineer learner-course features, and persist reusable training artifacts")
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.run(df_raw)
        feature_source = preprocessor.feature_source_
        del df_raw
        log_stage_result(
            "Data Preparation",
            time.perf_counter() - stage_start,
            f"interactions={len(feature_source):,} | users={feature_source['user_id'].nunique():,} | courses={feature_source['course_id'].nunique():,}",
        )
        del processed_df

        stage_start = time.perf_counter()
        log_stage(3, self._TOTAL_STAGES, "Artifact Loading", "Reload sparse collaborative inputs and course-level features from disk")
        payload = load_pickle(USER_ITEM_MATRIX_PATH)
        course_features = load_pickle(COURSE_FEATURES_PATH)
        log_stage_result(
            "Artifact Loading",
            time.perf_counter() - stage_start,
            f"user_item={payload['matrix'].shape[0]:,} x {payload['matrix'].shape[1]:,} | course_features={course_features.shape[0]:,} x {course_features.shape[1]}",
        )

        stage_start = time.perf_counter()
        log_stage(4, self._TOTAL_STAGES, "Feature Engineering", "Encode each course into a shared semantic + numeric representation")
        feature_engineer = FeatureEngineer()
        content_matrix = feature_engineer.build_course_content_matrix(course_features)
        log_stage_result(
            "Feature Engineering",
            time.perf_counter() - stage_start,
            f"content_matrix={content_matrix.shape[0]:,} x {content_matrix.shape[1]}",
        )

        stage_start = time.perf_counter()
        log_stage(5, self._TOTAL_STAGES, "Model Training", "Train baseline, personalized, KNN, and hybrid recommendation components")

        logger.info("Model block | popularity baseline")
        popularity_model = PopularityModel().fit(course_features)

        logger.info("Model block | content similarity engine")
        content_model = ContentBasedModel().fit(course_features, content_matrix)

        logger.info("Model block | explicit user-interest profiles")
        interest_model = UserInterestModel().fit(
            interactions_df=feature_source,
            course_features=course_features,
            content_matrix=content_matrix,
            feature_engineer=feature_engineer,
            course_similarity_matrix=content_model.sim_matrix,
        )

        # Keep a reference to content_matrix before deleting so it can be
        # passed to the evaluator — avoids reloading the SentenceTransformer.
        _eval_content_matrix = content_matrix
        del content_matrix

        logger.info("Model block | collaborative ALS")
        collaborative_model = CollaborativeModel().fit(payload)

        logger.info("Model block | KNN user-based")
        knn_model = KNNRecommender().fit(payload)
        del payload

        logger.info("Model block | hybrid ensemble")
        hybrid_model = HybridModel(
            collab_model=collaborative_model,
            content_model=content_model,
            popularity_model=popularity_model,
            knn_model=knn_model,
            interest_model=interest_model,
            course_features=course_features,
        ).fit()

        log_stage_result(
            "Model Training",
            time.perf_counter() - stage_start,
            "artifacts=popularity, content, interests, collaborative, knn, hybrid",
        )

        stage_start = time.perf_counter()
        log_stage(6, self._TOTAL_STAGES, "Offline Evaluation", "Benchmark the hybrid system against its component recommenders")
        RecommendationEvaluator().evaluate(feature_source, course_content_matrix=_eval_content_matrix)
        del feature_source
        log_stage_result(
            "Offline Evaluation",
            time.perf_counter() - stage_start,
            "report=reports/evaluation_metrics.json",
        )

        log_banner(
            "TRAINING PIPELINE COMPLETE",
            f"Runtime | {format_duration(time.perf_counter() - pipeline_start)}",
        )

        return hybrid_model