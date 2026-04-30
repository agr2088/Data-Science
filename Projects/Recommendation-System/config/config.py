"""
============================================================
Central Configuration — Online Course Recommendation System
============================================================

Purpose
-------
Single source of truth for all project constants, paths,
model settings, and feature definitions.
"""

import os


# ==========================================================
# 1. PROJECT METADATA
# ==========================================================

PROJECT_NAME    = "Online Course Recommendation System"
PROJECT_VERSION = "4.0 (Interest + ALS + KNN + MiniLM Edition)"


# ==========================================================
# 2. PROJECT ROOT DIRECTORY
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==========================================================
# 3. DIRECTORY STRUCTURE
# ==========================================================

DATA_DIR           = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR  = os.path.join(BASE_DIR, "models")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


# ==========================================================
# 4. FILE PATHS
# ==========================================================

RAW_DATA_PATH         = os.path.join(RAW_DATA_DIR,       "online_course_recommendation.xlsx")
PROCESSED_DATA_PATH   = os.path.join(PROCESSED_DATA_DIR, "processed_data.csv")
USER_ITEM_MATRIX_PATH = os.path.join(PROCESSED_DATA_DIR, "user_item_matrix.pkl")
COURSE_FEATURES_PATH  = os.path.join(PROCESSED_DATA_DIR, "course_features.pkl")

SCALER_PATH              = os.path.join(PROCESSED_DATA_DIR, "preprocessor_scaler.pkl")

COLLAB_MODEL_PATH        = os.path.join(MODELS_DIR, "collaborative_model.pkl")
CONTENT_MODEL_PATH       = os.path.join(MODELS_DIR, "content_model.pkl")
HYBRID_MODEL_PATH        = os.path.join(MODELS_DIR, "hybrid_model.pkl")
POPULARITY_MODEL_PATH    = os.path.join(MODELS_DIR, "popularity_model.pkl")
USER_INTEREST_MODEL_PATH = os.path.join(MODELS_DIR, "user_interest_model.pkl")
KNN_MODEL_PATH           = os.path.join(MODELS_DIR, "knn_model.pkl")

LOG_FILE               = os.path.join(LOGS_DIR,    "pipeline.log")
EVALUATION_REPORT_PATH = os.path.join(REPORTS_DIR, "evaluation_metrics.json")


# ==========================================================
# 5. DATASET SCHEMA
# ==========================================================

EXPECTED_COLUMNS = [
    "user_id", "course_id", "course_name", "instructor",
    "course_duration_hours", "certification_offered", "difficulty_level",
    "rating", "enrollment_numbers", "course_price", "feedback_score",
    "study_material_available", "time_spent_hours", "previous_courses_taken",
]

BINARY_COLUMNS   = ["certification_offered", "study_material_available"]
DIFFICULTY_ORDER = ["Beginner", "Intermediate", "Advanced"]


# ==========================================================
# 6. OUTLIER CLIPPING BOUNDS
# ==========================================================

FLOAT_BOUNDS = {
    "course_duration_hours":  (5.0,  100.0),
    "rating":                 (1.0,    5.0),
    "course_price":           (20.0, 500.0),
    "feedback_score":         (0.0,    1.0),
    "time_spent_hours":       (1.0,  100.0),
    "previous_courses_taken": (0.0,   50.0),
}


# ==========================================================
# 7. NORMALISATION COLUMNS
# ==========================================================

SCALE_COLUMNS = [
    "course_duration_hours", "rating", "enrollment_numbers",
    "course_price", "feedback_score", "time_spent_hours",
    "previous_courses_taken", "completion_rate", "engagement_score",
]


# ==========================================================
# 8. FEATURE ENGINEERING SETTINGS
# ==========================================================

ENGAGEMENT_WEIGHTS = {
    "rating":          0.4,
    "feedback_score":  0.3,
    "completion_rate": 0.3,
}

SEMANTIC_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

NUMERIC_FEATURE_COLS = [
    "course_duration_hours", "difficulty_level",
    "rating", "enrollment_numbers", "course_price", "feedback_score",
    "time_spent_hours", "completion_rate", "engagement_score",
    "previous_courses_taken",
]

PRICE_TIER_BINS   = [0, 100, 250, 500]
PRICE_TIER_LABELS = ["low", "mid", "high"]


# ==========================================================
# 9. MODEL SETTINGS
# ==========================================================

# Number of similar users to look at for collaborative filtering
N_SIMILAR_USERS = 20

# Number of nearest neighbours for KNN recommender
KNN_N_NEIGHBORS = 20

# FINAL TUNING: Dataset is highly sparse, so Collaborative/KNN perform poorly.
# The Semantic Content model is exceptionally strong (~90% HitRate).
# Weights are now heavily skewed toward Content and Popularity.

# Full-start weights: user has >= WARM_START_THRESHOLD interactions
HYBRID_WEIGHTS = {
    "content":       0.70,
    "popularity":    0.12,
    "collaborative": 0.08,
    "user_interest": 0.07,
    "knn":           0.03,
}

# Warm-start weights: COLD_START_THRESHOLD to WARM_START_THRESHOLD interactions
WARM_START_WEIGHTS = {
    "content":       0.65,
    "popularity":    0.18,
    "user_interest": 0.08,
    "collaborative": 0.06,
    "knn":           0.03,
}

# Cold-start weights: fewer than COLD_START_THRESHOLD interactions
COLD_START_WEIGHTS = {
    "popularity":    0.55,
    "content":       0.45,
    "user_interest": 0.00,
    "collaborative": 0.00,
    "knn":           0.00,
}
# Default number of recommendations to return
DEFAULT_N_RECOMMENDATIONS = 10

# Offline Evaluation Protocol Settings
EVAL_K            = 10
EVAL_MAX_USERS    = 500
EVAL_RANDOM_STATE = 42

# Minimum interaction history size to use interest/collaborative/knn arms
COLD_START_THRESHOLD = 2
WARM_START_THRESHOLD = 4

HYBRID_RANK_BLEND = 0.75

# Business logic boost multipliers applied in hybrid_model.py
CERT_BOOST_MULTIPLIER        = 1.02
MATERIALS_BOOST_MULTIPLIER   = 1.01
DIFFICULTY_PROGRESSION_BOOST = 1.03


# ==========================================================
# 10. UTILITY — Create Project Directories
# ==========================================================

def create_project_directories():
    """
    Create all required project folders if they do not exist.
    Called automatically when this module is imported.
    """
    for directory in [
        RAW_DATA_DIR, PROCESSED_DATA_DIR,
        MODELS_DIR, LOGS_DIR, REPORTS_DIR, FIGURES_DIR,
    ]:
        os.makedirs(directory, exist_ok=True)


# ==========================================================
# 11. INITIALIZE PROJECT STRUCTURE ON IMPORT
# ==========================================================

create_project_directories()