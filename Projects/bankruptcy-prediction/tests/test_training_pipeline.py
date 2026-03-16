"""
============================================================
Training Pipeline Integration Tests
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This test module verifies that the complete training
pipeline runs successfully from start to finish.

The training pipeline orchestrates all major stages
of the machine learning system:

Data Ingestion
      ↓
Data Validation
      ↓
Data Transformation
      ↓
Model Training
      ↓
Model Evaluation

Why This Test Matters
---------------------
This test acts as an integration test for the ML pipeline.

It ensures that:

• the full pipeline executes successfully
• a trained model artifact is produced
• evaluation metrics are valid
• the trained model can be loaded correctly
"""

# ==========================================================
# IMPORT REQUIRED MODULES
# ==========================================================

import os
import joblib
from sklearn.pipeline import Pipeline

from bankruptcy.pipeline.training_pipeline import TrainingPipeline
from bankruptcy.config.configuration import ConfigurationManager


# ==========================================================
# TEST: FULL TRAINING PIPELINE
# ==========================================================

def test_full_training_pipeline_runs_successfully():
    """
    Verify that the full ML training pipeline:

    • runs without errors
    • produces a trained model artifact
    • generates valid evaluation metrics
    • saves a loadable sklearn Pipeline
    """

    # ======================================================
    # 1️⃣ RUN FULL TRAINING PIPELINE
    # ======================================================

    pipeline = TrainingPipeline()

    model_artifact = pipeline.run_pipeline()


    # ======================================================
    # 2️⃣ VERIFY MODEL FILE EXISTS
    # ======================================================

    config_manager = ConfigurationManager()

    model_config = config_manager.get_model_trainer_config()

    assert os.path.exists(model_config.trained_model_path)


    # ======================================================
    # 3️⃣ VERIFY METRICS ARE VALID
    # ======================================================

    assert 0.0 <= model_artifact.test_accuracy <= 1.0

    assert 0.0 <= model_artifact.test_auc <= 1.0

    assert 0.0 <= model_artifact.cv_auc <= 1.0


    # ======================================================
    # 4️⃣ OVERFITTING SANITY CHECK
    # ======================================================
    """
    Ensure that cross-validation performance and test
    performance are reasonably close.

    Large differences may indicate severe overfitting.
    """

    assert abs(model_artifact.cv_auc - model_artifact.test_auc) < 0.2


    # ======================================================
    # 5️⃣ VERIFY MODEL CAN BE LOADED
    # ======================================================

    model = joblib.load(model_config.trained_model_path)

    assert isinstance(model, Pipeline)