"""
============================================================
Prediction Pipeline Tests
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This test module verifies that the Prediction Pipeline
works correctly for model inference.

The prediction pipeline is used during deployment to
generate real-time bankruptcy risk predictions.

Why This Test Matters
---------------------
The prediction pipeline sits at the end of the ML system:

Training Pipeline
        ↓
Trained Model Artifact
        ↓
Prediction Pipeline
        ↓
User Predictions

This test ensures that:

• the trained model loads successfully
• the prediction pipeline produces valid outputs
• invalid inputs are rejected properly
"""

# ==========================================================
# IMPORT REQUIRED MODULES
# ==========================================================

import pytest

from bankruptcy.pipeline.training_pipeline import TrainingPipeline
from bankruptcy.pipeline.prediction_pipeline import PredictionPipeline


# ==========================================================
# TEST: PREDICTION PIPELINE SUCCESS CASE
# ==========================================================

def test_prediction_pipeline_runs_successfully():
    """
    Verify that the prediction pipeline:

    • loads the trained model
    • accepts valid input features
    • produces a valid prediction result
    """

    # ======================================================
    # 1️⃣ ENSURE MODEL IS TRAINED
    # ======================================================

    training_pipeline = TrainingPipeline()

    training_pipeline.run_pipeline()


    # ======================================================
    # 2️⃣ LOAD PREDICTION PIPELINE
    # ======================================================

    pipeline = PredictionPipeline()

    # Load model to retrieve expected feature structure
    pipeline._load_model()

    expected_features = list(pipeline.model.feature_names_in_)


    # Build valid input dynamically
    valid_input = {feature: 0.0 for feature in expected_features}


    result = pipeline.predict(valid_input)


    # ======================================================
    # 3️⃣ VALIDATE OUTPUT STRUCTURE
    # ======================================================

    assert "prediction" in result

    assert "bankruptcy_probability" in result


    # ======================================================
    # 4️⃣ VALIDATE PREDICTION LABEL
    # ======================================================

    assert result["prediction"] in [
        "bankruptcy",
        "non-bankruptcy"
    ]


    # ======================================================
    # 5️⃣ VALIDATE PROBABILITY RANGE
    # ======================================================

    assert 0.0 <= result["bankruptcy_probability"] <= 1.0


# ==========================================================
# TEST: INVALID INPUT HANDLING
# ==========================================================

def test_prediction_pipeline_rejects_invalid_input():
    """
    Verify that the prediction pipeline rejects
    invalid inputs that do not match expected features.
    """

    pipeline = PredictionPipeline()


    # Intentionally provide incorrect feature set
    invalid_input = {"wrong_feature": 1.0}


    with pytest.raises(Exception):

        pipeline.predict(invalid_input)