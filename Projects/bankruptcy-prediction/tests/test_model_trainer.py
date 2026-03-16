"""
============================================================
Model Trainer Tests
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This test module verifies that the Model Training component
successfully trains a machine learning model and produces
a valid model artifact.

Why This Test Matters
---------------------
Model training is the core stage of the ML pipeline.

This test ensures that:

• the training pipeline executes successfully
• a trained model artifact is generated
• evaluation metrics fall within valid ranges
• the saved model is a valid sklearn Pipeline
• the trained model can perform predictions
• metrics.json serialises without numpy type errors
• the cv_auc_std represents fold stability, not param variance
• all_models summary is populated for every trained model

Changes from Original
---------------------

Change 1 — Full pipeline re-run removed from test body
    Original: every pytest run re-executed data ingestion,
              data transformation, AND model training.
              On a laptop with 5 models × 25 CV folds that
              takes 60–120 seconds per test run.  CI pipelines
              time out.

    Fix: two separate tests:

        test_model_trainer_uses_prebuilt_artifacts()
            Loads existing artifacts/data_transformation/*
            and trains a single fast model (LR, one C value).
            Runs in ~5 seconds.  This is the default test.

        test_model_trainer_full_pipeline()
            Marked @pytest.mark.slow — skipped unless you
            explicitly pass -m slow.  Runs the complete
            ingestion → transformation → training chain with
            all five models, same as the original test.

Change 2 — Added assertions for fixed model_trainer.py outputs
    • metrics.json contains 'all_models' key
    • metrics.json is valid JSON (proves numpy type fix works)
    • cv_auc_std is a float in [0, 1]
    • classification_report.txt exists and is non-empty
    • prediction output contains both required keys

Change 3 — feature_names_in_ assertion documented
    The original code had a comment in prediction_pipeline.py
    suggesting Pipeline never carries feature_names_in_.
    This is wrong: sklearn sets feature_names_in_ on the
    Pipeline when it is fitted with a pandas DataFrame
    (which our pipeline always does).

    The assertion is CORRECT and kept, but the docstring
    now explains the DataFrame-fitting dependency so future
    maintainers understand when it would be absent.
"""

# ==========================================================
# IMPORTS
# ==========================================================

import os
import json
import joblib
import pytest
import pandas as pd
from sklearn.pipeline import Pipeline

from bankruptcy.config.configuration import ConfigurationManager
from bankruptcy.components.data_ingestion import DataIngestion
from bankruptcy.components.data_transformation import DataTransformation
from bankruptcy.components.model_trainer import ModelTrainer
from bankruptcy.entity.artifact_entity import DataTransformationArtifact
from bankruptcy.entity.config_entity import ModelTrainerConfig


# ==========================================================
# FIXTURES
# ==========================================================

@pytest.fixture(scope="module")
def prebuilt_transformation_artifact():
    """
    Return a DataTransformationArtifact pointing at the
    already-committed artifact CSVs in artifacts/.

    Using pre-built artifacts avoids re-running ingestion
    and transformation on every test run, keeping the test
    suite fast enough for interactive development and CI.

    If the artifact files are absent (fresh clone), the
    test will fail with a clear FileNotFoundError from
    pd.read_csv() — not a silent assertion error.
    """
    return DataTransformationArtifact(
        transformed_train_file_path="artifacts/data_transformation/X_train.csv",
        transformed_test_file_path="artifacts/data_transformation/X_test.csv",
        transformed_y_train_file_path="artifacts/data_transformation/y_train.csv",
        transformed_y_test_file_path="artifacts/data_transformation/y_test.csv",
    )


@pytest.fixture(scope="module")
def fast_trainer_config():
    """
    A ModelTrainerConfig that trains a single Logistic
    Regression model with one hyperparameter value.

    This keeps the test to ~5 seconds instead of ~120.
    The full multi-model run is covered by test_model_trainer_full_pipeline.
    """
    return ModelTrainerConfig(
        root_dir="artifacts/model_trainer",
        trained_model_path="artifacts/model_trainer/model.pkl",
        param_grids={
            "logistic_regression": {
                "model": "LogisticRegression",
                "params": {"model__C": [1]},
            }
        },
        mlflow_tracking_uri="sqlite:///mlflow.db",
        mlflow_experiment_name="Bankruptcy_Prediction",
    )


# ==========================================================
# TEST 1 — FAST: pre-built artifacts, single model
# ==========================================================

def test_model_trainer_uses_prebuilt_artifacts(
    prebuilt_transformation_artifact,
    fast_trainer_config,
):
    """
    Train one fast model against pre-built transformation
    artifacts and verify the full output contract.

    Covers:
    • model file written to disk
    • artifact is a valid sklearn Pipeline
    • feature_names_in_ is set (Pipeline fitted with DataFrame)
    • all metric ranges are valid
    • metrics.json is valid JSON (numpy serialisation fix)
    • metrics.json contains all_models summary
    • cv_auc_std is a float in [0, 1]
    • classification_report.txt is non-empty
    • predictions have correct shape
    • prediction output contains required keys
    """

    trainer  = ModelTrainer(prebuilt_transformation_artifact, fast_trainer_config)
    artifact = trainer.initiate_model_training()


    # ── 1. Model file exists ──────────────────────────────
    assert os.path.exists(artifact.trained_model_file_path), (
        f"Expected model file at '{artifact.trained_model_file_path}'"
    )


    # ── 2. Metric ranges ─────────────────────────────────
    assert 0.0 <= artifact.test_accuracy <= 1.0
    assert 0.0 <= artifact.test_auc      <= 1.0
    assert 0.0 <= artifact.cv_auc        <= 1.0


    # ── 3. Load and type-check model ─────────────────────
    model = joblib.load(artifact.trained_model_file_path)
    assert isinstance(model, Pipeline), (
        f"Expected sklearn Pipeline, got {type(model).__name__}"
    )


    # ── 4. feature_names_in_ ─────────────────────────────
    # sklearn sets this attribute on a Pipeline when it is
    # fitted with a pandas DataFrame (as our trainer does).
    # It would be absent if fitted with a numpy array.
    # This assertion validates that the training data path
    # is correct and the scaler step received named columns.
    assert hasattr(model, "feature_names_in_"), (
        "Pipeline.feature_names_in_ is absent. "
        "This means the model was fitted with a numpy array "
        "instead of a DataFrame — check DataTransformation output."
    )

    X_test = pd.read_csv(
        prebuilt_transformation_artifact.transformed_test_file_path
    )
    assert list(model.feature_names_in_) == list(X_test.columns), (
        f"Feature mismatch.\n"
        f"  Model expects: {list(model.feature_names_in_)}\n"
        f"  Test has     : {list(X_test.columns)}"
    )


    # ── 5. Prediction shape ───────────────────────────────
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


    # ── 6. metrics.json — valid JSON (numpy type fix) ─────
    metrics_path = os.path.join(fast_trainer_config.root_dir, "metrics.json")
    assert os.path.exists(metrics_path), "metrics.json not written"

    with open(metrics_path) as f:
        metrics = json.load(f)   # raises if numpy types leaked through

    # ── 7. metrics.json — content contract ───────────────
    for key in ("best_model", "cv_auc_mean", "cv_auc_std",
                "test_auc", "test_accuracy", "best_params", "all_models"):
        assert key in metrics, f"metrics.json missing key '{key}'"

    assert isinstance(metrics["cv_auc_std"], float), (
        "cv_auc_std should be a float (fold stability std, not param std)"
    )
    assert 0.0 <= metrics["cv_auc_std"] <= 1.0

    assert isinstance(metrics["all_models"], dict), (
        "all_models should be a dict keyed by model config name"
    )
    assert len(metrics["all_models"]) >= 1

    # ── 8. classification_report.txt non-empty ────────────
    report_path = os.path.join(fast_trainer_config.root_dir, "classification_report.txt")
    assert os.path.exists(report_path), "classification_report.txt not written"
    assert os.path.getsize(report_path) > 0, "classification_report.txt is empty"


    # ── 9. End-to-end prediction output keys ─────────────
    from bankruptcy.pipeline.prediction_pipeline import PredictionPipeline

    pred_pipeline = PredictionPipeline(
        model_path=artifact.trained_model_file_path
    )
    sample_input = {col: 0.5 for col in model.feature_names_in_}
    result = pred_pipeline.predict(sample_input)

    assert "prediction" in result
    assert "bankruptcy_probability" in result
    assert result["prediction"] in ("bankruptcy", "non-bankruptcy")
    assert 0.0 <= result["bankruptcy_probability"] <= 1.0


# ==========================================================
# TEST 2 — SLOW: full pipeline, all models
# ==========================================================

@pytest.mark.slow
def test_model_trainer_full_pipeline():
    """
    Run the complete ingestion → transformation → training
    chain with all five models from config.yaml.

    This test is intentionally marked @pytest.mark.slow
    and skipped in normal pytest runs.

    Run explicitly with:
        pytest -m slow tests/test_model_trainer.py

    It verifies the same contract as the fast test but
    exercises every model in config.yaml and the full
    data pipeline, which gives higher confidence before
    a release or deployment.
    """

    config_manager = ConfigurationManager()

    # Full ingestion
    ingestion_config   = config_manager.get_data_ingestion_config()
    ingestion_artifact = DataIngestion(ingestion_config).initiate_data_ingestion()

    # Full transformation
    transformation_config   = config_manager.get_data_transformation_config()
    transformation_artifact = DataTransformation(
        ingestion_artifact, transformation_config
    ).initiate_data_transformation()

    # Full training (all 5 models)
    model_config = config_manager.get_model_trainer_config()
    trainer      = ModelTrainer(transformation_artifact, model_config)
    artifact     = trainer.initiate_model_training()

    # Core assertions (same as fast test)
    assert os.path.exists(artifact.trained_model_file_path)
    assert 0.0 <= artifact.test_accuracy <= 1.0
    assert 0.0 <= artifact.test_auc      <= 1.0
    assert 0.0 <= artifact.cv_auc        <= 1.0

    model = joblib.load(artifact.trained_model_file_path)
    assert isinstance(model, Pipeline)
    assert hasattr(model, "feature_names_in_")

    # All models should be in the summary
    metrics_path = os.path.join(model_config.root_dir, "metrics.json")
    metrics = json.load(open(metrics_path))
    expected_model_names = set(model_config.param_grids.keys())
    actual_model_names   = set(metrics["all_models"].keys())
    assert expected_model_names == actual_model_names, (
        f"all_models keys mismatch.\n"
        f"  Expected: {sorted(expected_model_names)}\n"
        f"  Got     : {sorted(actual_model_names)}"
    )