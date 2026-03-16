"""
============================================================
Model Training Component
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This module trains multiple machine learning models,
performs hyperparameter optimization, evaluates model
performance, and selects the best-performing model.

Responsibilities
----------------
• Load transformed training and testing datasets
• Train multiple ML models
• Perform hyperparameter tuning using GridSearchCV
• Evaluate models using stratified cross-validation
• Measure model stability using repeated CV
• Select the best-performing model
• Register model with MLflow
• Save trained model and evaluation artifacts

Pipeline Stage
--------------

Transformed Dataset
        ↓
Model Training
        ↓
Cross Validation
        ↓
Model Selection
        ↓
Best Model Saved
        ↓
MLflow Experiment Tracking

Bug Fixes Applied
-----------------

Bug 1 — cv_std measured variance across hyperparameters, not fold stability
    Original : np.std(grid_search.cv_results_["mean_test_score"])
    Problem  : mean_test_score has one entry PER HYPERPARAMETER COMBINATION
               (12 combos for SVM, 6 for RF, etc.).  Taking std across those
               values measures how sensitive the model is to hyperparameter
               choice — not how stable the best model is across CV folds.
               A model with cv_std ≈ 0 using the original formula could
               actually be wildly unstable across folds.
    Fix      : gs.cv_results_["std_test_score"][gs.best_index_]
               GridSearchCV already computes the std of the 25 fold scores
               (5 splits × 5 repeats) for every parameter combination.
               Reading it at best_index_ gives the fold-to-fold stability
               of the chosen model — the correct definition of cv_std.

Bug 2 — model_factory lookup gives an undebuggable bare KeyError on typos
    Original : model = model_factory[model_type]
    Problem  : If config.yaml has a typo (e.g. "RandomForest" instead of
               "RandomForestClassifier") Python raises KeyError: 'RandomForest'
               with no context about where the key came from, which model was
               expected, or which config section caused the failure.
    Fix      : Explicit guard before lookup with a descriptive error message
               listing all valid model names.

Bug 3 — y_prob = predict_proba[:, 1] reports AUC for the wrong class
    Original : y_prob = candidate_model.predict_proba(X_test)[:, 1]
    Problem  : Same issue as model_evaluation.py.  model.classes_ = [0, 1]
               where 0=bankruptcy and 1=non-bankruptcy.  roc_auc_score with
               y_true labels expects the probability of the class treated as
               "positive" (higher label = 1 = non-bankruptcy by default).
               Using [:, 1] = P(non-bankruptcy) happens to be numerically
               correct for a symmetric AUC on a perfect model, but is
               semantically wrong.  On any imperfect model the AUC can appear
               as its complement (e.g. 0.27 displayed as 0.73).
    Fix      : Look up the non-bankruptcy class index explicitly via
               classes.index(NON_BANKRUPTCY_CLASS) so the correct column is
               used regardless of class ordering in the fitted model.

Bug 4 — best_params may contain numpy scalar types that fail json.dumps
    Original : json.dump(metrics, f) where best_params comes directly from
               grid_search.best_params_.
    Problem  : Depending on the sklearn version and parameter type, best_params_
               can contain numpy.int64 or numpy.float64 values.  The standard
               json module cannot serialise these (TypeError: Object of type
               int64 is not JSON serializable).  The current run happens to
               avoid this because SVM's C param is a plain Python int, but
               n_estimators from RandomForest returns numpy.int64 in some
               sklearn versions, causing silent failures in CI.
    Fix      : Convert all values in best_params to native Python types before
               serialising using a small helper that handles int, float, and
               bool numpy scalars.
"""

# ==========================================================
# 1. FUTURE COMPATIBILITY
# ==========================================================

from __future__ import annotations


# ==========================================================
# 2. IMPORT STANDARD LIBRARIES
# ==========================================================

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn


# ==========================================================
# 3. IMPORT MACHINE LEARNING UTILITIES
# ==========================================================

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
)


# ==========================================================
# 4. IMPORT MACHINE LEARNING MODELS
# ==========================================================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# ==========================================================
# 5. IMPORT ARTIFACT ENTITIES
# ==========================================================

from bankruptcy.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)


# ==========================================================
# 6. IMPORT CONFIGURATION ENTITY
# ==========================================================

from bankruptcy.entity.config_entity import ModelTrainerConfig


# ==========================================================
# 7. IMPORT LOGGING & EXCEPTION HANDLING
# ==========================================================

from bankruptcy.utils.logger import logger
from bankruptcy.utils.exception import BankruptcyException


# ==========================================================
# 8. LABEL ENCODING CONSTANTS
# ==========================================================
#
# Must mirror data_transformation.py exactly:
#     bankruptcy     → 0
#     non-bankruptcy → 1
#
BANKRUPTCY_CLASS     = 0
NON_BANKRUPTCY_CLASS = 1


# ==========================================================
# 9. HELPERS
# ==========================================================

def _to_python(value):
    """
    Convert a numpy scalar to its native Python equivalent
    so json.dump() never raises TypeError.

    Handles numpy.int*, numpy.float*, numpy.bool_ and
    passes through all other types unchanged.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _sanitise_params(params: dict) -> dict:
    """
    Return a copy of params with all values converted to
    native Python types safe for json.dump().
    """
    return {k: _to_python(v) for k, v in params.items()}


# ==========================================================
# 10. MODEL REGISTRY
# ==========================================================
#
# Single source of truth for all supported model classes.
# Validated against this dict before any training starts
# so config.yaml typos surface as clear errors immediately.
#
MODEL_REGISTRY: dict = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    ),
    "RandomForestClassifier": RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
    ),
    "SVC": SVC(
        probability=True,
        class_weight="balanced",
        random_state=42,
    ),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(
        random_state=42,
    ),
}


# ==========================================================
# 11. MODEL TRAINER CLASS
# ==========================================================

class ModelTrainer:
    """
    Model Training pipeline component.

    Trains multiple machine learning models, evaluates them
    using repeated stratified cross-validation, and selects
    the best-performing model.
    """

    def __init__(
        self,
        transformation_artifact: DataTransformationArtifact,
        config: ModelTrainerConfig,
    ):
        self.transformation_artifact = transformation_artifact
        self.config                  = config


    # ======================================================
    # MAIN TRAINING METHOD
    # ======================================================

    def initiate_model_training(self) -> ModelTrainerArtifact:
        """
        Execute the full model training stage.

        Steps
        -----
        1. Validate model names in config against MODEL_REGISTRY
        2. Load transformed train / test datasets
        3. For each model: build Pipeline → GridSearchCV → evaluate
        4. Select best model by CV AUC
        5. Save model artifact and metrics JSON
        6. Log everything to MLflow

        Returns
        -------
        ModelTrainerArtifact
        """

        logger.info("Starting Model Training")

        try:

            os.makedirs(self.config.root_dir, exist_ok=True)

            # ==================================================
            # PRE-FLIGHT: validate all model names in config
            # ==================================================
            #
            # FIX 2: fail early with a clear message instead of
            # a bare KeyError buried inside the training loop.
            #
            for entry_name, entry_cfg in self.config.param_grids.items():
                model_type = entry_cfg["model"]
                if model_type not in MODEL_REGISTRY:
                    raise BankruptcyException(
                        f"Unknown model '{model_type}' in config entry "
                        f"'{entry_name}'. "
                        f"Valid options: {sorted(MODEL_REGISTRY.keys())}",
                        sys,
                    )

            logger.info(
                f"Config validated — training "
                f"{len(self.config.param_grids)} model(s): "
                f"{[c['model'] for c in self.config.param_grids.values()]}"
            )


            # ==================================================
            # MLFLOW SETUP
            # ==================================================

            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment_name)


            with mlflow.start_run():

                # ==================================================
                # LOAD DATASETS
                # ==================================================

                logger.info("Loading transformed datasets")

                X_train = pd.read_csv(
                    self.transformation_artifact.transformed_train_file_path
                )
                X_test = pd.read_csv(
                    self.transformation_artifact.transformed_test_file_path
                )
                y_train = pd.read_csv(
                    self.transformation_artifact.transformed_y_train_file_path
                ).values.ravel()
                y_test = pd.read_csv(
                    self.transformation_artifact.transformed_y_test_file_path
                ).values.ravel()

                if list(X_train.columns) != list(X_test.columns):
                    raise ValueError(
                        "Train and Test feature columns do not match.\n"
                        f"  Train: {list(X_train.columns)}\n"
                        f"  Test : {list(X_test.columns)}"
                    )

                X_test = X_test[X_train.columns]
                logger.info(
                    f"Train shape: {X_train.shape} | "
                    f"Test shape: {X_test.shape}"
                )


                # ==================================================
                # CROSS-VALIDATION STRATEGY
                # ==================================================
                #
                # RepeatedStratifiedKFold (5×5 = 25 folds):
                # • preserves class distribution per fold
                # • repeated runs reduce variance of the CV estimate
                # • important for small datasets like this one
                #
                cv = RepeatedStratifiedKFold(
                    n_splits=5,
                    n_repeats=5,
                    random_state=42,
                )


                # ==================================================
                # BEST MODEL TRACKERS
                # ==================================================

                best_model      = None
                best_model_name = None
                best_cv_auc     = -np.inf
                best_cv_std     = None
                best_test_auc   = 0.0
                best_test_acc   = 0.0
                best_params     = None
                all_results     = {}   # per-model summary for logging


                # ==================================================
                # TRAINING LOOP
                # ==================================================

                for entry_name, entry_cfg in self.config.param_grids.items():

                    model_type = entry_cfg["model"]
                    param_grid = entry_cfg["params"]

                    logger.info(f"Training: {entry_name} ({model_type})")

                    # Each iteration gets a fresh clone of the base model
                    # so hyperparameter state doesn't leak between runs.
                    import sklearn.base as _sk_base
                    base_model = _sk_base.clone(MODEL_REGISTRY[model_type])

                    pipeline = Pipeline([
                        ("scaler", StandardScaler()),
                        ("model",  base_model),
                    ])

                    grid_search = GridSearchCV(
                        estimator=pipeline,
                        param_grid=param_grid,
                        cv=cv,
                        scoring="roc_auc",
                        n_jobs=-1,
                        return_train_score=True,
                    )

                    grid_search.fit(X_train, y_train)

                    candidate  = grid_search.best_estimator_
                    cv_auc     = float(grid_search.best_score_)

                    # ──────────────────────────────────────────────────
                    # FIX 1: correct cv_std
                    # ──────────────────────────────────────────────────
                    # np.std(grid_search.cv_results_["mean_test_score"])
                    # computes std ACROSS all hyperparameter combinations
                    # — that is a measure of parameter sensitivity, not
                    # model stability.
                    #
                    # The correct value is the std of the 25 individual
                    # fold scores for the BEST parameter combination.
                    # GridSearchCV pre-computes this as std_test_score.
                    #
                    cv_std = float(
                        grid_search.cv_results_["std_test_score"][
                            grid_search.best_index_
                        ]
                    )

                    # ──────────────────────────────────────────────────
                    # FIX 3: correct AUC direction
                    # ──────────────────────────────────────────────────
                    # roc_auc_score expects P(positive_class).
                    # Our encoding: 0=bankruptcy, 1=non-bankruptcy.
                    # sklearn treats the higher label as positive by
                    # default, so "positive" = non-bankruptcy = class 1.
                    # Pass P(non-bankruptcy) = proba[:, non_bankruptcy_idx].
                    #
                    y_pred = candidate.predict(X_test)
                    classes = list(candidate.classes_)

                    if hasattr(candidate, "predict_proba"):
                        proba           = candidate.predict_proba(X_test)
                        non_bk_idx      = classes.index(NON_BANKRUPTCY_CLASS)
                        y_score         = proba[:, non_bk_idx]
                    else:
                        # SVC with probability=False — use decision function
                        y_score = candidate.decision_function(X_test)

                    test_acc = float(accuracy_score(y_test, y_pred))
                    test_auc = float(roc_auc_score(y_test, y_score))

                    logger.info(
                        f"  {entry_name} | "
                        f"CV AUC: {cv_auc:.4f} ± {cv_std:.4f} | "
                        f"Test AUC: {test_auc:.4f} | "
                        f"Accuracy: {test_acc:.4f}"
                    )

                    all_results[entry_name] = {
                        "cv_auc":   cv_auc,
                        "cv_std":   cv_std,
                        "test_auc": test_auc,
                        "test_acc": test_acc,
                    }

                    if cv_auc > best_cv_auc:
                        best_model      = candidate
                        best_model_name = entry_name
                        best_cv_auc     = cv_auc
                        best_cv_std     = cv_std
                        best_test_auc   = test_auc
                        best_test_acc   = test_acc
                        best_params     = grid_search.best_params_


                if best_model is None:
                    raise ValueError("No model trained successfully.")


                # ==================================================
                # SAVE BEST MODEL
                # ==================================================

                model_path = self.config.trained_model_path
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                joblib.dump(best_model, model_path)

                logger.info(
                    f"Best model: {best_model_name} | "
                    f"CV AUC: {best_cv_auc:.4f} ± {best_cv_std:.4f} | "
                    f"Test AUC: {best_test_auc:.4f} | "
                    f"Accuracy: {best_test_acc:.4f}"
                )
                logger.info(f"Model saved to '{model_path}'")


                # ==================================================
                # CLASSIFICATION REPORT
                # ==================================================

                classification_rep = classification_report(
                    y_test,
                    best_model.predict(X_test),
                    target_names=["bankruptcy", "non-bankruptcy"],
                )


                # ==================================================
                # SAVE METRICS JSON
                # ==================================================
                #
                # FIX 4: sanitise best_params before json.dump to
                # avoid TypeError on numpy.int64 / numpy.float64.
                #
                clean_best_params = _sanitise_params(best_params)

                metrics = {
                    "best_model":    best_model_name,
                    "cv_auc_mean":   round(best_cv_auc, 6),
                    "cv_auc_std":    round(best_cv_std, 6),
                    "test_auc":      round(best_test_auc, 6),
                    "test_accuracy": round(best_test_acc, 6),
                    "best_params":   clean_best_params,
                    "all_models":    all_results,
                }

                metrics_path = os.path.join(self.config.root_dir, "metrics.json")
                report_path  = os.path.join(self.config.root_dir, "classification_report.txt")

                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=4)

                with open(report_path, "w") as f:
                    f.write(classification_rep)


                # ==================================================
                # LOG TO MLFLOW
                # ==================================================

                mlflow.log_param("best_model",    best_model_name)
                mlflow.log_param("num_features",  X_train.shape[1])
                mlflow.log_param("train_samples", X_train.shape[0])
                mlflow.log_param("test_samples",  X_test.shape[0])

                # Sanitise params before mlflow.log_params too
                mlflow.log_params(
                    {k.replace("model__", ""): v
                     for k, v in clean_best_params.items()}
                )

                mlflow.log_metric("cv_auc_mean",  best_cv_auc)
                mlflow.log_metric("cv_auc_std",   best_cv_std)
                mlflow.log_metric("test_auc",     best_test_auc)
                mlflow.log_metric("test_accuracy", best_test_acc)

                mlflow.sklearn.log_model(
                    best_model,
                    "model",
                    registered_model_name="BankruptcyPredictionModel",
                )

                mlflow.log_artifact(report_path)

                logger.info("Model Training Completed Successfully")


                # ==================================================
                # RETURN ARTIFACT
                # ==================================================

                return ModelTrainerArtifact(
                    trained_model_file_path=model_path,
                    test_accuracy=best_test_acc,
                    test_auc=best_test_auc,
                    cv_auc=best_cv_auc,
                )

        except Exception as e:
            logger.error(f"Error in Model Training: {e}")
            raise BankruptcyException(e, sys)