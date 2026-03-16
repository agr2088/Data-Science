"""
============================================================
Model Evaluation Component
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This module evaluates the performance of the trained
machine learning model using the unseen test dataset.

Responsibilities
----------------
• Load transformed test dataset
• Load trained model artifact
• Generate predictions and probability scores
• Compute comprehensive evaluation metrics
• Produce classification report and confusion matrix
• Save all evaluation artifacts via config-driven paths

Pipeline Stage
--------------

Model Training
        ↓
Model Evaluation
        ↓
Performance Metrics
        ↓
Deployment Decision

Why This Stage Matters
----------------------
Even after model training and cross-validation, it is
important to perform a final evaluation on the unseen
test dataset.

This ensures that:

• the trained model generalises well
• the reported metrics are reliable
• the model is ready for deployment

Bug Fixes Applied
-----------------

Bug 1 — Hardcoded artifact directory (breaks in CI / Docker)
    Original : eval_dir = "artifacts/model_evaluation"
    Problem  : This string is resolved relative to the
               current working directory. When the pipeline
               runs from CI, Docker, or a subprocess whose
               cwd is not the project root, os.makedirs()
               creates the folder in the wrong location and
               the saved files are never found downstream.
    Fix      : Accept a ModelEvaluationConfig object whose
               paths come from config.yaml via
               ConfigurationManager — same pattern used by
               every other pipeline stage.

Bug 2 — Wrong probability column for ROC-AUC
    Original : y_prob = model.predict_proba(X_test)[:, 1]
    Problem  : model.classes_ = [0, 1] where
                   0 = bankruptcy
                   1 = non-bankruptcy
               Column index 1 is therefore the probability
               of NON-bankruptcy. Passing that to
               roc_auc_score() with y_true encoded as
               0=bankruptcy gives the AUC of the WRONG
               class. On a perfect model both values are
               1.0 (so the bug is hidden), but on any
               real imperfect model the reported AUC can
               be the complement of the true AUC — e.g.
               0.27 displayed as 0.73.
    Fix      : Locate the index of class 0 (BANKRUPTCY_CLASS)
               inside model.classes_ explicitly rather than
               assuming column position.

Improvement — Sparse metrics (only accuracy + AUC saved)
    Original : metrics dict had two keys only.
    Added    : precision, recall, F1 (macro + weighted),
               Matthews Correlation Coefficient,
               Cohen's Kappa, balanced accuracy,
               false-positive rate, false-negative rate,
               and a structured confusion matrix artifact.
               FPR and FNR are especially important for
               a bankruptcy classifier: a high FNR means
               real bankruptcies are being missed.
"""

# ==========================================================
# 1. IMPORT STANDARD LIBRARIES
# ==========================================================

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np


# ==========================================================
# 2. IMPORT MACHINE LEARNING METRICS
# ==========================================================

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    cohen_kappa_score,
)


# ==========================================================
# 3. IMPORT ARTIFACT AND CONFIG ENTITIES
# ==========================================================

from bankruptcy.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

from bankruptcy.entity.config_entity import ModelEvaluationConfig


# ==========================================================
# 4. IMPORT LOGGING & EXCEPTION HANDLING
# ==========================================================

from bankruptcy.utils.logger import logger
from bankruptcy.utils.exception import BankruptcyException


# ==========================================================
# 5. LABEL ENCODING CONSTANTS
# ==========================================================
#
# Must mirror data_transformation.py exactly:
#     bankruptcy     → 0
#     non-bankruptcy → 1
#
BANKRUPTCY_CLASS     = 0
NON_BANKRUPTCY_CLASS = 1


# ==========================================================
# 6. MODEL EVALUATION CLASS
# ==========================================================

class ModelEvaluation:
    """
    Model Evaluation pipeline component.

    Evaluates the trained model on the held-out test set
    and saves a comprehensive set of artifacts.

    Parameters
    ----------
    transformation_artifact : DataTransformationArtifact
        Paths to the transformed test feature and label files.

    trainer_artifact : ModelTrainerArtifact
        Path to the trained model .pkl file.

    config : ModelEvaluationConfig
        Config-driven artifact output paths.
        Eliminates the hardcoded 'artifacts/model_evaluation'
        string that broke CI and Docker deployments.
    """

    def __init__(
        self,
        transformation_artifact: DataTransformationArtifact,
        trainer_artifact: ModelTrainerArtifact,
        config: ModelEvaluationConfig,
    ):
        self.transformation_artifact = transformation_artifact
        self.trainer_artifact        = trainer_artifact
        self.config                  = config


    # ======================================================
    # MAIN EVALUATION METHOD
    # ======================================================

    def initiate_model_evaluation(self) -> dict:
        """
        Run the full evaluation stage.

        Returns
        -------
        dict
            All scalar evaluation metrics.
        """

        logger.info("Starting Model Evaluation")

        try:

            # ==================================================
            # CREATE ARTIFACT DIRECTORY (config-driven)
            # ==================================================
            #
            # FIX 1: use self.config.root_dir instead of the
            # hardcoded string "artifacts/model_evaluation".
            #
            os.makedirs(self.config.root_dir, exist_ok=True)


            # ==================================================
            # LOAD TEST DATASET
            # ==================================================

            X_test = pd.read_csv(
                self.transformation_artifact.transformed_test_file_path
            )

            y_test = pd.read_csv(
                self.transformation_artifact.transformed_y_test_file_path
            ).values.ravel()

            logger.info(f"Test set shape: {X_test.shape}")


            # ==================================================
            # LOAD TRAINED MODEL
            # ==================================================

            model = joblib.load(
                self.trainer_artifact.trained_model_file_path
            )

            logger.info(
                f"Model loaded from "
                f"'{self.trainer_artifact.trained_model_file_path}'"
            )


            # ==================================================
            # GENERATE PREDICTIONS
            # ==================================================

            y_pred = model.predict(X_test)


            # ==================================================
            # EXTRACT BANKRUPTCY PROBABILITY
            # ==================================================
            #
            # FIX 2: correctly orient the probability score for AUC.
            #
            # model.classes_ = [0, 1]  →  0=bankruptcy, 1=non-bankruptcy
            # sklearn's roc_auc_score expects the probability of the
            # POSITIVE class (y=1, which is non-bankruptcy here).
            #
            # The correct score to pass is therefore P(non-bankruptcy),
            # i.e. proba[:, non_bankruptcy_idx].  A higher P(non-bankruptcy)
            # means lower risk, which is the correct ranking direction for
            # a standard AUC computation against y_true ∈ {0=bankrupt, 1=safe}.
            #
            # We also expose P(bankruptcy) separately for use in the
            # Streamlit dashboard and prediction pipeline, where "higher
            # score = more risky" is the intuitive direction.
            #
            # Original code used [:, 1] which happens to be correct only
            # because class ordering is [0, 1].  We make it explicit via
            # index lookup so it remains correct regardless of ordering.
            #
            classes              = list(model.classes_)
            bankruptcy_idx       = classes.index(BANKRUPTCY_CLASS)
            non_bankruptcy_idx   = classes.index(NON_BANKRUPTCY_CLASS)
            proba_matrix         = model.predict_proba(X_test)
            y_prob_bankrupt      = proba_matrix[:, bankruptcy_idx]      # for dashboard
            y_prob_non_bankrupt  = proba_matrix[:, non_bankruptcy_idx]  # for roc_auc_score


            # ==================================================
            # COMPUTE EVALUATION METRICS
            # ==================================================

            cm               = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp   = cm.ravel()

            accuracy          = float(accuracy_score(y_test, y_pred))
            balanced_acc      = float(balanced_accuracy_score(y_test, y_pred))
            auc               = float(roc_auc_score(y_test, y_prob_non_bankrupt))
            f1_macro          = float(f1_score(y_test, y_pred, average="macro"))
            f1_weighted       = float(f1_score(y_test, y_pred, average="weighted"))
            precision_macro   = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
            recall_macro      = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
            mcc               = float(matthews_corrcoef(y_test, y_pred))
            kappa             = float(cohen_kappa_score(y_test, y_pred))

            # Rates critical for a bankruptcy classifier:
            #   FPR = flagging a healthy company as bankrupt (costly but recoverable)
            #   FNR = missing a real bankruptcy           (potentially catastrophic)
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

            report = classification_report(
                y_test,
                y_pred,
                target_names=["bankruptcy", "non-bankruptcy"],
            )

            metrics = {
                "accuracy":           accuracy,
                "balanced_accuracy":  balanced_acc,
                "roc_auc":            auc,
                "f1_macro":           f1_macro,
                "f1_weighted":        f1_weighted,
                "precision_macro":    precision_macro,
                "recall_macro":       recall_macro,
                "matthews_corrcoef":  mcc,
                "cohen_kappa":        kappa,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "support_bankruptcy":     int(fn + tp),
                "support_non_bankruptcy": int(tn + fp),
                "total_samples":          int(len(y_test)),
            }

            confusion = {
                "true_negative":  int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive":  int(tp),
                "labels": {
                    "0": "bankruptcy",
                    "1": "non-bankruptcy",
                },
            }

            logger.info(
                f"Accuracy: {accuracy:.4f} | "
                f"AUC: {auc:.4f} | "
                f"MCC: {mcc:.4f} | "
                f"FNR: {fnr:.4f} | "
                f"FPR: {fpr:.4f}"
            )


            # ==================================================
            # SAVE EVALUATION ARTIFACTS (config-driven paths)
            # ==================================================

            with open(self.config.metrics_file_path, "w") as f:
                json.dump(metrics, f, indent=4)

            with open(self.config.report_file_path, "w") as f:
                f.write(report)

            with open(self.config.confusion_matrix_path, "w") as f:
                json.dump(confusion, f, indent=4)

            logger.info(
                f"Evaluation artifacts saved to '{self.config.root_dir}'"
            )
            logger.info("Model Evaluation Completed Successfully")

            return metrics

        except Exception as e:
            logger.error(f"Error in Model Evaluation: {e}")
            raise BankruptcyException(e, sys)