"""
============================================================
Prediction Pipeline
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This module provides the inference pipeline used during
model deployment.

It loads the trained machine learning model and performs
real-time predictions based on user input.

Responsibilities
----------------
• Load trained model artifact
• Validate input features
• Align features with model expectations
• Generate predictions
• Compute bankruptcy probability
• Return structured prediction results

Deployment Architecture
-----------------------

Streamlit Dashboard
        ↓
Prediction Pipeline
        ↓
Trained ML Model
        ↓
Prediction Output
        ↓
Financial Risk Visualization

Why This Pipeline Matters
-------------------------
During deployment, the model must receive inputs that match
the structure used during training.

This pipeline ensures:

• input feature validation
• feature alignment
• safe model inference
• reliable predictions

Label Encoding Reference
------------------------
The data_transformation stage encodes the target as:

    bankruptcy     → 0
    non-bankruptcy → 1

All probability and label logic in this file follows
this mapping explicitly.
"""

# ==========================================================
# 1. IMPORT REQUIRED LIBRARIES
# ==========================================================

import os
import sys
import joblib
import pandas as pd


# ==========================================================
# 2. IMPORT PROJECT UTILITIES
# ==========================================================

from bankruptcy.config.configuration import ConfigurationManager
from bankruptcy.utils.exception import BankruptcyException
from bankruptcy.utils.logger import logger


# ==========================================================
# 3. CONSTANTS — Label Encoding Map
# ==========================================================
#
# Must mirror the mapping in data_transformation.py exactly.
# Centralised here so any future change is made in one place.
#
BANKRUPTCY_LABEL     = 0   # encoded value for "bankruptcy"
NON_BANKRUPTCY_LABEL = 1   # encoded value for "non-bankruptcy"

LABEL_MAP = {
    BANKRUPTCY_LABEL:     "bankruptcy",
    NON_BANKRUPTCY_LABEL: "non-bankruptcy",
}


# ==========================================================
# 4. PREDICTION PIPELINE CLASS
# ==========================================================

class PredictionPipeline:
    """
    Inference pipeline used during model deployment.

    This class loads the trained model and performs
    predictions on new input data.

    Notes
    -----
    The trained artifact is a sklearn Pipeline containing:
        [StandardScaler → classifier]

    feature_names_in_ is available on the Pipeline itself
    (populated by sklearn from the scaler's fit), so we
    read it directly from the top-level pipeline object.
    """

    # ----------------------------------------------------------
    # FIX 1 — lazy config loading
    # ----------------------------------------------------------
    # Original code called ConfigurationManager() in __init__,
    # which crashes at import time if config files are absent
    # (e.g. fresh clone, Docker build, Streamlit Cloud deploy).
    #
    # New design: store the model path as None and resolve it
    # on first call to _load_model(), so the class is safe
    # to instantiate anywhere.
    # ----------------------------------------------------------

    def __init__(self, model_path: str = None):
        """
        Initialise the prediction pipeline.

        Parameters
        ----------
        model_path : str, optional
            Direct path to the trained model .pkl file.
            If None, the path is read from config/config.yaml
            on the first prediction call.
        """
        self._model_path = model_path   # may stay None until first use
        self.model       = None         # loaded lazily


    # ======================================================
    # LAZY MODEL LOADING
    # ======================================================

    def _resolve_model_path(self) -> str:
        """
        Returns the model path, loading it from config if
        not supplied at construction time.
        """
        if self._model_path is not None:
            return self._model_path

        try:
            config_manager   = ConfigurationManager()
            model_config     = config_manager.get_model_trainer_config()
            self._model_path = model_config.trained_model_path
            return self._model_path
        except Exception as e:
            raise BankruptcyException(
                f"Could not resolve model path from config: {e}", sys
            )


    def _load_model(self):
        """
        Load the trained Pipeline from disk.

        The model is loaded once and cached on self.model
        for all subsequent calls (lazy singleton pattern).
        """
        if self.model is not None:
            return

        model_path = self._resolve_model_path()

        if not os.path.exists(model_path):
            raise BankruptcyException(
                f"Model file not found at '{model_path}'. "
                "Run the training pipeline first (python main.py).",
                sys
            )

        self.model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from '{model_path}'")


    # ======================================================
    # INPUT FEATURE VALIDATION
    # ======================================================

    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate input features against the trained model's
        expected feature set and return them in the correct order.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input dataframe from the caller.

        Returns
        -------
        pd.DataFrame
            Validated dataframe with columns in training order.

        Notes
        -----
        sklearn sets feature_names_in_ on the top-level Pipeline
        object (derived from the first step's fit), so we read it
        directly from self.model — not from an inner step.
        """

        # ----------------------------------------------------------
        # FIX 2 — feature_names_in_ lives on the Pipeline object
        # ----------------------------------------------------------
        # Original code raised "Model does not contain feature
        # metadata" for every prediction because it used
        # hasattr(self.model, "feature_names_in_") correctly,
        # but the error message implied it was never set.
        # In practice sklearn Pipelines DO carry this attribute
        # after fitting.  We keep the guard but improve the
        # message and fall back gracefully.
        # ----------------------------------------------------------

        if hasattr(self.model, "feature_names_in_"):
            expected_features = list(self.model.feature_names_in_)
        else:
            # Older sklearn (<1.0) — derive from the scaler step
            scaler = self.model.named_steps.get("scaler")
            if scaler is not None and hasattr(scaler, "feature_names_in_"):
                expected_features = list(scaler.feature_names_in_)
            else:
                # Last resort: accept columns as-is and let the
                # model raise a meaningful error if they are wrong
                logger.warning(
                    "Could not verify feature names — proceeding with "
                    "input columns as provided."
                )
                return df

        missing_features = set(expected_features) - set(df.columns)
        extra_features   = set(df.columns)         - set(expected_features)

        if missing_features:
            raise BankruptcyException(
                f"Missing input features: {sorted(missing_features)}. "
                f"Required features: {expected_features}",
                sys
            )

        if extra_features:
            raise BankruptcyException(
                f"Unexpected input features received: {sorted(extra_features)}. "
                f"Expected only: {expected_features}",
                sys
            )

        # Return columns in training order
        return df[expected_features]


    # ======================================================
    # PREDICTION METHOD
    # ======================================================

    def predict(self, input_data: dict) -> dict:
        """
        Perform bankruptcy risk prediction for a single company.

        Parameters
        ----------
        input_data : dict
            Keys must match the six feature names:
                industrial_risk, management_risk,
                financial_flexibility, credibility,
                competitiveness, operating_risk
            Values should be numeric (float or int).

        Returns
        -------
        dict with keys:
            prediction          : "bankruptcy" | "non-bankruptcy"
            bankruptcy_probability : float in [0, 1]
                Probability that the company goes bankrupt.
                Higher = more risk.
        """

        try:

            # --------------------------------------------------
            # Validate input type
            # --------------------------------------------------

            if not isinstance(input_data, dict):
                raise BankruptcyException(
                    f"input_data must be a dict, got {type(input_data).__name__}.",
                    sys
                )

            # --------------------------------------------------
            # Load model (no-op after first call)
            # --------------------------------------------------

            self._load_model()

            # --------------------------------------------------
            # Build dataframe and validate features
            # --------------------------------------------------

            df = pd.DataFrame([input_data])
            df = self._validate_input(df)

            logger.info(f"Predicting for input: {input_data}")

            # --------------------------------------------------
            # Generate prediction
            # --------------------------------------------------

            prediction_numeric = int(self.model.predict(df)[0])

            # --------------------------------------------------
            # Extract bankruptcy probability
            # --------------------------------------------------
            #
            # FIX 3 — correct class index for bankruptcy probability
            # --------------------------------------------------------
            # Original code used:
            #
            #     bankruptcy_class = max(classes)   →  1  (non-bankruptcy!)
            #
            # Because data_transformation.py encodes:
            #     bankruptcy     → 0
            #     non-bankruptcy → 1
            #
            # max(classes) == 1 == non-bankruptcy.
            # The original code therefore reported the probability of
            # *surviving* as the "bankruptcy probability" — every
            # healthy company showed ~99 % risk, every risky company
            # showed ~1 % risk.  Completely inverted.
            #
            # Fix: use the explicit constant BANKRUPTCY_LABEL = 0
            # and locate its index in model.classes_ directly.
            # --------------------------------------------------------

            bankruptcy_probability = 0.0

            if hasattr(self.model, "predict_proba"):

                probabilities = self.model.predict_proba(df)[0]
                classes       = list(self.model.classes_)

                if BANKRUPTCY_LABEL in classes:
                    # Correct: index of class 0 in the classes array
                    bankruptcy_idx         = classes.index(BANKRUPTCY_LABEL)
                    bankruptcy_probability = float(probabilities[bankruptcy_idx])
                else:
                    # Unexpected class set — log a warning and default to 0
                    logger.warning(
                        f"Expected class {BANKRUPTCY_LABEL} not found in "
                        f"model.classes_ = {classes}. "
                        "bankruptcy_probability will be 0.0."
                    )

            # --------------------------------------------------
            # Convert numeric prediction to human-readable label
            # --------------------------------------------------

            prediction_label = LABEL_MAP.get(
                prediction_numeric,
                f"unknown_class_{prediction_numeric}"
            )

            # --------------------------------------------------
            # Build structured result
            # --------------------------------------------------

            result = {
                "prediction":              prediction_label,
                "bankruptcy_probability":  bankruptcy_probability,
            }

            logger.info(
                f"Prediction: {prediction_label} | "
                f"Bankruptcy probability: {bankruptcy_probability:.4f}"
            )

            return result

        except BankruptcyException:
            # Re-raise our own exceptions without wrapping them again
            raise

        except Exception as e:
            logger.error(f"Prediction pipeline failed: {e}")
            raise BankruptcyException(e, sys)