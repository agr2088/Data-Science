"""
============================================================
Data Validation Component
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This module validates the integrity and structure of the
datasets generated during the Data Ingestion stage.

It ensures that both the training and testing datasets
conform to the predefined dataset schema.

Responsibilities
----------------
• Validate dataset schema
• Check column existence
• Detect missing or extra columns
• Verify column data types
• Detect null values
• Enforce column ordering
• Generate validation status artifacts

Pipeline Stage
--------------

Data Ingestion
     ↓
Data Validation
     ↓
Validated Dataset
     ↓
Data Transformation

Bug Fixes Applied
-----------------

Bug 1 — Exact dtype string comparison breaks on pandas 3.x (CRITICAL)
    Original : str(df[col].dtype) == expected_dtype
               e.g. 'str' == 'object'  →  False

    Problem  : pandas 3.0 changed how string columns report
               their dtype. In pandas < 3.0:
                   df["class"].dtype  → dtype('O')
                   str(...)           → 'object'
               In pandas 3.0+:
                   df["class"].dtype  → dtype('str')  ← StringDtype
                   str(...)           → 'str'

               schema.yaml correctly defines 'class' as 'object'
               (the canonical pandas term for string columns), but
               pandas 3.0 returns 'str' — causing an exact string
               comparison to fail on EVERY run.

               This bug silently blocked the entire training pipeline.
               Every call to python main.py raised:

                   "Train column 'class' dtype mismatch.
                    Expected object, got str"

               The existing artifacts/data_validation/status.txt
               showing "Validation successful" was written by a
               previous run on pandas < 3.0 and was stale.

    Fix      : Compare dtype.kind (a single character code from
               numpy's type system) instead of the string
               representation. dtype.kind is stable across all
               pandas and numpy versions:

                   'f' = floating-point  (float32, float64, Float64)
                   'i' = signed integer  (int32, int64, Int64)
                   'u' = unsigned integer
                   'O' = object / string (object, str, StringDtype)
                   'b' = boolean

               A lookup table maps each schema dtype string to its
               expected kind character, so the check is both version-
               independent and self-documenting.

Bug 2 — Status file never written on validation failure
    Original : The status_file_path was written only on success.
               On failure, the function returned early with
               validation_status=False but left the file absent
               (or stale from a previous successful run).

    Problem  : Any external tool — a CI step, a Streamlit health
               check, a monitoring script — that reads the status
               file to determine whether validation passed would
               interpret "file absent" as "not yet run" rather
               than "failed", masking real failures.

    Fix      : Always write the status file with either
               "Validation successful" or "Validation failed: <reason>"
               before returning, so the file always reflects the
               current run's outcome.
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
import pandas as pd


# ==========================================================
# 3. IMPORT PROJECT ARTIFACT ENTITIES
# ==========================================================

from bankruptcy.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)


# ==========================================================
# 4. IMPORT CONFIGURATION ENTITY
# ==========================================================

from bankruptcy.entity.config_entity import DataValidationConfig


# ==========================================================
# 5. IMPORT LOGGING AND EXCEPTION HANDLING
# ==========================================================

from bankruptcy.utils.logger import logger
from bankruptcy.utils.exception import BankruptcyException
from bankruptcy.utils.common import read_yaml


# ==========================================================
# 6. DTYPE KIND MAP
# ==========================================================
#
# Maps the dtype strings used in schema.yaml to the single-
# character numpy dtype.kind code that is stable across all
# pandas / numpy versions.
#
# dtype.kind reference:
#   'f' → floating-point   (float32, float64, Float64)
#   'i' → signed integer   (int32, int64)
#   'u' → unsigned integer (uint32, uint64)
#   'O' → object / string  (object, str, StringDtype, category)
#   'b' → boolean          (bool, bool_)
#
SCHEMA_DTYPE_TO_KIND: dict[str, str] = {
    # Floating point
    "float16": "f",
    "float32": "f",
    "float64": "f",
    "Float32": "f",   # pandas nullable
    "Float64": "f",   # pandas nullable
    # Signed integers
    "int8":    "i",
    "int16":   "i",
    "int32":   "i",
    "int64":   "i",
    "Int8":    "i",   # pandas nullable
    "Int16":   "i",   # pandas nullable
    "Int32":   "i",   # pandas nullable
    "Int64":   "i",   # pandas nullable
    # Unsigned integers
    "uint8":   "u",
    "uint16":  "u",
    "uint32":  "u",
    "uint64":  "u",
    # Object / string
    "object":  "O",
    "str":     "O",
    "string":  "O",
    # Boolean
    "bool":    "b",
    "bool_":   "b",
    "boolean": "b",
}


# ==========================================================
# 7. DATA VALIDATION CLASS
# ==========================================================

class DataValidation:
    """
    Data Validation pipeline component.

    Validates datasets produced by the Data Ingestion stage
    against the predefined schema before model training.
    """

    def __init__(
        self,
        ingestion_artifact: DataIngestionArtifact,
        config: DataValidationConfig,
    ):
        self.ingestion_artifact = ingestion_artifact
        self.config             = config


    # ======================================================
    # INTERNAL: WRITE STATUS FILE
    # ======================================================

    def _write_status(self, message: str) -> None:
        """
        Write validation outcome to the status file.

        Always called — on both success and failure — so the
        file always reflects the most recent run's result.
        """
        os.makedirs(self.config.root_dir, exist_ok=True)
        with open(self.config.status_file_path, "w", encoding="utf-8") as f:
            f.write(message)


    # ======================================================
    # INTERNAL: COLUMN PRESENCE CHECK
    # ======================================================

    @staticmethod
    def _check_columns(
        df: pd.DataFrame,
        expected_columns: list[str],
        dataset_name: str,
    ) -> tuple[bool, str]:
        """
        Check that the dataset has exactly the expected columns.

        Returns
        -------
        (True, "<name> Passed") on success.
        (False, descriptive message) on failure.
        """
        actual   = set(df.columns.tolist())
        expected = set(expected_columns)

        missing = sorted(expected - actual)
        extra   = sorted(actual   - expected)

        if missing or extra:
            return False, (
                f"{dataset_name} column check failed | "
                f"Missing: {missing} | "
                f"Extra: {extra}"
            )

        return True, f"{dataset_name} column check passed"


    # ======================================================
    # INTERNAL: DTYPE KIND CHECK
    # ======================================================

    @staticmethod
    def _check_dtypes(
        df: pd.DataFrame,
        schema_columns: dict[str, str],
        dataset_name: str,
    ) -> tuple[bool, str]:
        """
        Validate column dtypes using dtype.kind comparison.

        dtype.kind is a single character from numpy's type
        system that is stable across all pandas versions.
        str(dtype) is version-dependent (e.g. 'object' vs
        'str' for string columns in pandas 2.x vs 3.x).

        Parameters
        ----------
        df : pd.DataFrame
        schema_columns : dict[str, str]
            Column → expected dtype string from schema.yaml.
        dataset_name : str

        Returns
        -------
        (True, "<name> Passed") or (False, descriptive message)
        """
        for col, schema_dtype in schema_columns.items():

            if col not in df.columns:
                # Column presence is checked before this — skip
                continue

            expected_kind = SCHEMA_DTYPE_TO_KIND.get(schema_dtype)

            if expected_kind is None:
                # Unknown schema dtype — skip with a warning rather
                # than crashing, so adding a new type to schema.yaml
                # doesn't break validation silently.
                logger.warning(
                    f"Unknown schema dtype '{schema_dtype}' for column "
                    f"'{col}' — dtype check skipped for this column. "
                    f"Add it to SCHEMA_DTYPE_TO_KIND in data_validation.py."
                )
                continue

            actual_kind = df[col].dtype.kind

            if actual_kind != expected_kind:
                return False, (
                    f"{dataset_name} dtype check failed for column '{col}': "
                    f"schema='{schema_dtype}' (kind='{expected_kind}') but "
                    f"actual dtype='{df[col].dtype}' (kind='{actual_kind}'). "
                    f"This is version-independent: kind mismatch means the "
                    f"column contains the wrong numeric family entirely."
                )

        return True, f"{dataset_name} dtype check passed"


    # ======================================================
    # MAIN VALIDATION METHOD
    # ======================================================

    def validate_columns(self) -> DataValidationArtifact:
        """
        Run all validation checks and return a DataValidationArtifact.

        Checks (in order):
            1. Column presence — both train and test
            2. dtype kind      — both train and test
            3. Null values     — both train and test

        The status file is ALWAYS written, whether the
        validation passes or fails.

        Returns
        -------
        DataValidationArtifact
        """

        logger.info("Starting Data Validation")

        try:

            os.makedirs(self.config.root_dir, exist_ok=True)

            # ──────────────────────────────────────────────
            # LOAD DATASETS
            # ──────────────────────────────────────────────

            train_df = pd.read_csv(self.ingestion_artifact.train_file_path)
            test_df  = pd.read_csv(self.ingestion_artifact.test_file_path)

            logger.info(
                f"Loaded train {train_df.shape} / test {test_df.shape}"
            )

            # ──────────────────────────────────────────────
            # LOAD SCHEMA
            # ──────────────────────────────────────────────

            schema          = read_yaml(self.config.schema_path)
            schema_columns  = schema["columns"]          # {col: dtype_str}
            expected_columns = list(schema_columns.keys())


            # ──────────────────────────────────────────────
            # HELPER: return failure artifact + write file
            # ──────────────────────────────────────────────

            def _fail(reason: str) -> DataValidationArtifact:
                status_msg = f"Validation failed: {reason}"
                self._write_status(status_msg)
                logger.error(reason)
                return DataValidationArtifact(
                    validation_status=False,
                    message=reason,
                    status_file_path=self.config.status_file_path,
                )


            # ==================================================
            # CHECK 1: Column presence
            # ==================================================

            ok, msg = self._check_columns(train_df, expected_columns, "Train")
            if not ok:
                return _fail(msg)

            ok, msg = self._check_columns(test_df, expected_columns, "Test")
            if not ok:
                return _fail(msg)

            # Enforce schema column order
            train_df = train_df[expected_columns]
            test_df  = test_df[expected_columns]

            logger.info("Column presence check passed")


            # ==================================================
            # CHECK 2: Dtype kind validation
            # ==================================================
            #
            # Uses dtype.kind (stable across pandas versions)
            # instead of str(dtype) (breaks in pandas 3.x where
            # string columns return 'str' not 'object').
            #
            ok, msg = self._check_dtypes(train_df, schema_columns, "Train")
            if not ok:
                return _fail(msg)

            ok, msg = self._check_dtypes(test_df, schema_columns, "Test")
            if not ok:
                return _fail(msg)

            logger.info("Dtype check passed")


            # ==================================================
            # CHECK 3: Null values
            # ==================================================

            train_nulls = int(train_df.isnull().sum().sum())
            if train_nulls > 0:
                return _fail(
                    f"Train dataset contains {train_nulls} null value(s)."
                )

            test_nulls = int(test_df.isnull().sum().sum())
            if test_nulls > 0:
                return _fail(
                    f"Test dataset contains {test_nulls} null value(s)."
                )

            logger.info("Null value check passed")


            # ==================================================
            # ALL CHECKS PASSED
            # ==================================================

            final_message = "Validation successful"
            self._write_status(final_message)
            logger.info(final_message)

            return DataValidationArtifact(
                validation_status=True,
                message=final_message,
                status_file_path=self.config.status_file_path,
            )

        except Exception as e:
            logger.error(f"Error in Data Validation: {e}")
            raise BankruptcyException(e, sys)