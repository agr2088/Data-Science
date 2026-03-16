"""
============================================================
Data Transformation Component
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This module transforms the validated dataset into a format
that can be directly used for machine learning model training.

Responsibilities
----------------
• Load validated train and test datasets
• Enforce schema-based column ordering
• Separate features and target variable
• Validate target class values
• Encode categorical target labels
• Save transformed datasets as artifacts

Pipeline Stage
--------------

Validated Dataset
        ↓
Data Transformation
        ↓
Feature Matrix (X)
        ↓
Target Vector (y)
        ↓
Model Training

Why This Stage Matters
----------------------
Machine learning models require clean, structured datasets.

This stage ensures that:

• feature columns are correctly selected
• target labels are encoded numerically
• dataset structure matches schema expectations
• model training receives clean inputs
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
import yaml


# ==========================================================
# 3. IMPORT ARTIFACT ENTITIES
# ==========================================================

from bankruptcy.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact
)


# ==========================================================
# 4. IMPORT CONFIGURATION ENTITY
# ==========================================================

from bankruptcy.entity.config_entity import DataTransformationConfig


# ==========================================================
# 5. IMPORT LOGGING & EXCEPTION HANDLING
# ==========================================================

from bankruptcy.utils.logger import logger
from bankruptcy.utils.common import read_yaml
from bankruptcy.utils.exception import BankruptcyException


# ==========================================================
# 6. DATA TRANSFORMATION CLASS
# ==========================================================

class DataTransformation:
    """
    Data Transformation pipeline component.

    This class converts the validated dataset into
    model-ready format by separating features and target
    variables and performing label encoding.
    """

    def __init__(
        self,
        ingestion_artifact: DataIngestionArtifact,
        config: DataTransformationConfig
    ):
        """
        Initializes the Data Transformation component.

        Parameters
        ----------
        ingestion_artifact : DataIngestionArtifact
            Artifact containing paths to train and test datasets.

        config : DataTransformationConfig
            Configuration object containing transformation settings.
        """

        self.ingestion_artifact = ingestion_artifact
        self.config = config


    # ======================================================
    # MAIN TRANSFORMATION METHOD
    # ======================================================

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Executes the data transformation stage.

        Returns
        -------
        DataTransformationArtifact
            Contains file paths of transformed feature and
            target datasets.
        """

        logger.info("Starting Data Transformation")

        try:

            # ==================================================
            # CREATE TRANSFORMATION ARTIFACT DIRECTORY
            # ==================================================

            os.makedirs(self.config.root_dir, exist_ok=True)


            # ==================================================
            # LOAD TRAIN AND TEST DATASETS
            # ==================================================

            train_df = pd.read_csv(self.ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.ingestion_artifact.test_file_path)


            # ==================================================
            # LOAD DATASET SCHEMA
            # ==================================================

            schema = read_yaml(self.config.schema_path)

            target_column = schema["target_column"]

            feature_columns = [
                col for col in schema["columns"].keys()
                if col != target_column
            ]


            # ==================================================
            # ENFORCE COLUMN ORDER
            # ==================================================
            """
            Ensures dataset columns follow the exact order
            defined in the schema. This prevents feature
            misalignment during model training.
            """

            train_df = train_df[feature_columns + [target_column]]
            test_df = test_df[feature_columns + [target_column]]


            # ==================================================
            # SEPARATE FEATURES AND TARGET
            # ==================================================

            X_train = train_df[feature_columns]
            y_train = train_df[target_column]

            X_test = test_df[feature_columns]
            y_test = test_df[target_column]


            # ==================================================
            # VALIDATE TARGET VALUES
            # ==================================================
            """
            Ensures that the target column contains only
            expected class labels.
            """

            allowed_values = {"bankruptcy", "non-bankruptcy"}

            if not set(y_train.unique()).issubset(allowed_values):
                raise ValueError(
                    "Unexpected target values found in training data."
                )

            if not set(y_test.unique()).issubset(allowed_values):
                raise ValueError(
                    "Unexpected target values found in test data."
                )


            # ==================================================
            # TARGET ENCODING
            # ==================================================
            """
            Convert categorical class labels into numeric form.

            Mapping:
                bankruptcy → 0
                non-bankruptcy → 1
            """

            mapping = {
                "bankruptcy": 0,
                "non-bankruptcy": 1
            }

            y_train = y_train.map(mapping)
            y_test = y_test.map(mapping)

            if y_train.isnull().any() or y_test.isnull().any():
                raise ValueError("Target encoding produced NaN values.")

            logger.info("Target encoding completed")


            # ==================================================
            # SAVE TRANSFORMATION ARTIFACTS
            # ==================================================

            X_train.to_csv(
                self.config.transformed_train_path,
                index=False
            )

            X_test.to_csv(
                self.config.transformed_test_path,
                index=False
            )

            y_train.to_csv(
                self.config.transformed_y_train_path,
                index=False
            )

            y_test.to_csv(
                self.config.transformed_y_test_path,
                index=False
            )

            logger.info("Transformation artifacts saved successfully")


            # ==================================================
            # RETURN TRANSFORMATION ARTIFACT
            # ==================================================

            return DataTransformationArtifact(
                transformed_train_file_path=self.config.transformed_train_path,
                transformed_test_file_path=self.config.transformed_test_path,
                transformed_y_train_file_path=self.config.transformed_y_train_path,
                transformed_y_test_file_path=self.config.transformed_y_test_path
            )


        except Exception as e:

            logger.error("Error occurred in Data Transformation")

            raise BankruptcyException(e, sys)