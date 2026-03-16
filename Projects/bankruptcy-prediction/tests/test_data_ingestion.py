"""
============================================================
Data Ingestion Tests
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This test module verifies that the Data Ingestion component
correctly loads the dataset and produces valid train/test
splits.

Why This Test Matters
---------------------
Data ingestion is the first stage of the ML pipeline.

If ingestion fails, all downstream stages will fail:

Data Ingestion
      ↓
Data Validation
      ↓
Data Transformation
      ↓
Model Training

This test ensures that the ingestion stage produces
valid and usable datasets.
"""

# ==========================================================
# IMPORT REQUIRED MODULES
# ==========================================================

import os
import pandas as pd

from bankruptcy.config.configuration import ConfigurationManager
from bankruptcy.components.data_ingestion import DataIngestion


# ==========================================================
# TEST: DATA INGESTION PIPELINE
# ==========================================================

def test_data_ingestion_creates_valid_train_test_split():
    """
    Verify that Data Ingestion:

    • creates train and test datasets
    • saves artifact files correctly
    • preserves target column
    • performs stratified splitting
    """

    # ------------------------------------------------------
    # Load ingestion configuration
    # ------------------------------------------------------

    config_manager = ConfigurationManager()

    ingestion_config = config_manager.get_data_ingestion_config()


    # ------------------------------------------------------
    # Run data ingestion pipeline
    # ------------------------------------------------------

    ingestion = DataIngestion(ingestion_config)

    artifact = ingestion.initiate_data_ingestion()


    # ======================================================
    # 1️⃣ VERIFY ARTIFACT FILES EXIST
    # ======================================================

    assert os.path.exists(artifact.train_file_path)

    assert os.path.exists(artifact.test_file_path)


    # ======================================================
    # 2️⃣ VERIFY DATASETS ARE NOT EMPTY
    # ======================================================

    train_df = pd.read_csv(artifact.train_file_path)

    test_df = pd.read_csv(artifact.test_file_path)

    assert len(train_df) > 0

    assert len(test_df) > 0


    # ======================================================
    # 3️⃣ VERIFY TARGET COLUMN EXISTS
    # ======================================================

    assert "class" in train_df.columns

    assert "class" in test_df.columns


    # ======================================================
    # 4️⃣ BASIC SHAPE VALIDATION
    # ======================================================

    assert train_df.shape[0] > 0

    assert test_df.shape[0] > 0


    # ======================================================
    # 5️⃣ STRATIFICATION VALIDATION
    # ======================================================
    """
    Ensure that class distribution is preserved between
    train and test datasets.

    Since stratified splitting is used, the class ratios
    should be approximately equal.
    """

    train_distribution = train_df["class"].value_counts(normalize=True)

    test_distribution = test_df["class"].value_counts(normalize=True)

    # Allow small tolerance for sampling variation
    for label in train_distribution.index:

        assert abs(
            train_distribution[label] - test_distribution[label]
        ) < 0.10