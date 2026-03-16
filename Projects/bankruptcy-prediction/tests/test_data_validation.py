"""
============================================================
Data Validation Tests
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This test module verifies that the Data Validation
component correctly validates the dataset produced
by the Data Ingestion stage.

The validation stage ensures that the dataset conforms
to the schema definition before proceeding further
in the ML pipeline.

Why This Test Matters
---------------------
Data validation acts as a quality gate.

If validation fails, the pipeline should stop before
model training to prevent:

• incorrect feature structures
• missing columns
• invalid data types
• corrupted datasets
"""

# ==========================================================
# IMPORT REQUIRED MODULES
# ==========================================================

import os

from bankruptcy.config.configuration import ConfigurationManager
from bankruptcy.components.data_ingestion import DataIngestion
from bankruptcy.components.data_validation import DataValidation


# ==========================================================
# TEST: DATA VALIDATION PIPELINE
# ==========================================================

def test_data_validation_passes():
    """
    Verify that the validation pipeline:

    • runs successfully
    • validates dataset structure
    • produces validation artifact
    • records validation success
    """

    config_manager = ConfigurationManager()


    # ======================================================
    # 1️⃣ RUN DATA INGESTION
    # ======================================================

    ingestion_config = config_manager.get_data_ingestion_config()

    ingestion = DataIngestion(ingestion_config)

    ingestion_artifact = ingestion.initiate_data_ingestion()


    # ======================================================
    # 2️⃣ RUN DATA VALIDATION
    # ======================================================

    validation_config = config_manager.get_data_validation_config()

    validation = DataValidation(
        ingestion_artifact,
        validation_config
    )

    validation_artifact = validation.validate_columns()


    # ======================================================
    # 3️⃣ VALIDATION MUST PASS
    # ======================================================

    assert validation_artifact.validation_status is True


    # ======================================================
    # 4️⃣ VALIDATION STATUS FILE MUST EXIST
    # ======================================================

    assert os.path.exists(validation_artifact.status_file_path)


    # ======================================================
    # 5️⃣ VALIDATION MESSAGE MUST BE RECORDED
    # ======================================================

    with open(validation_artifact.status_file_path, "r") as f:

        content = f.read()

    assert "Validation successful" in content