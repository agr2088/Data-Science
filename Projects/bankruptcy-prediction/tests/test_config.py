"""
============================================================
Configuration Manager Tests
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This test module verifies that the ConfigurationManager
correctly loads and returns configuration objects used
throughout the machine learning pipeline.

Why Testing Configuration Matters
---------------------------------
Configuration files control the behavior of the entire
ML pipeline.

If configuration loading fails, the following stages
will break:

Data Ingestion
      ↓
Data Validation
      ↓
Data Transformation
      ↓
Model Training

Therefore, this test ensures that the configuration
system works correctly before running the pipeline.
"""

# ==========================================================
# IMPORT REQUIRED MODULES
# ==========================================================

from bankruptcy.config.configuration import ConfigurationManager


# ==========================================================
# TEST: CONFIGURATION LOADING
# ==========================================================

def test_configuration_returns_valid_configs():
    """
    Test that ConfigurationManager returns valid
    configuration objects for pipeline stages.

    This test verifies:

    • configuration manager initializes successfully
    • data ingestion configuration exists
    • model trainer configuration exists
    • required attributes are present
    """

    # Initialize configuration manager
    config = ConfigurationManager()


    # Retrieve pipeline configurations
    ingestion_config = config.get_data_ingestion_config()
    model_config = config.get_model_trainer_config()


    # ------------------------------------------------------
    # Basic existence checks
    # ------------------------------------------------------

    assert ingestion_config is not None
    assert model_config is not None


    # ------------------------------------------------------
    # Validate ingestion configuration attributes
    # ------------------------------------------------------

    assert hasattr(ingestion_config, "train_file_path")
    assert hasattr(ingestion_config, "test_file_path")
    assert hasattr(ingestion_config, "test_size")
    assert hasattr(ingestion_config, "random_state")


    # ------------------------------------------------------
    # Validate model trainer configuration attributes
    # ------------------------------------------------------

    assert hasattr(model_config, "trained_model_path")
    assert hasattr(model_config, "param_grids")