"""
============================================================
Configuration Manager
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This module is responsible for loading and managing all
pipeline configurations from external YAML files.

Configuration files allow us to:

• Avoid hardcoding parameters
• Maintain flexibility in pipeline configuration
• Easily modify paths and settings
• Improve project scalability and maintainability

Configuration Sources
---------------------

config/config.yaml
    Contains pipeline configuration parameters such as:
        • data paths
        • training parameters
        • MLflow tracking settings

config/schema.yaml
    Contains dataset schema definition used for
    validating dataset structure during data validation.

Architecture Role
-----------------

YAML Config Files
       ↓
ConfigurationManager
       ↓
Config Entities
       ↓
Pipeline Components

This ensures a clean separation between configuration
management and pipeline logic.
"""

# ==========================================================
# 1. IMPORT STANDARD LIBRARIES
# ==========================================================

import os
import yaml
import sys


# ==========================================================
# 2. IMPORT CONFIG ENTITY CLASSES
# ==========================================================
"""
Config entities are structured configuration objects
that store configuration parameters for each pipeline stage.

This approach ensures strong typing and better
code maintainability.
"""

from bankruptcy.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)


# ==========================================================
# 3. IMPORT CUSTOM EXCEPTION HANDLING
# ==========================================================

from bankruptcy.utils.exception import BankruptcyException


# ==========================================================
# 4. CONFIGURATION MANAGER CLASS
# ==========================================================

class ConfigurationManager:
    """
    Central configuration loader for the ML pipeline.

    This class loads configuration parameters from YAML files
    and converts them into structured configuration objects
    used by different pipeline components.
    """

    def __init__(
        self,
        config_filepath: str = "config/config.yaml",
        schema_filepath: str = "config/schema.yaml"
    ):
        """
        Initializes configuration manager.

        Parameters
        ----------
        config_filepath : str
            Path to the pipeline configuration YAML file.

        schema_filepath : str
            Path to the dataset schema YAML file.
        """

        try:

            # --------------------------------------------------
            # Validate Config File Existence
            # --------------------------------------------------

            if not os.path.exists(config_filepath):
                raise FileNotFoundError(
                    f"Config file not found at {config_filepath}"
                )

            if not os.path.exists(schema_filepath):
                raise FileNotFoundError(
                    f"Schema file not found at {schema_filepath}"
                )


            # --------------------------------------------------
            # Load Main Configuration File
            # --------------------------------------------------
            from bankruptcy.utils.common import read_yaml

            self.config = read_yaml(config_filepath)


            # --------------------------------------------------
            # Load Schema File
            # --------------------------------------------------

            self.schema = read_yaml(schema_filepath)


            # Store schema path for downstream pipeline stages
            self.schema_filepath = schema_filepath

        except Exception as e:
            raise BankruptcyException(e, sys)


    # ==================================================
    # SAFE CONFIG FETCH
    # ==================================================

    def _get_config_section(self, section_name: str):
        """
        Safely retrieves a specific configuration section.

        Parameters
        ----------
        section_name : str
            Name of the section inside config.yaml.

        Returns
        -------
        dict
            Configuration section dictionary.
        """

        if section_name not in self.config:
            raise KeyError(f"Missing '{section_name}' section in config.yaml")

        return self.config[section_name]


    # ==================================================
    # DATA INGESTION CONFIGURATION
    # ==================================================

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieves configuration for the Data Ingestion stage.

        Returns
        -------
        DataIngestionConfig
            Configuration object containing ingestion parameters.
        """

        config = self._get_config_section("data_ingestion")

        return DataIngestionConfig(
            data_path=config["data_path"],
            root_dir=config["root_dir"],
            train_file_path=config["train_file_path"],
            test_file_path=config["test_file_path"],
            test_size=config["test_size"],
            random_state=config["random_state"]
        )


    # ==================================================
    # DATA VALIDATION CONFIGURATION
    # ==================================================

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Retrieves configuration for the Data Validation stage.

        Returns
        -------
        DataValidationConfig
            Configuration object containing validation parameters.
        """

        config = self._get_config_section("data_validation")

        return DataValidationConfig(
            root_dir=config["root_dir"],
            schema_path=self.schema_filepath,
            status_file_path=config["status_file_path"]
        )


    # ==================================================
    # DATA TRANSFORMATION CONFIGURATION
    # ==================================================

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Retrieves configuration for the Data Transformation stage.

        Returns
        -------
        DataTransformationConfig
            Configuration object containing transformation parameters.
        """

        config = self._get_config_section("data_transformation")

        return DataTransformationConfig(
            root_dir=config["root_dir"],
            transformed_train_path=config["transformed_train_path"],
            transformed_test_path=config["transformed_test_path"],
            transformed_y_train_path=config["transformed_y_train_path"],
            transformed_y_test_path=config["transformed_y_test_path"],
            schema_path=self.schema_filepath
        )


    # ==================================================
    # MODEL TRAINER CONFIGURATION
    # ==================================================

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Retrieves configuration for the Model Training stage.

        Returns
        -------
        ModelTrainerConfig
            Configuration object containing training parameters.
        """

        config = self._get_config_section("model_trainer")

        return ModelTrainerConfig(
            root_dir=config["root_dir"],
            trained_model_path=config["trained_model_path"],
            param_grids=config["param_grids"],
            mlflow_tracking_uri=config["mlflow_tracking_uri"],
            mlflow_experiment_name=config["mlflow_experiment_name"],
        )

    # ==================================================
    # MODEL EVALUATION CONFIGURATION
    # ==================================================

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Retrieves configuration for the Model Evaluation stage.

        Returns
        -------
        ModelEvaluationConfig
            Configuration object containing evaluation artifact paths.
        """

        config = self._get_config_section("model_evaluation")

        return ModelEvaluationConfig(
            root_dir=config["root_dir"],
            metrics_file_path=config["metrics_file_path"],
            report_file_path=config["report_file_path"],
            confusion_matrix_path=config["confusion_matrix_path"],
        )