"""
============================================================
Configuration Entities
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This module defines structured configuration objects used
throughout the Machine Learning pipeline.

Instead of passing raw dictionaries from configuration files,
we use dataclasses to create strongly-typed configuration
objects.

Benefits
--------
• Improves code readability
• Enforces structured configuration
• Reduces configuration errors
• Makes pipeline components cleaner

Architecture Role
-----------------

config.yaml
      ↓
ConfigurationManager
      ↓
Config Entities (This File)
      ↓
Pipeline Components

Each pipeline stage receives its own configuration object.
"""

# ==========================================================
# 1. IMPORT REQUIRED LIBRARIES
# ==========================================================

from dataclasses import dataclass
from typing import Dict


# ==========================================================
# 2. DATA INGESTION CONFIGURATION ENTITY
# ==========================================================

@dataclass
class DataIngestionConfig:
    """
    Configuration object for the Data Ingestion stage.

    Attributes
    ----------
    data_path : str
        Path to the raw bankruptcy dataset.

    root_dir : str
        Directory where ingestion artifacts will be stored.

    train_file_path : str
        File path where the training dataset will be saved.

    test_file_path : str
        File path where the testing dataset will be saved.

    test_size : float
        Fraction of dataset used for testing split.

    random_state : int
        Random seed used to ensure reproducible dataset splits.
    """

    data_path: str
    root_dir: str
    train_file_path: str
    test_file_path: str
    test_size: float
    random_state: int


# ==========================================================
# 3. DATA VALIDATION CONFIGURATION ENTITY
# ==========================================================

@dataclass
class DataValidationConfig:
    """
    Configuration object for the Data Validation stage.

    Attributes
    ----------
    root_dir : str
        Directory where validation artifacts will be stored.

    schema_path : str
        Path to the dataset schema YAML file used for
        validating dataset structure.

    status_file_path : str
        File path where validation status results are stored.
    """

    root_dir: str
    schema_path: str
    status_file_path: str


# ==========================================================
# 4. DATA TRANSFORMATION CONFIGURATION ENTITY
# ==========================================================

@dataclass
class DataTransformationConfig:
    """
    Configuration object for the Data Transformation stage.

    Attributes
    ----------
    root_dir : str
        Directory where transformation artifacts will be stored.

    transformed_train_path : str
        File path for transformed training feature dataset.

    transformed_test_path : str
        File path for transformed testing feature dataset.

    transformed_y_train_path : str
        File path for transformed training target labels.

    transformed_y_test_path : str
        File path for transformed testing target labels.

    schema_path : str
        Path to the dataset schema used for feature selection
        and validation.
    """

    root_dir: str
    transformed_train_path: str
    transformed_test_path: str
    transformed_y_train_path: str
    transformed_y_test_path: str
    schema_path: str


# ==========================================================
# 5. MODEL TRAINER CONFIGURATION ENTITY
# ==========================================================

@dataclass
class ModelTrainerConfig:
    """
    Configuration object for the Model Training stage.

    Attributes
    ----------
    root_dir : str
        Directory where training artifacts and models
        will be stored.

    trained_model_path : str
        File path where the trained model will be saved.

    param_grids : Dict
        Dictionary containing hyperparameter search space
        for different machine learning models.

    mlflow_tracking_uri : str
        URI used by MLflow to track experiment runs.

    mlflow_experiment_name : str
        Name of the MLflow experiment used for tracking
        model performance and training metadata.
    """

    root_dir: str
    trained_model_path: str
    param_grids: Dict
    mlflow_tracking_uri: str
    mlflow_experiment_name: str


# ==========================================================
# 6. MODEL EVALUATION CONFIGURATION ENTITY  ← NEW
# ==========================================================

@dataclass
class ModelEvaluationConfig:
    """
    Configuration object for the Model Evaluation stage.

    Previously this stage had no config entity, so the
    artifact directory was hardcoded as the string literal
    "artifacts/model_evaluation" inside the component.
    That breaks whenever the working directory is not the
    project root (CI, Docker, subprocess calls).

    Attributes
    ----------
    root_dir : str
        Directory where evaluation artifacts are written.
        Read from config.yaml → model_evaluation.root_dir.

    metrics_file_path : str
        Full path for the JSON file containing all scalar
        evaluation metrics.

    report_file_path : str
        Full path for the text classification report.

    confusion_matrix_path : str
        Full path for the JSON confusion matrix artifact.
    """

    root_dir: str
    metrics_file_path: str
    report_file_path: str
    confusion_matrix_path: str