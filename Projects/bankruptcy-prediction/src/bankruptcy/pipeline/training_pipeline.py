"""
============================================================
Training Pipeline — Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This module orchestrates the entire Machine Learning pipeline
for the Bankruptcy Prediction System.

It sequentially executes the following stages:

    1. Data Ingestion
    2. Data Validation
    3. Data Transformation
    4. Model Training
    5. Model Evaluation

Why This Pipeline Exists
------------------------
• Maintain a clean and modular ML architecture
• Ensure reproducible pipeline execution
• Centralize pipeline control in one place
• Allow easy debugging and monitoring

Pipeline Architecture
---------------------

Raw Dataset
     ↓
Data Ingestion
     ↓
Data Validation
     ↓
Data Transformation
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Final Model Artifact
"""

# ==========================================================
# 1. IMPORT STANDARD LIBRARIES
# ==========================================================

import sys


# ==========================================================
# 2. IMPORT PROJECT CONFIGURATION MANAGER
# ==========================================================
"""
ConfigurationManager is responsible for loading pipeline
configuration settings from YAML files.

Each pipeline stage receives its configuration from here.
"""

from bankruptcy.config.configuration import ConfigurationManager


# ==========================================================
# 3. IMPORT PIPELINE COMPONENTS
# ==========================================================
"""
Each component represents a stage in the ML pipeline.
"""

from bankruptcy.components.data_ingestion import DataIngestion
from bankruptcy.components.data_validation import DataValidation
from bankruptcy.components.data_transformation import DataTransformation
from bankruptcy.components.model_trainer import ModelTrainer
from bankruptcy.components.model_evaluation import ModelEvaluation


# ==========================================================
# 4. IMPORT LOGGING & EXCEPTION HANDLING
# ==========================================================

from bankruptcy.utils.logger import logger
from bankruptcy.utils.exception import BankruptcyException


# ==========================================================
# 5. TRAINING PIPELINE CLASS
# ==========================================================

class TrainingPipeline:
    """
    Central controller of the entire ML pipeline.

    This class coordinates all pipeline stages and ensures
    they execute sequentially.

    Attributes
    ----------
    config_manager : ConfigurationManager
        Responsible for loading pipeline configuration.
    """

    def __init__(self):
        """
        Initializes the configuration manager that provides
        configuration objects for each pipeline stage.
        """
        self.config_manager = ConfigurationManager()


    # ======================================================
    # PIPELINE EXECUTION METHOD
    # ======================================================

    def run_pipeline(self):
        """
        Executes the complete ML training pipeline.

        Returns
        -------
        model_artifact
            Contains trained model and performance metrics.
        """

        logger.info("=" * 60)
        logger.info("Training Pipeline Started")
        logger.info("=" * 60)

        try:

            # =====================================================
            # 1️⃣ DATA INGESTION
            # =====================================================
            """
            Loads the raw dataset and prepares training and
            testing splits.
            """

            logger.info("Stage 1: Data Ingestion")

            ingestion_config = self.config_manager.get_data_ingestion_config()

            data_ingestion = DataIngestion(ingestion_config)

            ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logger.info("Data Ingestion Completed Successfully")


            # =====================================================
            # 2️⃣ DATA VALIDATION
            # =====================================================
            """
            Ensures dataset integrity by validating schema,
            column structure, and dataset consistency.
            """

            logger.info("Stage 2: Data Validation")

            validation_config = self.config_manager.get_data_validation_config()

            data_validation = DataValidation(
                ingestion_artifact,
                validation_config
            )

            validation_artifact = data_validation.validate_columns()

            if not validation_artifact.validation_status:

                logger.error(
                    f"Data Validation Failed: {validation_artifact.message}"
                )

                raise BankruptcyException(
                    f"Validation Failed: {validation_artifact.message}",
                    sys
                )

            logger.info("Data Validation Completed Successfully")


            # =====================================================
            # 3️⃣ DATA TRANSFORMATION
            # =====================================================
            """
            Applies preprocessing such as feature scaling,
            encoding, and preparing datasets for model training.
            """

            logger.info("Stage 3: Data Transformation")

            transformation_config = (
                self.config_manager.get_data_transformation_config()
            )

            data_transformation = DataTransformation(
                ingestion_artifact,
                transformation_config
            )

            transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )

            logger.info("Data Transformation Completed Successfully")


            # =====================================================
            # 4️⃣ MODEL TRAINING
            # =====================================================
            """
            Trains multiple ML models and selects the best
            performing model based on evaluation metrics.
            """

            logger.info("Stage 4: Model Training")

            model_config = self.config_manager.get_model_trainer_config()

            model_trainer = ModelTrainer(
                transformation_artifact,
                model_config
            )

            model_artifact = model_trainer.initiate_model_training()

            logger.info(
                f"Best Model Performance | "
                f"Accuracy: {model_artifact.test_accuracy:.4f} | "
                f"Test AUC: {model_artifact.test_auc:.4f} | "
                f"CV AUC: {model_artifact.cv_auc:.4f}"
            )


            # =====================================================
            # 5️⃣ MODEL EVALUATION
            # =====================================================
            """
            Evaluates the trained model on unseen data and
            calculates final evaluation metrics.
            """

            logger.info("Stage 5: Model Evaluation")

            evaluation_config = self.config_manager.get_model_evaluation_config()

            model_evaluation = ModelEvaluation(
                transformation_artifact,
                model_artifact,
                evaluation_config,
            )

            evaluation_metrics = model_evaluation.initiate_model_evaluation()

            logger.info(
                f"Evaluation Results | "
                f"Accuracy: {evaluation_metrics['accuracy']:.4f} | "
                f"AUC: {evaluation_metrics['roc_auc']:.4f} | "
                f"MCC: {evaluation_metrics['matthews_corrcoef']:.4f} | "
                f"FNR: {evaluation_metrics['false_negative_rate']:.4f} | "
                f"FPR: {evaluation_metrics['false_positive_rate']:.4f}"
            )

            logger.info("Model Evaluation Completed Successfully")


            # =====================================================
            # PIPELINE COMPLETION
            # =====================================================

            logger.info("=" * 60)
            logger.info("Training Pipeline Completed Successfully")
            logger.info("=" * 60)

            return model_artifact


        except Exception as e:

            logger.error("Error occurred in Training Pipeline")

            raise BankruptcyException(e, sys)