"""
============================================================
Data Ingestion Component
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This module is responsible for loading the raw bankruptcy
dataset, performing basic integrity checks, and splitting
the dataset into training and testing datasets.

Responsibilities
----------------
• Load dataset from Excel file
• Validate dataset integrity
• Handle malformed dataset structures
• Perform dataset quality checks
• Split dataset into train and test sets
• Save processed datasets as pipeline artifacts

Pipeline Stage
--------------

Raw Dataset
     ↓
Data Ingestion
     ↓
Train/Test Split
     ↓
Artifacts Stored

Why This Stage Matters
----------------------
A reliable ML system begins with reliable data ingestion.
If the dataset is corrupted, empty, or malformed, the entire
pipeline would produce unreliable results.

Therefore this stage performs strict dataset validation.
"""

# ==========================================================
# 1. IMPORT STANDARD LIBRARIES
# ==========================================================

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================================
# 2. IMPORT PROJECT CONFIGURATION & ARTIFACT ENTITIES
# ==========================================================

from bankruptcy.entity.config_entity import DataIngestionConfig
from bankruptcy.entity.artifact_entity import DataIngestionArtifact


# ==========================================================
# 3. IMPORT LOGGING & EXCEPTION HANDLING
# ==========================================================

from bankruptcy.utils.logger import logger
from bankruptcy.utils.exception import BankruptcyException


# ==========================================================
# 4. DATA INGESTION CLASS
# ==========================================================

class DataIngestion:
    """
    Data Ingestion pipeline component.

    This class loads the dataset, performs validation checks,
    and splits the dataset into training and testing sets.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initializes Data Ingestion with configuration.

        Parameters
        ----------
        config : DataIngestionConfig
            Configuration object containing ingestion parameters.
        """

        self.config = config


    # ======================================================
    # MAIN DATA INGESTION METHOD
    # ======================================================

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Executes the data ingestion pipeline stage.

        Returns
        -------
        DataIngestionArtifact
            Contains file paths of generated train and test datasets.
        """

        logger.info("Starting Data Ingestion...")

        try:

            # =====================================================
            # 1️⃣ CHECK DATASET EXISTENCE
            # =====================================================

            if not os.path.exists(self.config.data_path):
                raise FileNotFoundError(
                    f"Dataset not found at {self.config.data_path}"
                )

            logger.info("Reading Excel dataset")

            df = pd.read_excel(self.config.data_path)


            # =====================================================
            # 2️⃣ EMPTY DATASET CHECK
            # =====================================================

            if df.empty:
                raise ValueError("Loaded dataset is empty.")


            # =====================================================
            # 3️⃣ HANDLE MALFORMED EXCEL FILE
            # =====================================================
            """
            Some Excel files may contain all values in a single
            column separated by semicolons.

            This block attempts to repair such datasets.
            """

            if df.shape[1] == 1:

                logger.warning(
                    "Detected malformed Excel file. "
                    "Attempting to split semicolon-separated values."
                )

                df = df.iloc[:, 0].astype(str).str.split(";", expand=True)

                if df.shape[1] != 7:
                    raise ValueError(
                        "Malformed dataset could not be parsed correctly."
                    )

                df.columns = [
                    "industrial_risk",
                    "management_risk",
                    "financial_flexibility",
                    "credibility",
                    "competitiveness",
                    "operating_risk",
                    "class",
                ]


            # =====================================================
            # 4️⃣ CLEAN COLUMN NAMES
            # =====================================================

            df.columns = (
                df.columns
                .astype(str)
                .str.strip()
                .str.lower()
            )

            logger.info(f"Columns detected: {df.columns.tolist()}")
            logger.info(f"Dataset shape: {df.shape}")


            # =====================================================
            # 5️⃣ TARGET COLUMN CHECK
            # =====================================================

            if "class" not in df.columns:
                raise ValueError("Target column 'class' not found in dataset.")


            # =====================================================
            # 6️⃣ NULL VALUE CHECK
            # =====================================================

            null_count = df.isnull().sum().sum()

            if null_count > 0:
                raise ValueError(
                    f"Dataset contains {null_count} null values. "
                    "Please clean the dataset."
                )


            # =====================================================
            # 7️⃣ DUPLICATE ANALYSIS
            # =====================================================
            """
            Duplicate rows are detected but NOT removed
            because they may represent valid repeated
            observations in real-world data.
            """

            duplicate_count = df.duplicated().sum()
            duplicate_percentage = (duplicate_count / len(df)) * 100

            if duplicate_count > 0:

                logger.warning(
                    f"Dataset contains {duplicate_count} duplicate rows "
                    f"({duplicate_percentage:.2f}%)."
                )

                logger.warning(
                    "Duplicates detected but preserved to maintain "
                    "original dataset distribution."
                )

                if duplicate_percentage > 50:
                    logger.warning(
                        "Duplicate ratio is extremely high. "
                        "Dataset may contain repeated observations."
                    )


            # =====================================================
            # 8️⃣ CLASS DISTRIBUTION CHECK
            # =====================================================

            class_counts = df["class"].value_counts()

            if class_counts.min() < 2:
                raise ValueError(
                    "One of the classes has fewer than 2 samples. "
                    "Stratified split cannot be performed."
                )

            logger.info(f"Class distribution:\n{class_counts}")


            # =====================================================
            # 9️⃣ CREATE ARTIFACT DIRECTORY
            # =====================================================

            os.makedirs(self.config.root_dir, exist_ok=True)


            # =====================================================
            # 🔟 STRATIFIED TRAIN-TEST SPLIT
            # =====================================================
            """
            Stratified splitting ensures that the proportion
            of classes remains consistent in both training
            and testing datasets.
            """

            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=df["class"],
            )

            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)

            logger.info(f"Train shape: {train_df.shape}")
            logger.info(f"Test shape: {test_df.shape}")


            # =====================================================
            # 11️⃣ SAVE TRAIN AND TEST DATASETS
            # =====================================================

            train_df.to_csv(self.config.train_file_path, index=False)
            test_df.to_csv(self.config.test_file_path, index=False)

            logger.info("Data Ingestion Completed Successfully")


            # =====================================================
            # RETURN ARTIFACT
            # =====================================================

            return DataIngestionArtifact(
                train_file_path=self.config.train_file_path,
                test_file_path=self.config.test_file_path
            )


        except Exception as e:

            logger.error("Error occurred in Data Ingestion")

            raise BankruptcyException(e, sys)