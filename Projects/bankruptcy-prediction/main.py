"""
============================================================
Main Entry Point — Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This file serves as the main execution entry point for the
Bankruptcy Prediction Machine Learning Pipeline.

It triggers the complete ML workflow including:

    1. Data Ingestion
    2. Data Validation
    3. Data Transformation
    4. Model Training
    5. Model Evaluation
    6. Model Persistence

Why This File Exists
--------------------
• Provides a single command to run the entire pipeline
• Keeps execution logic separated from pipeline implementation
• Ensures modular and scalable project architecture

Pipeline Flow
-------------
main.py
   ↓
TrainingPipeline
   ↓
Data Ingestion
   ↓
Data Validation
   ↓
Data Transformation
   ↓
Model Training
   ↓
Evaluation & Artifacts
"""

# ==========================================================
# 1. IMPORT STANDARD LIBRARIES
# ==========================================================

import os
import sys


# ==========================================================
# 2. ADD PROJECT SOURCE DIRECTORY TO PYTHON PATH
# ==========================================================
"""
Python needs to know where to find our custom modules.

Since the project source code is located inside the `src`
directory, we manually add it to the system path so Python
can import our internal modules like:

    bankruptcy.pipeline.training_pipeline
"""

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


# ==========================================================
# 3. IMPORT TRAINING PIPELINE
# ==========================================================
"""
TrainingPipeline is the orchestrator of the entire ML workflow.

It sequentially executes all pipeline stages such as:

• Data ingestion
• Data validation
• Data transformation
• Model training
"""

from bankruptcy.pipeline.training_pipeline import TrainingPipeline


# ==========================================================
# 4. MAIN EXECUTION BLOCK
# ==========================================================
"""
This condition ensures the pipeline runs ONLY when this file
is executed directly.

Example:
    python main.py

If this file is imported as a module somewhere else, the
pipeline will NOT execute automatically.
"""

if __name__ == "__main__":

    # ------------------------------------------------------
    # Initialize Training Pipeline
    # ------------------------------------------------------
    """
    Creates an instance of the TrainingPipeline class,
    which controls the entire ML workflow.
    """

    pipeline = TrainingPipeline()


    # ------------------------------------------------------
    # Execute the Pipeline
    # ------------------------------------------------------
    """
    Starts execution of the complete ML pipeline including
    data processing and model training.
    """

    print("🚀 Starting Bankruptcy Prediction Pipeline...")
    pipeline.run_pipeline()
    print("✅ Pipeline Execution Completed Successfully!")