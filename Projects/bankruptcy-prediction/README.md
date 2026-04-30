# ⚖️ Bankruptcy Risk Intelligence System

An end-to-end MLOps pipeline for predicting corporate bankruptcy risk, featuring automated model training, experiment tracking with MLflow, and an interactive Streamlit dashboard.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Architecture](#pipeline-architecture)
- [ML Models](#ml-models)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Experiment Tracking](#experiment-tracking)
- [Analysis Modules](#analysis-modules)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Configuration](#configuration)
- [License](#license)

---

## Overview

The Bankruptcy Risk Intelligence System is a production-ready machine learning pipeline that classifies companies as **bankruptcy** or **non-bankruptcy** risk based on six financial and operational indicators. The system follows MLOps best practices — modular components, YAML-driven configuration, schema validation, MLflow experiment tracking, and a polished Streamlit UI for real-time inference.

---

## Features

- **Automated ML Pipeline** — end-to-end training from raw Excel data to a saved model artifact
- **Multi-Model Training** — trains and compares Logistic Regression, Random Forest, SVM, KNN, and Gradient Boosting
- **Hyperparameter Tuning** — GridSearchCV with Repeated Stratified K-Fold cross-validation
- **MLflow Tracking** — logs parameters, metrics, and model artifacts to a local SQLite backend
- **Schema Validation** — YAML-defined data schema enforced before training begins
- **Streamlit Dashboard** — four-page interactive app with animated risk gauge, EDA explorer, model metrics, and MLflow run comparison
- **SHAP Explainability** — feature importance analysis via SHAP values
- **Comprehensive Tests** — pytest suite covering all pipeline stages
- **GitHub Actions CI** — automated test runs on every push and pull request

---

## Project Structure

```
bankruptcy-prediction/
├── app.py                          # Streamlit dashboard (4 pages)
├── main.py                         # Pipeline entry point
├── setup.py                        # Package installation config
├── requirements.txt                # Python dependencies
│
├── config/
│   ├── config.yaml                 # Central pipeline configuration
│   └── schema.yaml                 # Dataset schema definition
│
├── data/
│   └── raw/
│       └── bankruptcy-prevention.xlsx   # Source dataset
│
├── notebooks/
│   └── advanced_eda.ipynb          # Exploratory data analysis notebook
│
├── src/bankruptcy/
│   ├── analysis/                   # EDA & model analysis modules
│   │   ├── univariate_analysis.py
│   │   ├── bivariate_analysis.py
│   │   ├── multivariate_analysis.py
│   │   ├── advanced_analysis.py
│   │   ├── data_inspection.py
│   │   └── model_analysis/
│   │       ├── feature_importance.py
│   │       ├── learning_curve_analysis.py
│   │       ├── model_benchmark.py
│   │       ├── robustness_testing.py
│   │       ├── threshold_optimization.py
│   │       └── duplicate_impact.py
│   │
│   ├── components/                 # Pipeline stage implementations
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   │
│   ├── config/
│   │   └── configuration.py        # ConfigurationManager
│   │
│   ├── entity/
│   │   ├── config_entity.py        # Dataclasses for stage configs
│   │   └── artifact_entity.py      # Dataclasses for stage outputs
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py    # Orchestrates full training run
│   │   └── prediction_pipeline.py  # Inference pipeline for the UI
│   │
│   └── utils/
│       ├── common.py
│       ├── logger.py
│       └── exception.py
│
├── tests/                          # pytest test suite
│   ├── test_config.py
│   ├── test_data_ingestion.py
│   ├── test_data_transformation.py
│   ├── test_data_validation.py
│   ├── test_model_trainer.py
│   ├── test_prediction_pipeline.py
│   ├── test_training_pipeline.py
│   ├── test_exception.py
│   └── test_logger.py
│
└── .github/workflows/
    └── ci.yml                      # GitHub Actions CI pipeline
```

---

## Dataset

**File:** `data/raw/bankruptcy-prevention.xlsx`

The dataset contains six financial risk indicators and a target label:

| Feature | Type | Description |
|---|---|---|
| `industrial_risk` | float64 | Industrial environment risk level |
| `management_risk` | float64 | Management quality risk indicator |
| `financial_flexibility` | float64 | Financial flexibility of the company |
| `credibility` | float64 | Company credibility score |
| `competitiveness` | float64 | Market competitiveness indicator |
| `operating_risk` | float64 | Operational risk exposure |
| `class` *(target)* | object | `bankruptcy` or `non-bankruptcy` |

The target is label-encoded during transformation: `bankruptcy → 0`, `non-bankruptcy → 1`.

---

## Installation

**Prerequisites:** Python ≥ 3.9

```bash
# 1. Clone the repository
git clone https://github.com/your-username/bankruptcy-prediction.git
cd bankruptcy-prediction

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the project as a package (enables internal imports)
pip install -e .
```

---

## Usage

### Run the Training Pipeline

```bash
python main.py
```

This executes the full ML pipeline in sequence:

1. Data Ingestion → `artifacts/data_ingestion/`
2. Data Validation → `artifacts/data_validation/status.txt`
3. Data Transformation → `artifacts/data_transformation/`
4. Model Training → `artifacts/model_trainer/model.pkl`
5. Model Evaluation → `artifacts/model_evaluation/`

### Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

Run this from the project root directory so that configuration paths resolve correctly.

---

## Pipeline Architecture

```
Raw Dataset (Excel)
        │
        ▼
┌─────────────────┐
│  Data Ingestion  │  Load, validate integrity, train/test split (80/20)
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│  Data Validation  │  Enforce schema (column names + dtypes)
└────────┬─────────┘
         │
         ▼
┌────────────────────────┐
│  Data Transformation    │  Label encode target, prepare feature matrices
└────────┬───────────────┘
         │
         ▼
┌─────────────────┐
│  Model Trainer  │  GridSearchCV + Repeated Stratified K-Fold across 5 models
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│  Model Evaluation │  Accuracy, AUC, F1, MCC, confusion matrix, class report
└────────┬─────────┘
         │
         ▼
   model.pkl artifact  +  MLflow experiment log
```

Each stage reads its settings from `config/config.yaml` via `ConfigurationManager` and writes typed artifact dataclasses to disk.

---

## ML Models

Five classifiers are trained and compared. The best model by cross-validated score is saved.

| Model | Hyperparameters Tuned |
|---|---|
| Logistic Regression | `C` |
| Random Forest | `n_estimators`, `max_depth` |
| Support Vector Machine | `C`, `kernel`, `gamma` |
| K-Nearest Neighbors | `n_neighbors` |
| Gradient Boosting | `n_estimators`, `learning_rate` |

**Cross-validation strategy:** Repeated Stratified K-Fold (5 splits × 5 repeats).

**Evaluation metrics:** Accuracy, ROC-AUC, F1-Score, Matthews Correlation Coefficient (MCC), confusion matrix.

---

## Streamlit Dashboard

The `app.py` dashboard has four pages:

| Page | Description |
|---|---|
| **Risk Predictor** | Adjust six risk sliders and get an instant prediction with an animated gauge and probability bar |
| **Model Performance** | View accuracy, AUC, F1, MCC, confusion matrix, and full classification report |
| **EDA Explorer** | Explore feature distributions, class balance, and correlation heatmaps |
| **MLflow Tracker** | Compare all experiment runs in a sortable table |

---

## Experiment Tracking

MLflow is configured with a local SQLite backend:

```yaml
mlflow_tracking_uri: "sqlite:///mlflow.db"
mlflow_experiment_name: "Bankruptcy_Prediction"
```

To browse experiments in the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://localhost:5000` in your browser.

---

## Analysis Modules

The `src/bankruptcy/analysis/` package provides standalone analytical utilities used in the EDA notebook and dashboard:

- **`univariate_analysis.py`** — distributions and summary statistics per feature
- **`bivariate_analysis.py`** — feature-vs-target relationships
- **`multivariate_analysis.py`** — correlation matrices and pair plots
- **`advanced_analysis.py`** — SHAP-based explainability and advanced visualizations
- **`data_inspection.py`** — dataset quality checks and anomaly detection
- **`model_analysis/feature_importance.py`** — permutation and SHAP feature importance
- **`model_analysis/learning_curve_analysis.py`** — bias-variance diagnostics
- **`model_analysis/model_benchmark.py`** — side-by-side model comparison
- **`model_analysis/robustness_testing.py`** — performance under distribution shift
- **`model_analysis/threshold_optimization.py`** — classification threshold tuning
- **`model_analysis/duplicate_impact.py`** — effect of duplicate records on metrics

---

## Testing

The test suite covers all major pipeline stages and utilities.

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_model_trainer.py
```

Test files:

| File | Covers |
|---|---|
| `test_config.py` | ConfigurationManager |
| `test_data_ingestion.py` | DataIngestion component |
| `test_data_validation.py` | DataValidation component |
| `test_data_transformation.py` | DataTransformation component |
| `test_model_trainer.py` | ModelTrainer component |
| `test_prediction_pipeline.py` | Inference pipeline |
| `test_training_pipeline.py` | Full training pipeline integration |
| `test_exception.py` | Custom exception handling |
| `test_logger.py` | Logger utility |

---

## CI/CD

GitHub Actions runs the test suite automatically on every push and pull request to `main`.

**Workflow:** `.github/workflows/ci.yml`

Steps:
1. Checkout repository
2. Set up Python 3.11
3. Install `requirements.txt`, `pytest`, `mlflow`, and the project package (`pip install -e .`)
4. Run `pytest`

---

## Configuration

All pipeline parameters are defined in `config/config.yaml`. Key settings:

```yaml
data_ingestion:
  data_path: data/raw/bankruptcy-prevention.xlsx
  test_size: 0.2
  random_state: 42

model_trainer:
  mlflow_tracking_uri: "sqlite:///mlflow.db"
  mlflow_experiment_name: "Bankruptcy_Prediction"
  trained_model_path: artifacts/model_trainer/model.pkl
```

The dataset schema (expected columns and dtypes) is defined in `config/schema.yaml` and enforced at the Data Validation stage.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

**Author:** Aruri Gowtham — [arurigowthamraj@gmail.com](mailto:arurigowthamraj@gmail.com)
