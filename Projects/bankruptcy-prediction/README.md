<<<<<<< HEAD
# 🚀 Data Science & Machine Learning Portfolio

This repository documents my structured transition into the field of Data Science and Machine Learning, built through rigorous hands-on assignments, analytical exploration, and systematic implementation of core concepts.

The work presented here reflects progressive learning — from Python programming foundations to advanced machine learning, deep learning, and time series modeling — with emphasis on clean coding, statistical reasoning, and model evaluation.

This portfolio serves as a growing technical foundation and will continue expanding with production-oriented projects and real-world case studies.

---

## 🎯 Professional Focus

- Building strong analytical foundations
- Implementing end-to-end machine learning workflows
- Applying statistical reasoning to real-world problems
- Comparing and optimizing predictive models
- Designing structured and reproducible codebases
- Preparing for scalable, project-driven ML development

---

## 🧠 Technical Expertise

### Programming & Foundations
- Python (data-centric programming)
- Object-Oriented Programming
- Modular code design
- Logical problem solving

### Data Analysis & Engineering
- Data cleaning and preprocessing
- Feature engineering
- Handling missing values and outliers
- Encoding and scaling techniques
- Data merging and transformation

### Statistics & Inference
- Descriptive analytics
- Probability fundamentals
- Confidence intervals
- Hypothesis testing
- Statistical decision-making

### Machine Learning

#### Supervised Learning
- Multiple Linear Regression
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machines
- Decision Trees
- Random Forest
- Gradient Boosting (XGBoost, LightGBM)

#### Unsupervised Learning
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- Principal Component Analysis (PCA)

#### Applied Case Implementations
- Healthcare prediction models
- Drug response classification
- Business cost analysis
- Survival prediction (Titanic)
- Recommendation systems

### Model Evaluation & Optimization
- Train-test split and cross-validation
- Hyperparameter tuning (GridSearchCV)
- ROC-AUC analysis
- Precision, Recall, F1-score
- Confusion matrix analysis
- Feature importance interpretation
- Silhouette score evaluation

### Time Series Modeling
- Stationarity testing
- Differencing techniques
- ARIMA/SARIMA forecasting
- Forecast performance evaluation

### Deep Learning
- Artificial Neural Networks (ANN)
- Activation functions and architecture design
- Recurrent Neural Networks (RNN)
- Sequential data modeling
- Natural Language Processing (TF-IDF, tokenization)

---

## 🧰 Technology Stack

**Programming:** Python  
**Data Processing:** NumPy, Pandas, SciPy  
**Visualization:** Matplotlib, Seaborn  
**Machine Learning:** Scikit-learn, Statsmodels, XGBoost, LightGBM  
**Deep Learning:** TensorFlow, Keras  
**Tools:** Git, GitHub, Jupyter Notebook, VS Code  

---

### Databases
- SQL (Joins, Aggregations, Subqueries, Window Functions)

### Business Intelligence & Visualization
- Tableau (Dashboards, Interactive Reports, Data Storytelling)

---

## 📁 Repository Structure

```
data-science-assignments-and-practice/
│
├── Assignments/
│   ├── 1_Basics_of_Python/
│   ├── 2_Data_Structures_Functions_Numpy_Pandas/
│   ├── 3_Basic_Statistics_Descriptive/
│   ├── 4_Basic_Statistics_Inferential/
│   ├── 5_EDA/
│   ├── 6_Hypothesis_Testing_&_CI/
│   ├── 7_Multiple_Linear_Regression/
│   ├── 8_Logistic_Regression/
│   ├── 9_Data_Transformation/
│   ├── 10_SVM_Drug_Response/
│   ├── 11_Decision_Tree_Heart_Disease/
│   ├── 12_Random_Forest_Glass/
│   ├── 13_XGBoost_LightGBM_Titanic/
│   ├── 14_PCA/
│   ├── 15_Clustering/
│   ├── 16_Recommendation_System/
│   ├── 17_Time_Series/
│   ├── 18_Neural_Networks/
│   ├── 19_NLP/
│   └── 20_RNN/
│
├── Practice/
├── Projects/              # Upcoming real-world implementations
├── Datasets/
├── Notes/
├── requirements.txt
=======
# **💼 Bankruptcy Prevention — End-to-End MLOps Pipeline**

A **production-grade Machine Learning system** that predicts company bankruptcy using financial risk indicators.

This project demonstrates a **complete MLOps architecture** including:

* Config-driven pipeline
* Modular ML components
* Artifact-based workflow
* MLflow experiment tracking
* CI integration
* Unit testing
* Advanced statistical analysis
* Streamlit deployment

---

# 📌 Problem Statement

Financial institutions need to identify companies at risk of bankruptcy early.

This project builds a **machine learning pipeline** that predicts bankruptcy using structured risk indicators.

Dataset:

```
bankruptcy-prevention.xlsx
```

Target variable:

```
class → bankruptcy / non-bankruptcy
```

---

# 🧠 System Architecture

The system follows a  **layered ML pipeline architecture** :

```
Raw Data
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
Prediction Pipeline
   ↓
Streamlit Dashboard
```

Each stage produces  **artifacts stored in the `artifacts/` directory** .

---

# 🏗️ Project Structure

```
bankruptcy-prevention-mlops
│
├── .github/workflows
│   └── ci.yml
│
├── artifacts
│   ├── data_ingestion
│   ├── data_validation
│   ├── data_transformation
|   ├── model_evaluation
│   └── model_trainer
│
├── config
│   ├── config.yaml
│   └── schema.yaml
│
├── data
│   ├── raw
│   │   └── bankruptcy-prevention.xlsx
│   ├── processed
│   └── external
│
├── logs
├── mlruns
│
├── notebooks
│   ├── Bankruptcy_Prediction_End_to_End.ipynb
│   └── Enterprise-level-EDA.ipynb
│
├── src
│   └── bankruptcy
│       ├── analysis
│       │   ├── univariate_analysis.py
│       │   ├── bivariate_analysis.py
│       │   ├── multivariate_analysis.py
│       │   ├── advanced_analysis.py
|       |   ├── data_inspection.py
|       ├   └── model_analysis
│       │   	├── model_benchmark.py
│       │   	├── robustness_testing.py
│       │   	├── threshold_optimization.py
│       │   	├── learning_curve_analysis.py
│       │   	├── feature_importance.py
│       │   	└── duplicate_impact.py
│       │
│       ├── components
│       │   ├── data_ingestion.py
│       │   ├── data_validation.py
│       │   ├── data_transformation.py
|       |   ├── model_evaluation.py
│       │   └── model_trainer.py
│       │
│       ├── config
│       │   └── configuration.py
│       │
│       ├── entity
│       │   ├── config_entity.py
│       │   └── artifact_entity.py
│       │
│       ├── pipeline
│       │   ├── training_pipeline.py
│       │   └── prediction_pipeline.py
│       │
│       └── utils
|           ├── common.py
│           ├── logger.py
│           └── exception.py
│
├── tests
│   ├── test_data_ingestion.py
│   ├── test_data_validation.py
│   ├── test_data_transformation.py
│   ├── test_model_trainer.py
│   ├── test_training_pipeline.py
│   ├── test_prediction_pipeline.py
│   ├── test_logger.py
│   └── test_exception.py
│
├── app.py
├── main.py
├── setup.py
├── requirements.txt
├── runtime.txt
├── mlflow.db
├── conftest.py
>>>>>>> ca24b44 (Initial commit - Bankruptcy ML pipeline)
└── README.md
```

---

<<<<<<< HEAD
## 📌 Portfolio Highlights

- Structured progression from fundamentals to advanced ML concepts
- Comparative modeling across multiple algorithms
- Emphasis on interpretability and evaluation
- Exposure to healthcare, business analytics, classification, clustering, forecasting, and recommendation systems
- Clean and modular repository architecture
- Designed to scale with future real-world projects

---

## 📈 Upcoming Expansion

This repository will soon include:

- End-to-end ML projects with problem statements
- Structured ML pipelines
- Performance benchmarking reports
- Reproducible experimentation workflows
- Deployment-oriented implementations

---
=======
# ⚙️ Pipeline Components

## 1️⃣ Data Ingestion

Responsibilities:

* Load raw dataset
* Handle malformed Excel files
* Perform stratified train/test split
* Save dataset artifacts

Artifacts:

```
train.csv
test.csv
```

---

## 2️⃣ Data Validation

Validates dataset against schema:

```
config/schema.yaml
```

Checks:

* column names
* data types
* null values
* dataset structure

---

## 3️⃣ Data Transformation

Responsible for:

* preprocessing
* feature preparation
* dataset formatting

Outputs:

```
X_train.csv
X_test.csv
y_train.csv
y_test.csv
```

---

## 4️⃣ Model Training

Current models:

* Logistic Regression
* Random Forest
* SVM
* KNN
* Gradient Boosting

Training includes:

* cross-validation
* hyperparameter search
* MLflow experiment logging

Artifacts generated:

```
model.pkl
metrics.json
classification_report.txt
```

---

# 📊 Advanced Analysis Toolkit

The project includes **enterprise-level EDA modules** inside:

```
src/bankruptcy/analysis/
```

Modules include:

| Module                  | Purpose                   |
| ----------------------- | ------------------------- |
| univariate_analysis     | feature distribution      |
| bivariate_analysis      | pairwise relationships    |
| multivariate_analysis   | correlation analysis      |
| missing_values_analysis | missing value detection   |
| model_benchmark         | algorithm comparison      |
| learning_curve_analysis | model diagnostics         |
| robustness_testing      | model reliability         |
| threshold_optimization  | decision threshold tuning |

This provides  **deep analytical insights beyond standard ML pipelines** .

---

# 🧪 Testing

The repository includes  **comprehensive unit tests** .

Covered components:

* data ingestion
* validation
* transformation
* training pipeline
* prediction pipeline
* logger
* custom exceptions

Run tests:

```bash
pytest
```

---

# 📈 MLflow Experiment Tracking

All experiments are logged automatically.

Tracking directory:

```
mlruns/
```

Database:

```
mlflow.db
```

Start MLflow UI:

```bash
mlflow ui
```

Open:

```
http://localhost:5000
```

---

# 🚀 Running the Project

## Install dependencies

```bash
pip install -r requirements.txt
```

Install project package:

```bash
pip install -e .
```

---

## Run Training Pipeline

```bash
python main.py
```

---

## Launch Streamlit Dashboard

```bash
streamlit run app.py
```

---

# 📊 Example Prediction Dashboard

The Streamlit app allows users to:

* input financial indicators
* simulate bankruptcy risk
* visualize probability predictions

---

# 🧠 Key Highlights

✔ End-to-end ML pipeline
✔ Modular MLOps architecture
✔ Config-driven system
✔ Artifact-based pipeline design
✔ MLflow experiment tracking
✔ Streamlit deployment
✔ Advanced EDA toolkit
✔ Comprehensive unit testing
✔ CI workflow integration

---

# 👨‍💻 Author

**Aruri Gowtham**

MLOps & Data Science Enthusiast

---

# ⭐ If you find this project useful

Consider **starring the repository** to support the work.
>>>>>>> ca24b44 (Initial commit - Bankruptcy ML pipeline)
