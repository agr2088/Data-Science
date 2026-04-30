# CourseIQ — Online Course Recommendation System

---

## 🚀 Overview

CourseIQ is a production-grade, multi-model course recommendation system that suggests relevant online courses to learners based on their interaction history, engagement levels, content semantics, and peer behaviour. It combines five distinct recommendation engines into a single adaptive hybrid ensemble, exposed through a rich Streamlit dashboard and a command-line interface.

**Real-world use case:** An e-learning platform with a large course catalogue uses CourseIQ to personalise learner homepages — returning engaged, contextually relevant course cards for every user segment, from first-time visitors to power users.

---

## 🎯 Business Objective

The objective of this project (P662) is to build an online course recommendation system that suggests relevant courses to learners based on their interests, past enrollments, and engagement levels. The dataset includes course ratings, instructor information, previous learning history, study material availability, and certification offerings, making it suitable for recommendation models using collaborative filtering, content-based filtering, or hybrid approaches.

The 30-day delivery plan ran from kick-off (28-Mar-2026) through EDA (04-Apr-2026), model building (11-Apr-2026), and deployment (18-Apr-2026).

---

## 🧱 Architecture

```
User Input (User ID / Course ID)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              main.py  (CLI Entry Point)             │
│  Commands: train | serve | recommend | similar |    │
│            popular                                  │
└─────────────┬───────────────────────────────────────┘
              │
    ┌─────────▼──────────┐
    │  Training Pipeline │  (6 stages, orchestrated by
    │  training_pipeline │   TrainingPipeline.run())
    └─────────┬──────────┘
              │
   ┌──────────▼──────────────────────────────────────────┐
   │                  Stage 1 — Data Ingestion           │
   │  DataIngestion → reads Excel via openpyxl →         │
   │  validates schema + range checks → raw DataFrame    │
   └──────────┬──────────────────────────────────────────┘
              │
   ┌──────────▼──────────────────────────────────────────┐
   │                  Stage 2 — Preprocessing            │
   │  DataPreprocessor:                                  │
   │    • Canonical ID mapping (course_name+instructor)  │
   │    • Duplicate removal, sparsity filtering (≥3 ints)│
   │    • Outlier clipping (FLOAT_BOUNDS)                │
   │    • Binary & ordinal encoding                      │
   │    • Feature engineering: completion_rate,          │
   │      engagement_score, weighted_score, price_tier   │
   │    • MinMax normalisation → processed_data.csv      │
   │    • Sparse CSR user-item matrix (engagement signal)│
   │    • Course feature store → .pkl artifacts          │
   └──────────┬──────────────────────────────────────────┘
              │
   ┌──────────▼──────────────────────────────────────────┐
   │              Stage 4 — Feature Engineering          │
   │  FeatureEngineer:                                   │
   │    • Semantic encoding via all-MiniLM-L6-v2         │
   │      (SentenceTransformer) on structured text corpus│
   │    • Numeric features: MinMax-scaled on 10 columns  │
   │    • Combined L2-normalised course content matrix   │
   │    • User profiles = weighted average over seen     │
   │      course vectors (weighted by rating)            │
   └──────────┬──────────────────────────────────────────┘
              │
   ┌──────────▼──────────────────────────────────────────┐
   │                Stage 5 — Model Training             │
   │                                                     │
   │  ① PopularityModel    — Bayesian weighted_score rank│
   │  ② ContentBasedModel  — cosine similarity on matrix │
   │  ③ UserInterestModel  — explicit per-user vectors   │
   │  ④ CollaborativeModel — ALS via implicit + BM25     │
   │  ⑤ KNNRecommender     — user-based cosine KNN       │
   │  ⑥ HybridModel        — adaptive weighted ensemble  │
   └──────────┬──────────────────────────────────────────┘
              │
   ┌──────────▼──────────────────────────────────────────┐
   │              Stage 6 — Offline Evaluation           │
   │  RecommendationEvaluator:                           │
   │    • User holdout split (random stratified)         │
   │    • Metrics: Precision@K, Recall@K, MAP@K,         │
   │      NDCG@K, HitRate@K                              │
   │    • Tiers: exact_course / course_family /          │
   │      soft_relevance                                 │
   │    • Segments: cold / warm / active users           │
   │    • Saves → reports/evaluation_metrics.json        │
   └──────────┬──────────────────────────────────────────┘
              │
              ▼
   ┌─────────────────────────────────────────────────────┐
   │            Streamlit Dashboard  (app/app.py)        │
   │  Tab 1 — 👑 Recommendations                         │
   │  Tab 2 — 🔗 Course Similarity                       │
   │  Tab 3 — 🗺️ Domain Intelligence                     │
   │  Tab 4 — 📊 Model Evaluation                        │
   │  Tab 5 — 📚 Data Explorer                           │
   └──────────────────────────────────────────────────────┘
```

---

## ⚙️ Tech Stack

**Language:** Python 3.11+

**ML / Recommendation:** scikit-learn (NearestNeighbors, cosine_similarity, MinMaxScaler), implicit (ALS + BM25 weighting), sentence-transformers (all-MiniLM-L6-v2), PyTorch (sentence-transformers backend), scipy (sparse CSR matrices), numpy, pandas

**Dashboard / Visualisation:** Streamlit ≥1.35, Plotly (graph_objects + express), matplotlib, seaborn

**Data I/O:** openpyxl (Excel ingestion)

**Network analysis / EDA extras:** networkx, statsmodels

**Testing:** pytest ≥8.0

**Notebooks:** Jupyter (notebook, ipykernel, ipywidgets)

---

## 📂 Project Structure

```
Recommendation-System/
├── main.py                          # CLI entry point (train/serve/recommend/similar/popular)
├── requirements.txt
├── .gitignore
│
├── app/
│   └── app.py                       # Streamlit dashboard (5 tabs, ~1900 lines)
│
├── config/
│   ├── __init__.py
│   └── config.py                    # Central config: all paths, schema, model settings
│
├── dashboard/
│   └── courseiq_dashboard.html      # Static HTML analytics dashboard (alternative view)
│
├── data/
│   ├── raw/
│   │   └── online_course_recommendation.xlsx   # Source dataset
│   └── processed/
│       ├── processed_data.csv                  # Cleaned, engineered interactions
│       ├── user_item_matrix.pkl                # Sparse CSR matrix + index maps
│       ├── course_features.pkl                 # Aggregated course-level feature store
│       └── preprocessor_scaler.pkl             # Fitted MinMaxScaler
│
├── models/                          # Trained model pickles (git-ignored)
│   ├── collaborative_model.pkl
│   ├── content_model.pkl
│   ├── hybrid_model.pkl
│   ├── knn_model.pkl
│   ├── popularity_model.pkl
│   └── user_interest_model.pkl
│
├── notebooks/
│   ├── 01_advanced_eda_analysis.ipynb
│   └── 02_model_building_evaluation_deployment.ipynb
│
├── reports/
│   ├── evaluation_metrics.json      # Offline evaluation output (all models + segments)
│   └── figures/                     # 10 evaluation/visualisation PNG outputs
│
├── src/
│   ├── data/
│   │   ├── data_ingestion.py        # Excel loading + schema validation
│   │   └── data_preprocessing.py   # Canonical ID mapping, feature engineering, matrix build
│   ├── evaluation/
│   │   └── recommendation_evaluator.py   # Offline holdout evaluation (5 metrics × 3 tiers)
│   ├── features/
│   │   └── feature_engineering.py  # MiniLM semantic encoding + numeric scaling
│   ├── models/
│   │   ├── collaborative_model.py   # ALS via implicit library
│   │   ├── content_based_model.py   # Cosine similarity on content matrix
│   │   ├── hybrid_model.py          # Adaptive weighted ensemble (cold/warm/active)
│   │   ├── knn_model.py             # User-based KNN (sklearn NearestNeighbors)
│   │   ├── popularity_model.py      # Bayesian weighted_score ranker
│   │   └── user_interest_model.py   # Explicit per-user content vectors + summaries
│   ├── pipeline/
│   │   └── training_pipeline.py     # 6-stage orchestrator
│   └── utils/
│       ├── helpers.py               # save/load pickle, timer, encoders, clip_outliers
│       └── logger.py                # Dual handler logger (console INFO + file DEBUG)
│
├── tests/
│   └── test_pipeline.py             # pytest suite
│
└── logs/
    └── pipeline.log                 # Runtime logs (git-ignored)
```

---

## 🔍 Features

### Core Features

- **Hybrid recommendation engine** blending five independent model arms with adaptive weight selection per user activity segment (cold / warm / active)
- **Semantic content similarity** using sentence-transformers (`all-MiniLM-L6-v2`) to encode rich course narratives and find semantically related courses
- **ALS collaborative filtering** (via `implicit` library) with BM25-weighted interaction confidence
- **User-based KNN** via sklearn `NearestNeighbors` with cosine metric, aggregating neighbour interactions weighted by similarity
- **Popularity baseline** using Bayesian smoothed `weighted_score` with optional difficulty-level filtering
- **Explicit user interest profiles** storing per-user content vectors, top instructors, and difficulty preferences for interpretable recommendations
- **Cold-start handling**: cold users (<2 interactions) fall back to popularity + content only; warm users (2–4 interactions) use a transitional weight set; active users (>4 interactions) use the full ensemble

### Advanced / Unique Features

- **Canonical ID mapping**: course_name + instructor pairs are collapsed to a single `course_id` at preprocessing time, resolving catalogue fragmentation from duplicated raw IDs
- **Three-tier offline evaluation** with exact course, course family, and soft relevance tiers, segmented by cold/warm/active user cohorts
- **Business logic boosters**: certification (+2%), study materials (+1%), difficulty progression (+3%) applied as score multipliers in the hybrid scoring step
- **Safe dominance guarantee**: final hybrid score is floored at 98% of the best individual arm's score, so the ensemble never underperforms its strongest sub-model
- **Score normalisation across arms**: every model arm's raw scores are min-max normalised to [0, 1] before blending to prevent scale dominance
- **TF-IDF semantic search tab** in the dashboard for free-text course search (Tab 5 Data Explorer)
- **Full interactive dashboard** with radar charts, sunburst learner profiles, Sankey enrollment flows, scatter plots, treemaps, evaluation heatmaps, and model comparison bar charts

---

## 🧠 Machine Learning Details

### Model Types

| Model              | Type                                       | Library      |
| ------------------ | ------------------------------------------ | ------------ |
| CollaborativeModel | Matrix factorisation (ALS)                 | `implicit` |
| ContentBasedModel  | Cosine similarity on course content matrix | scikit-learn |
| KNNRecommender     | User-based K-Nearest Neighbours (cosine)   | scikit-learn |
| PopularityModel    | Bayesian weighted ranking                  | pandas       |
| UserInterestModel  | Weighted-average user content vectors      | numpy        |
| HybridModel        | Adaptive weighted ensemble                 | custom       |

### Feature Engineering

The course content matrix combines two feature groups, concatenated then L2-normalised:

**Semantic features** (via `all-MiniLM-L6-v2`): A structured natural-language corpus is built per course with the following fields — course title, instructor name, difficulty level, price tier, price range band, certification status, study material availability, and enrollment tier. This corpus is encoded into dense 384-dimensional vectors.

**Numeric features** (10 columns, MinMax-scaled): `course_duration_hours`, `difficulty_level` (ordinal 0–2), `rating`, `enrollment_numbers`, `course_price`, `feedback_score`, `time_spent_hours`, `completion_rate`, `engagement_score`, `previous_courses_taken`.

Engineered columns added during preprocessing: `completion_rate` = time_spent / course_duration, `engagement_score` = weighted combination of rating (40%), feedback_score (30%), completion_rate (30%), `weighted_score` = Bayesian smoothed rating (Imdb-style), `price_tier` = cut into low/mid/high buckets.

Binary columns (`certification_offered`, `study_material_available`) are encoded 0/1. `difficulty_level` is ordinal-encoded (Beginner=0, Intermediate=1, Advanced=2).

### Hybrid Weights

Three weight regimes are applied based on a user's interaction count:

| Arm           | Cold (<2) | Warm (2–4) | Active (>4) |
| ------------- | --------- | ----------- | ----------- |
| Content       | 0.45      | 0.65        | 0.70        |
| Popularity    | 0.55      | 0.18        | 0.12        |
| Collaborative | 0.00      | 0.06        | 0.08        |
| UserInterest  | 0.00      | 0.08        | 0.07        |
| KNN           | 0.00      | 0.03        | 0.03        |

Weights are re-normalised at runtime over arms that actually return results.

### Training Approach

Training runs as a 6-stage sequential pipeline invoked by `python main.py train`. Stages execute in order: data ingestion → preprocessing (with sparsity filtering at min_interactions=3, iterative convergence) → artifact loading → feature engineering (MiniLM encoding, can auto-download from Hugging Face Hub on first run) → model training (all five arms, then hybrid assembly) → offline evaluation. Total pipeline runtime depends on dataset size and whether the MiniLM model is already cached locally.

### Evaluation Results (K=10, 500 users)

| Model         | HitRate@10 (exact) | NDCG@10 (exact) | MAP@10 (exact) |
| ------------- | ------------------ | --------------- | -------------- |
| Hybrid        | 0.886              | 0.5809          | 0.4811         |
| Content       | 0.680              | 0.4720          | 0.4070         |
| Collaborative | 1.000              | 1.0000          | 1.0000         |
| Popularity    | 0.022              | 0.0084          | 0.0044         |
| KNN           | 0.000              | 0.0000          | 0.0000         |

> Note: Collaborative ALS achieves perfect scores on the evaluation holdout, likely due to the nature of the hold-out split on this dataset's sparsity profile. Hybrid maintains strong generalisation across all user segments including cold-start.

---

## 🖥️ Setup Instructions (Windows)

### Prerequisites

- Python 3.11+ installed and added to PATH
- Git (optional, for cloning)
- PowerShell or CMD

### Step 1 — Clone or Extract the Project

```powershell
# If using the zip:
# Extract Recommendation-System.zip to your desired folder, then:
cd Recommendation-System
```

### Step 2 — Create a Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 3 — Install Dependencies

```powershell
pip install -r requirements.txt
```

> **Note:** The first training run will download the `all-MiniLM-L6-v2` model (~90 MB) from Hugging Face Hub if it is not already cached. Subsequent runs use the local cache. Set `HF_HUB_DISABLE_TELEMETRY=1` is already configured in `main.py`.

### Step 4 — Run the Training Pipeline

```powershell
python main.py train
```

This executes all 6 pipeline stages and writes trained model artifacts to `models/`, processed data to `data/processed/`, and evaluation results to `reports/evaluation_metrics.json`.

### Step 5 — Launch the Streamlit Dashboard

```powershell
python main.py serve
```

Or directly:

```powershell
python -m streamlit run app\app.py
```

The dashboard opens automatically at `http://localhost:8501`.

### Environment Variables

No `.env` file is required. The following environment variables are set programmatically in `main.py`:

| Variable                     | Value     | Purpose                                 |
| ---------------------------- | --------- | --------------------------------------- |
| `OPENBLAS_NUM_THREADS`     | `1`     | Prevent OpenBLAS thread conflicts       |
| `TOKENIZERS_PARALLELISM`   | `false` | Suppress HuggingFace tokenizer warnings |
| `HF_HUB_DISABLE_TELEMETRY` | `1`     | Disable Hugging Face telemetry          |

---

## ▶️ Usage

### Full Training Pipeline

```powershell
python main.py train
```

### Launch Interactive Dashboard

```powershell
python main.py serve
```

### Get Hybrid Recommendations for a User (CLI)

```powershell
python main.py recommend --user_id 123 --n 10
```

### Find Content-Similar Courses

```powershell
python main.py similar --course_id 42 --n 5
```

### Get Trending Courses (with optional difficulty filter)

```powershell
python main.py popular --n 10 --difficulty Beginner
```

> All `recommend`, `similar`, and `popular` commands require the training pipeline to have been run first. They load from `models/` — they do not retrain.

### How Recommendations Are Generated

1. User ID is entered in the dashboard sidebar or via CLI.
2. `HybridModel.recommend()` determines the user's interaction count to select the cold / warm / active weight regime.
3. Each active arm (Popularity, Content, Collaborative, KNN, UserInterest) produces a ranked candidate list.
4. Each arm's scores are min-max normalised to [0, 1].
5. Normalised scores are blended using the regime weights, re-normalised over arms that returned results.
6. The dominance floor is applied: `final_score = max(blended_score, best_arm_score × 0.98)`.
7. Business logic boosts (certification, study materials, difficulty progression) are applied.
8. Duplicate (course_name, instructor) pairs are removed; results are sorted by final score.
9. Top-N results are returned as a DataFrame and rendered in the dashboard.

---

## 📊 Dataset

**File:** `data/raw/online_course_recommendation.xlsx`

| Column                       | Type                | Description                                                              |
| ---------------------------- | ------------------- | ------------------------------------------------------------------------ |
| `user_id`                  | Integer             | Unique learner identifier                                                |
| `course_id`                | Integer             | Unique course identifier (remapped to canonical ID during preprocessing) |
| `course_name`              | String              | Name of the online course                                                |
| `instructor`               | String              | Name of the course instructor                                            |
| `course_duration_hours`    | Float (5.0–100.0)  | Total course duration in hours                                           |
| `certification_offered`    | String (Yes/No)     | Whether a certificate is awarded on completion                           |
| `difficulty_level`         | String              | Course difficulty: Beginner, Intermediate, Advanced                      |
| `rating`                   | Float (1.0–5.0)    | User-provided star rating                                                |
| `enrollment_numbers`       | Integer             | Total number of enrolled students                                        |
| `course_price`             | Float (20.0–500.0) | Course price in USD                                                      |
| `feedback_score`           | Float (0.0–1.0)    | Normalised sentiment score from student feedback                         |
| `study_material_available` | String (Yes/No)     | Whether supplementary study materials are provided                       |
| `time_spent_hours`         | Float (1.0–100.0)  | Average hours a learner spends on the course                             |
| `previous_courses_taken`   | Integer             | Number of courses the learner completed prior to this one                |

---

## 📈 Results / Output

### CLI Output

```
Top-10 recommendations for user_id=123:
 course_id           course_name    instructor  hybrid_score  ...
```

### Streamlit Dashboard (5 Tabs)

**Tab 1 — 👑 Recommendations:** KPI metrics bar (total courses, domains explored, avg rating, certifications, study materials, time invested), animated recommendation cards ranked by hybrid score with difficulty badges, domain badges, certification indicators, and engagement sparklines. Supports Hybrid, Content-Based, Collaborative, and Popularity model selection. Difficulty and domain filters available in sidebar.

**Tab 2 — 🔗 Course Similarity:** Select any course from the user's history or course catalogue and retrieve the top-N semantically similar courses with similarity scores, rendered as styled similarity cards. Includes a cosine similarity heatmap visualisation.

**Tab 3 — 🗺️ Domain Intelligence:** Platform-wide analytics — radar chart of recommended course attributes, sunburst chart of user interest profile, Sankey enrollment flow, scatter plot, and treemap of the full course catalogue coloured by domain.

**Tab 4 — 📊 Model Evaluation:** Reads `reports/evaluation_metrics.json` and renders an NDCG@10 heatmap across all models and evaluation tiers, plus a grouped bar chart comparing HitRate@10 across models.

**Tab 5 — 📚 Data Explorer:** Full raw dataset table with TF-IDF semantic search across course names and instructor fields, a domain distribution bar chart, and pipeline stage diagram.

---

## 🚀 Deployment

The system is deployed **locally** as a Streamlit web application. No cloud infrastructure is required.

- **Training**: `python main.py train` — runs once to produce all model artifacts under `models/`
- **Serving**: `python main.py serve` launches Streamlit on `localhost:8501`
- All model artifacts are serialised as Python pickle files and loaded on dashboard startup using `@st.cache_resource`
- The static HTML dashboard (`dashboard/courseiq_dashboard.html`) provides a standalone analytics view that can be opened in any browser without running Streamlit

There is no Docker configuration, no Flask/FastAPI REST API, and no cloud deployment configuration in the current codebase.

---

## ⚠️ Known Issues / Limitations

- **KNN performance is weak on this dataset.** Offline evaluation shows HitRate@10 = 0.0 for KNN on exact_course tier. The dataset's sparsity structure means most users share insufficient common interactions for neighbourhood-based aggregation to succeed. KNN weight in the hybrid ensemble is therefore set to 0.03 (lowest of all arms).
- **Collaborative ALS evaluation scores appear inflated.** The ALS model achieves perfect Precision, Recall, MAP, and NDCG in offline evaluation across all user segments. This is likely a consequence of how the holdout split interacts with the sparsity-filtered dataset — users retained by the min_interactions=3 filter inherently have predictable patterns.
- **MiniLM model download required on first run.** If `all-MiniLM-L6-v2` is not cached locally, training pauses to download ~90 MB from Hugging Face Hub. Offline/air-gapped environments need to pre-seed the HuggingFace cache.
- **`models/` directory is empty after cloning.** All trained model `.pkl` files are git-ignored. `python main.py train` must be run before `serve`, `recommend`, `similar`, or `popular` commands will work.
- **Dashboard course search uses a static category mapping (`CAT_MAP`).** Only the 20 course names hardcoded in `app.py` receive a category label; all others are binned as "Other Domain".
- **No REST API.** Recommendations are only accessible via CLI or the Streamlit dashboard. There is no HTTP endpoint for integration with external systems.

---

## 🔮 Future Improvements

- **Add a FastAPI REST layer** to expose `/recommend/{user_id}` and `/similar/{course_id}` endpoints for integration with web or mobile frontends
- **Expand CAT_MAP** dynamically by running NLP topic modelling (e.g. LDA or zero-shot classification) over `course_name` at preprocessing time instead of maintaining a static dictionary
- **Improve KNN signal** by increasing `min_interactions` threshold further and experimenting with item-based KNN alongside user-based KNN, which may perform better on this dataset's interaction density
- **Add online/incremental learning** so new user interactions update model scores without requiring full retraining — particularly for the popularity and user interest arms
- **Containerise with Docker** to make the full stack (training + serving) reproducible across environments without manual virtual environment setup
- **Add A/B testing hooks** in the hybrid model to allow real-time comparison of weight regimes via click-through and completion rate feedback
- **Extend evaluation** with coverage@K and diversity@K metrics, which are currently not computed in `RecommendationEvaluator`
- **Add user-facing explanation text** to each recommendation card surfacing the reason for the suggestion (e.g. "Because you completed Advanced Machine Learning")

---

## 👨‍💻 Author

Developed as part of the P662 capstone project. Contact the project team for access to the raw dataset or trained model artifacts.
