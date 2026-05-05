# 🌍 World Development Clustering

> **Unsupervised ML pipeline that classifies 198 countries into development tiers using 5 clustering algorithms and 22 socioeconomic indicators.**

---

## 📌 Overview

This project applies unsupervised machine learning to the World Bank's development dataset to stratify nations into meaningful economic clusters — **High-Income Developed**, **Upper-Middle Income**, **Lower-Middle Income**, and **Low-Income / Fragile States** — without any predefined labels.

Five algorithms are trained in parallel (KMeans, GMM, Agglomerative, Spectral, BIRCH), evaluated on cluster quality metrics, and the best model is selected automatically. Results are served through a 7-layer interactive Streamlit dashboard.

| Stat | Value |
|---|---|
| Countries | 198 |
| Raw features | 22 |
| Training features (post-selection) | 19 + 4 trend signals |
| Clusters | 4 |
| Models evaluated | 5 |
| PCA variance retained | 90% (viz only) |
| Best model | **KMeans** (Silhouette: 0.2237) |

---

## 🗂️ Project Structure

```
World-Development-Clustering/
│
├── data/
│   ├── raw/
│   │   └── World_development_mesurement.xlsx   # Source: World Bank (2000–2012)
│   └── processed/
│       ├── cleaned_data.csv                    # Post-ingestion, pre-scaling
│       ├── scaled_data.csv                     # StandardScaler output
│       ├── final_features.csv                  # After feature selection
│       ├── pca_data.csv                        # 2-component PCA (viz only)
│       ├── final_clustered_data.csv            # All 5 cluster label columns
│       ├── imputer.pkl                         # Saved SimpleImputer
│       ├── scaler.pkl                          # Saved StandardScaler
│       └── pca_model.pkl                       # Saved PCA model
│
├── models/
│   ├── kmeans_model.pkl
│   ├── gmm_model.pkl
│   ├── hierarchical_model.pkl
│   ├── spectral_model.pkl
│   ├── birch_model.pkl
│   └── linkage_matrix.pkl
│
├── reports/
│   ├── evaluation_metrics.json                 # Silhouette, DB, CH scores
│   ├── cluster_profiles_*.csv                  # Per-model cluster Z-score profiles
│   ├── feature_importance_*.csv                # Top discriminating features
│   └── gdp_validation_*.csv                    # GDP-based label sanity check
│
├── app/
│   ├── layers/
│   │   ├── layer_0_hero.py                     # Animated title banner
│   │   ├── layer_1_command.py                  # Global controls sidebar
│   │   ├── layer_2_intelligence.py             # Cluster stats + KPIs
│   │   ├── layer_3_world_map.py                # Choropleth world map
│   │   ├── layer_4_cluster_atlas.py            # PCA scatter + radar profiles
│   │   ├── layer_5_country_lens.py             # Per-country drill-down
│   │   ├── layer_6_model_arena.py              # Model comparison table
│   │   └── layer_7_feature_vault.py            # Feature importance explorer
│   └── utils/
│       ├── charts.py                           # Reusable Plotly chart builders
│       ├── loader.py                           # Cached data loading
│       └── theme.py                            # CSS / color tokens
│
├── src/
│   ├── data/
│   │   ├── data_ingestion.py                   # Stage 1: Load + trend computation
│   │   └── data_preprocessing.py               # Stage 2: Impute + scale
│   ├── features/
│   │   ├── feature_engineering.py              # Stage 3a: Drop multicollinear
│   │   └── feature_selection.py                # Stage 3b: Drop low-variance
│   ├── models/
│   │   ├── kmeans_model.py
│   │   ├── gmm_model.py
│   │   ├── hierarchical_model.py
│   │   ├── spectral_model.py
│   │   └── birch_model.py
│   ├── evaluation/
│   │   └── cluster_evaluator.py                # Metrics, GDP validation, profiles
│   ├── pipeline/
│   │   └── training_pipeline.py                # Full 13-stage orchestrator
│   └── utils/
│       ├── logger.py
│       └── helpers.py
│
├── notebooks/
│   ├── 01_eda_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_building.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_cluster_profiling.ipynb
│
├── config/
│   └── config.py                               # Single source of truth for all paths & params
├── tests/
│   └── test_pipeline.py
├── main.py                                     # CLI entry point
└── requirements.txt
```

---

## ⚙️ Pipeline

The pipeline runs 13 sequential stages, fully orchestrated by `src/pipeline/training_pipeline.py`.

```
Stage 1  → Data Ingestion        Load Excel (2704 rows, 2000–2012 panel), parse currency/percent
                                  strings, compute trend slopes (GDP, Internet, CO2, etc.),
                                  aggregate to 208 cross-sectional rows.

Stage 2  → Preprocessing         Drop cols >40% missing → drop rows >50% missing →
                                  IQR cap (3×) → SimpleImputer (median) →
                                  log1p (GDP, Tourism) → StandardScaler.

Stage 3a → Feature Engineering   Drop multicollinear columns (Energy Usage,
                                  Life Expectancy Male, Population 15–64).
                                  Compute PCA for visualization only (not training).

Stage 3b → Feature Selection     Drop near-zero-variance features (threshold = 0.01).

Stage 4  → GMM Covariance Scan   Compare full / tied / diag covariance types via BIC.
                                  Select best type automatically.

Stage 5  → Optimal K Selection   Sweep k ∈ [3, 6] with Silhouette + BIC consensus vote.

Stage 6  → Train 5 Models        KMeans · GMM · Agglomerative · Spectral · BIRCH
                                  All trained at consensus k.

Stage 7  → Evaluate              Silhouette Score · Davies-Bouldin · Calinski-Harabasz.
                                  Reject models with any cluster < 5% of total.

Stage 8  → Select Best           KMeans preferred if within 0.01 Silhouette of leader.

Stage 9  → Label Clusters        Map numeric cluster IDs → development tier names
                                  via GDP-proxy ranking.

Stage 10 → GDP Validation        Cross-check cluster assignments against mean GDP
                                  to confirm label ordering.

Stage 11 → Feature Importance    Compute per-cluster Z-score profiles; rank top 8
                                  discriminating features.

Stage 12 → Save Artifacts        Models (.pkl), clustered CSV, metrics JSON,
                                  cluster profiles CSV.
```

---

## 📊 Model Results

| Model | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Valid |
|---|---|---|---|---|
| **KMeans** ★ | **0.2237** | 1.4700 | **57.93** | ✅ |
| Spectral | 0.2164 | 1.4965 | 55.21 | ✅ |
| Agglomerative | 0.2108 | **1.2199** | 27.88 | ❌ (tiny clusters) |
| BIRCH | 0.1983 | 1.4429 | 48.60 | ✅ |
| GMM (diag) | 0.1749 | 1.6825 | 52.02 | ✅ |

KMeans is selected as the primary model. Agglomerative is marked invalid due to two clusters containing only 2 countries each, failing the minimum cluster size check (≥5% of total).

---

## 🗺️ Cluster Profiles (KMeans, k=4)

| Cluster | Label | Countries | Key Signals |
|---|---|---|---|
| 3 | High-Income Developed | 28 | High GDP, high internet usage, high health spend, aging population (Pop 65+ +1.66σ) |
| 2 | Upper-Middle Income | 31 | Industrialising, high CO2 growth trend (+1.57σ), growing urban population |
| 0 | Lower-Middle Income | 72 | Moderate internet adoption, positive trend signals, transitional |
| 1 | Low-Income / Fragile States | 67 | Low GDP, low internet (-0.88σ), high birth rate (+1.14σ), low urban share |

---

## 🚀 Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py
```

### 3. Run a specific stage

```bash
python main.py --stage ingest        # Stage 1 only
python main.py --stage preprocess    # Stage 2 only
python main.py --stage features      # Stages 3a + 3b
python main.py --stage train         # Stages 4–12
```

### 4. Force a specific number of clusters

```bash
python main.py --k 3
```

### 5. Launch the Streamlit dashboard

```bash
streamlit run app/main.py
```

---

## 📈 Streamlit Dashboard (7 Layers)

| Layer | Name | Description |
|---|---|---|
| 0 | Hero Banner | Animated title, model pills, key stat badges |
| 1 | Command Centre | Global model selector, cluster filter sidebar |
| 2 | Intelligence Hub | Cluster KPIs, distribution charts, silhouette plots |
| 3 | World Map | Interactive choropleth coloured by cluster tier |
| 4 | Cluster Atlas | PCA scatter plot + per-cluster radar profiles |
| 5 | Country Lens | Per-country drill-down: feature bar chart + cluster history |
| 6 | Model Arena | Side-by-side model comparison table with metric rankings |
| 7 | Feature Vault | Feature importance explorer, correlation heatmap |

---

## 🔧 Configuration

All paths, column names, and hyperparameters live in **`config/config.py`** — no other file hardcodes these values.

Key parameters:

```python
KMEANS_K_RANGE            = range(3, 7)      # k sweep range
PCA_VARIANCE_THRESHOLD    = 0.90             # visualization PCA only
CORRELATION_DROP_THRESHOLD = 0.90            # drop one from |r| > 0.90 pairs
IQR_CAP_MULTIPLIER        = 3.0              # outlier capping
HIGH_MISSING_COL_THRESHOLD = 0.40            # drop columns above this
HIGH_MISSING_ROW_THRESHOLD = 0.50            # drop rows above this
MIN_CLUSTER_SIZE_PCT      = 0.05             # reject if any cluster < 5%
```

---

## 🧪 Tests

```bash
pytest tests/test_pipeline.py -v
```

---

## 📦 Requirements

```
pandas>=2.0.0        numpy>=1.24.0       scikit-learn>=1.4.0
scipy>=1.11.0        matplotlib>=3.7.0   seaborn>=0.12.0
plotly>=5.20.0       streamlit>=1.35.0   joblib>=1.3.0
openpyxl>=3.1.0      pytest>=7.4.0
```

---

## 📁 Key Output Files

| File | Description |
|---|---|
| `data/processed/final_clustered_data.csv` | 198 countries × all 5 cluster label columns |
| `reports/evaluation_metrics.json` | Silhouette, Davies-Bouldin, CH scores per model |
| `reports/cluster_profiles_KMeans_Cluster.csv` | Z-score profile per cluster (primary model) |
| `reports/feature_importance_KMeans_Cluster.csv` | Top discriminating features |
| `models/kmeans_model.pkl` | Serialised primary model for inference |

---

## 📝 Design Decisions

**No PCA before clustering.** PCA is used only for 2D visualization. All models are trained on the full 19+4 feature space to preserve interpretability.

**No DBSCAN.** World development data has no natural concept of noise points — every country belongs to a cluster. Density-based methods were excluded by design.

**KMeans preference rule.** If KMeans scores within 0.01 Silhouette of the top model, it is selected. This favours the most interpretable and reproducible algorithm over marginal metric gains.

**Trend features.** GDP, Internet Usage, CO2, Life Expectancy, and Mobile Usage trends are computed on the raw panel (2000–2012) before cross-sectional aggregation, capturing trajectory not just static level.
