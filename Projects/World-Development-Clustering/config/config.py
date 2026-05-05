"""
config/config.py
World Development Clustering — Central Configuration

Single source of truth for ALL paths, column names, and model parameters.
No other file hardcodes paths, column lists, or hyperparameters.
"""

import os

# ── Base directory (project root, two levels above this file) ─────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
REPORTS_DIR   = os.path.join(BASE_DIR, "reports")
FIGURES_DIR   = os.path.join(REPORTS_DIR, "figures")
LOGS_DIR      = os.path.join(BASE_DIR, "logs")

# ── Raw data ──────────────────────────────────────────────────────────────────
RAW_FILE   = os.path.join(RAW_DIR, "World_development_mesurement.xlsx")
SHEET_NAME = "world_development"

# ── Processed data paths ──────────────────────────────────────────────────────
CLEANED_DATA   = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
SCALED_DATA    = os.path.join(PROCESSED_DIR, "scaled_data.csv")
CLUSTERED_DATA = os.path.join(PROCESSED_DIR, "final_clustered_data.csv")

# ── Preprocessor artifact paths ───────────────────────────────────────────────
IMPUTER_PATH           = os.path.join(PROCESSED_DIR, "imputer.pkl")
SCALER_PATH            = os.path.join(PROCESSED_DIR, "scaler.pkl")
PCA_MODEL_PATH         = os.path.join(PROCESSED_DIR, "pca_model.pkl")

# Legacy path aliases for backward compatibility
KNN_IMPUTER_PATH       = IMPUTER_PATH
POWER_TRANSFORMER_PATH = os.path.join(PROCESSED_DIR, "power_transformer.pkl")

# ── Column configuration ───────────────────────────────────────────────────────
# Columns that arrive as currency strings in the raw Excel
CURRENCY_COLS = [
    "GDP",
    "Health Exp/Capita",
    "Tourism Inbound",
    "Tourism Outbound",
]

# Columns that arrive as percent strings in the raw Excel
PERCENT_COLS = [
    "Business Tax Rate",
]

# Columns dropped due to extreme missingness or zero variance
DROP_COLS = ["Ease of Business", "Number of Records"]

# Post-aggregation: drop columns with > this fraction missing
HIGH_MISSING_COL_THRESHOLD = 0.40

# Column used as country identifier (never used as a feature)
ID_COL = "Country"

# Columns that receive log1p transform (right-skewed by nature)
LOG1P_COLS = [
    "GDP",
    "Tourism Inbound",
    "Tourism Outbound",
]

# Columns to drop before clustering due to known multicollinearity
MULTICOLLINEAR_DROP = [
    "Energy Usage",
    "Life Expectancy Male",
    "Population 15-64",
]

# Features used for trend slope computation (on panel before aggregation)
TREND_COLS = [
    "GDP",
    "Internet Usage",
    "CO2 Emissions",
    "Life Expectancy Female",
    "Mobile Phone Usage",
]

# All numeric feature columns available after ingestion (aggregated mean)
ALL_NUMERIC_COLS = [
    "Birth Rate",
    "CO2 Emissions",
    "Days to Start Business",
    "Energy Usage",
    "GDP",
    "Health Exp % GDP",
    "Health Exp/Capita",
    "Hours to do Tax",
    "Infant Mortality Rate",
    "Internet Usage",
    "Lending Interest",
    "Life Expectancy Female",
    "Life Expectancy Male",
    "Mobile Phone Usage",
    "Population 0-14",
    "Population 15-64",
    "Population 65+",
    "Population Total",
    "Population Urban",
    "Tourism Inbound",
    "Tourism Outbound",
    "Business Tax Rate",
]

# Country name mapping: World Bank names → Plotly choropleth-compatible names
COUNTRY_NAME_MAP = {
    "Bahamas, The"                   : "Bahamas",
    "Brunei Darussalam"              : "Brunei",
    "Congo, Dem. Rep."               : "Democratic Republic of the Congo",
    "Congo, Rep."                    : "Republic of Congo",
    "Cote d'Ivoire"                  : "Ivory Coast",
    "Czech Republic"                 : "Czechia",
    "Egypt, Arab Rep."               : "Egypt",
    "Faeroe Islands"                 : "Faroe Islands",
    "Gambia, The"                    : "Gambia",
    "Hong Kong SAR, China"           : "Hong Kong",
    "Iran, Islamic Rep."             : "Iran",
    "Korea, Dem. Rep."               : "North Korea",
    "Korea, Rep."                    : "South Korea",
    "Kyrgyz Republic"                : "Kyrgyzstan",
    "Lao PDR"                        : "Laos",
    "Macao SAR, China"               : "Macao",
    "Macedonia, FYR"                 : "North Macedonia",
    "Micronesia, Fed. Sts."          : "Micronesia",
    "Russian Federation"             : "Russia",
    "Sint Maarten (Dutch part)"      : "Sint Maarten",
    "Slovak Republic"                : "Slovakia",
    "St. Kitts and Nevis"            : "Saint Kitts and Nevis",
    "St. Lucia"                      : "Saint Lucia",
    "St. Martin (French part)"       : "Saint Martin",
    "St. Vincent and the Grenadines" : "Saint Vincent and the Grenadines",
    "Swaziland"                      : "Eswatini",
    "Syrian Arab Republic"           : "Syria",
    "Venezuela, RB"                  : "Venezuela",
    "Virgin Islands (U.S.)"          : "United States Virgin Islands",
    "West Bank and Gaza"             : "Palestine",
    "Yemen, Rep."                    : "Yemen",
}

# ── Preprocessing parameters ──────────────────────────────────────────────────
HIGH_MISSING_ROW_THRESHOLD = 0.50
IQR_CAP_MULTIPLIER         = 3.0

# ── Feature engineering parameters ───────────────────────────────────────────
PCA_VARIANCE_THRESHOLD     = 0.90   # for visualization only — NOT used in training
VARIANCE_FILTER_THRESHOLD  = 0.01
CORRELATION_DROP_THRESHOLD = 0.90   # drop one from pairs with |r| > 0.90

# ── Clustering parameters ─────────────────────────────────────────────────────
KMEANS_K_RANGE = range(3, 7)   # k ∈ [3,4,5,6]; k=2 excluded unless silhouette >15% better

KMEANS_PARAMS = {
    "n_clusters"  : 4,
    "init"        : "k-means++",
    "n_init"      : 10,
    "random_state": 42,
    "max_iter"    : 500,
}

HIERARCHICAL_PARAMS = {
    "n_clusters" : 4,
    "linkage"    : "ward",
    "metric"     : "euclidean",
}

GMM_PARAMS = {
    "n_components"   : 4,
    "covariance_type": "full",
    "n_init"         : 10,
    "max_iter"       : 200,
    "random_state"   : 42,
}

SPECTRAL_PARAMS = {
    "n_clusters"    : 4,
    "affinity"      : "rbf",
    "n_init"        : 10,
    "random_state"  : 42,
    "assign_labels" : "kmeans",
}

BIRCH_PARAMS = {
    "n_clusters"     : 4,
    "threshold"      : 0.5,
    "branching_factor": 50,
}

# ── Cluster balance thresholds ────────────────────────────────────────────────
MIN_CLUSTER_SIZE        = 10
MAX_IMBALANCE_RATIO     = 10.0
MIN_CLUSTER_SIZE_PCT    = 0.05   # Reject model if any cluster < 5% of total samples

# ── Random seed ───────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Cluster profile names (populated after training) ──────────────────────────
CLUSTER_NAMES = {
    0: "High-Income Developed",
    1: "Upper-Middle Income",
    2: "Lower-Middle Income",
    3: "Low-Income / Fragile States",
}