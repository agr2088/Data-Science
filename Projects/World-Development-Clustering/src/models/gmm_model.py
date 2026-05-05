"""
src/models/gmm_model.py
World Development Clustering - Gaussian Mixture Model

Tuning:
  - covariance_type: full, tied, diag - selected by BIC
  - n_components: from shared optimal_k
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from config.config import GMM_PARAMS, MODELS_DIR, KMEANS_K_RANGE, RANDOM_STATE
from src.utils.logger import get_logger
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

logger = get_logger(__name__)

MODEL_PATH = os.path.join(MODELS_DIR, "gmm_model.pkl")

_LABEL_COLS = {
    "KMeans_Cluster", "Hierarchical_Cluster", "GMM_Cluster",
    "Spectral_Cluster", "Birch_Cluster", "GMM_Confidence",
}

# All covariance types to compare
_COV_TYPES = ("full", "tied", "diag")

def compare_covariance_types(X: np.ndarray, k: int) -> dict:
    """
    Compare GMM covariance types: full, tied, diag using BIC.
    - full:  each cluster has its own unrestricted covariance matrix
    - tied:  all clusters share one covariance matrix
    - diag:  diagonal covariance per cluster (features uncorrelated within cluster)

    Lower BIC = better model accounting for complexity penalty.
    Returns dict with BIC/AIC/silhouette per type and 'recommended' key.
    """
    logger.info(f"GMM covariance search at k={k}")
    results = {}

    for cov_type in _COV_TYPES:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=cov_type,
            n_init=10,
            max_iter=200,
            random_state=RANDOM_STATE,
        )
        gmm.fit(X)

        labels  = gmm.predict(X)
        bic     = gmm.bic(X)
        aic     = gmm.aic(X)
        sil     = silhouette_score(X, labels) if k > 1 else None

        sil_str = f"{sil:.4f}" if sil is not None else "N/A"

        results[cov_type] = {
            "bic"       : round(bic, 2),
            "aic"       : round(aic, 2),
            "silhouette": round(sil, 4) if sil is not None else None,
        }

        logger.info(
            f"  covariance_type='{cov_type}' | "
            f"BIC={bic:.2f} | AIC={aic:.2f} | Silhouette={sil_str}"
        )

    winner   = min(_COV_TYPES, key=lambda t: results[t]["bic"])
    bic_vals = {t: results[t]["bic"] for t in _COV_TYPES}
    logger.info(f"GMM covariance selected: '{winner}' (lowest BIC) | BIC scores: {bic_vals}")
    
    results["recommended"] = winner
    return results

def find_optimal_components_bic(X: np.ndarray, best_cov_type: str = "full") -> dict:
    """
    Sweep k over KMEANS_K_RANGE using the selected covariance type.
    Returns BIC/AIC/silhouette per k.
    """
    logger.info(f"GMM k search over {list(KMEANS_K_RANGE)} (cov='{best_cov_type}')")
    k_results = {}

    for k in KMEANS_K_RANGE:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=best_cov_type,
            n_init=5,
            random_state=RANDOM_STATE,
        )
        gmm.fit(X)

        labels  = gmm.predict(X)
        bic     = gmm.bic(X)
        aic     = gmm.aic(X)
        sil     = silhouette_score(X, labels) if k > 1 else None

        sil_str = f"{sil:.4f}" if sil is not None else "N/A"

        k_results[k] = {
            "bic"       : round(bic, 2),
            "aic"       : round(aic, 2),
            "silhouette": round(sil, 4) if sil is not None else None,
        }
        logger.info(f"  k={k} | BIC={bic:.2f} | AIC={aic:.2f} | silhouette={sil_str}")

    best_bic_k = min(k_results, key=lambda k: k_results[k]["bic"])
    logger.info(f"  k={best_bic_k} | BIC={k_results[best_bic_k]['bic']:.2f} - BEST")
    
    return k_results

def train_gmm(
    X: np.ndarray,
    n_components: int = None,
    covariance_type: str = None,
) -> GaussianMixture:
    """Train GMM with BIC-selected covariance type and optimal k."""
    k        = n_components or GMM_PARAMS["n_components"]
    cov_type = covariance_type or GMM_PARAMS["covariance_type"]

    params = {**GMM_PARAMS, "n_components": k, "covariance_type": cov_type}
    gmm    = GaussianMixture(**params)
    gmm.fit(X)

    labels = gmm.predict(X)
    sil    = silhouette_score(X, labels)
    bic    = gmm.bic(X)

    logger.info(f"[MODEL: GMM] k={k} | covariance_type='{cov_type}' | Silhouette={sil:.4f} | BIC={bic:.2f} | Converged={gmm.converged_} | Status: COMPLETE")
    
    return gmm

def save_model(model: GaussianMixture) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"GMM model saved - {MODEL_PATH}")

def load_model() -> GaussianMixture:
    model = joblib.load(MODEL_PATH)
    logger.info(f"GMM model loaded - {MODEL_PATH}")
    return model

def get_cluster_labels(df: pd.DataFrame, model: GaussianMixture) -> pd.DataFrame:
    """Adds GMM_Cluster and GMM_Confidence columns."""
    feature_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in _LABEL_COLS
    ]
    
    df = df.copy()
    X  = df[feature_cols].values

    df["GMM_Cluster"]    = model.predict(X)
    df["GMM_Confidence"] = model.predict_proba(X).max(axis=1)

    probs = model.predict_proba(X)
    for i in range(model.n_components):
        df[f"GMM_Prob_Cluster_{i}"] = probs[:, i]

    counts = df["GMM_Cluster"].value_counts().sort_index()
    sizes_str = " | ".join([f"{idx}: {cnt}" for idx, cnt in counts.items()])
    logger.info(f"[GMM] Cluster sizes: {sizes_str}")
    
    return df