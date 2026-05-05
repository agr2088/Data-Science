"""
src/models/hierarchical_model.py
World Development Clustering - Agglomerative Hierarchical Clustering

Tuning:
  - linkage: ward, complete, average - selected by silhouette score
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from config.config import HIERARCHICAL_PARAMS, MODELS_DIR, RANDOM_STATE
from src.utils.logger import get_logger
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage as scipy_linkage
from sklearn.metrics import silhouette_score

logger = get_logger(__name__)

MODEL_PATH   = os.path.join(MODELS_DIR, "hierarchical_model.pkl")
LINKAGE_PATH = os.path.join(MODELS_DIR, "linkage_matrix.pkl")

_LABEL_COLS = {
    "KMeans_Cluster", "Hierarchical_Cluster", "GMM_Cluster",
    "Spectral_Cluster", "Birch_Cluster", "GMM_Confidence",
}

_LINKAGE_METHODS = ("ward", "complete", "average")


def tune_linkage(X: np.ndarray, n_clusters: int) -> str:
    """
    Try linkage methods: ward, complete, average.
    Select by highest silhouette score at the given n_clusters.
    Returns the best linkage method name.
    """
    logger.info(f"Agglomerative linkage search at k={n_clusters}")
    results = {}

    for method in _LINKAGE_METHODS:
        params = {"n_clusters": n_clusters, "linkage": method}
        if method != "ward":
            params["metric"] = "euclidean"

        model  = AgglomerativeClustering(**params)
        labels = model.fit_predict(X)
        sil    = silhouette_score(X, labels)
        results[method] = sil
        logger.info(f"  linkage='{method}' | silhouette={sil:.4f}")

    best_linkage = max(results, key=results.get)
    logger.info(f"  linkage='{best_linkage}' | silhouette={results[best_linkage]:.4f} - BEST")
    return best_linkage


def compute_linkage_matrix(X: np.ndarray, method: str = "ward") -> np.ndarray:
    """Compute scipy linkage matrix for dendrogram visualization."""
    logger.info(f"Computing linkage matrix (method={method})")
    Z = scipy_linkage(X, method=method)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(LINKAGE_PATH, "wb") as f:
        pickle.dump(Z, f)
        
    logger.info(f"Linkage matrix saved - {LINKAGE_PATH}")
    return Z


def train_hierarchical(
    X: np.ndarray,
    n_clusters: int = None,
    linkage_method: str = None,
) -> AgglomerativeClustering:
    """
    Train Agglomerative Clustering with tuned linkage.
    If linkage_method is None, tunes over ward/complete/average and selects
    the one with the highest silhouette score.
    Note: 'metric' parameter is only valid for linkage != 'ward' in sklearn >= 1.4.
    """
    k = n_clusters or HIERARCHICAL_PARAMS["n_clusters"]

    if linkage_method is None:
        linkage_method = tune_linkage(X, k)

    params = {"n_clusters": k, "linkage": linkage_method}
    if linkage_method != "ward":
        params["metric"] = HIERARCHICAL_PARAMS.get("metric", "euclidean")

    model = AgglomerativeClustering(**params)
    model.fit(X)

    sil = silhouette_score(X, model.labels_)

    try:
        from sklearn.metrics import davies_bouldin_score
        db_score = davies_bouldin_score(X, model.labels_)
        logger.info(f"[MODEL: Agglomerative] k={k} | linkage='{linkage_method}' | Silhouette={sil:.4f} | Davies-Bouldin={db_score:.4f} | Status: COMPLETE")
    except Exception:
        logger.info(f"[MODEL: Agglomerative] k={k} | linkage='{linkage_method}' | Silhouette={sil:.4f} | Status: COMPLETE")

    return model


def save_model(model: AgglomerativeClustering) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Agglomerative model saved - {MODEL_PATH}")


def load_model() -> AgglomerativeClustering:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Agglomerative model loaded - {MODEL_PATH}")
    return model


def get_cluster_labels(df: pd.DataFrame, model: AgglomerativeClustering) -> pd.DataFrame:
    """Adds 'Hierarchical_Cluster' column using stored labels_."""
    df                         = df.copy()
    df["Hierarchical_Cluster"] = model.labels_

    counts = df["Hierarchical_Cluster"].value_counts().sort_index()
    sizes_str = " | ".join([f"{idx}: {cnt}" for idx, cnt in counts.items()])
    logger.info(f"[Agglomerative] Cluster sizes: {sizes_str}")

    return df