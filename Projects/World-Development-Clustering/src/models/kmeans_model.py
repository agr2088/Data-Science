"""
src/models/kmeans_model.py
World Development Clustering - KMeans Clustering Model (PRIMARY)

Tuning:
  - n_clusters: dynamically selected via silhouette score over KMEANS_K_RANGE
  - n_init >= 10 (config enforced)
  - Optimal K from both Elbow (inertia) and Silhouette analysis
  - k=2 excluded unless silhouette improvement > 15% over best k in [3,6]
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from config.config import KMEANS_PARAMS, KMEANS_K_RANGE, MODELS_DIR, RANDOM_STATE
from src.utils.logger import get_logger
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = get_logger(__name__)

MODEL_PATH = os.path.join(MODELS_DIR, "kmeans_model.pkl")

_LABEL_COLS = {
    "KMeans_Cluster", "Hierarchical_Cluster", "GMM_Cluster",
    "Spectral_Cluster", "Birch_Cluster", "GMM_Confidence",
}

def find_optimal_k(X: np.ndarray) -> dict:
    """
    Elbow method + Silhouette analysis over KMEANS_K_RANGE (3-6).
    k=2 is excluded unless silhouette improvement > 15% over best k in range.
    Returns dict with inertia and silhouette per k.
    Optimal k = argmax(silhouette_score) within constrained range.
    Logs both inertia (elbow) and silhouette for full analysis.
    """
    logger.info(f"KMeans k search over {list(KMEANS_K_RANGE)}")
    results = {}
    for k in KMEANS_K_RANGE:
        km = KMeans(
            n_clusters=k, init="k-means++", n_init=10,
            random_state=RANDOM_STATE, max_iter=300
        )
        labels  = km.fit_predict(X)
        sil     = silhouette_score(X, labels) if k > 1 else None
        sil_str = f"{sil:.4f}" if sil is not None else "N/A"
        results[k] = {"inertia": round(km.inertia_, 2), "silhouette": sil}
        logger.info(f"  k={k} | inertia={km.inertia_:.2f} | silhouette={sil_str}")

    # k=2 only admitted if its silhouette exceeds best-in-range by >15%
    km2  = KMeans(n_clusters=2, init="k-means++", n_init=10,
                  random_state=RANDOM_STATE, max_iter=300)
    sil2 = silhouette_score(X, km2.fit_predict(X))
    best_in_range_sil = max(
        results[k]["silhouette"] for k in results if results[k]["silhouette"] is not None
    )

    if sil2 > best_in_range_sil * 1.15:
        results[2] = {"inertia": round(km2.inertia_, 2), "silhouette": sil2}
        logger.info(
            f"  k=2 admitted: silhouette={sil2:.4f} > 15% improvement over "
            f"best in range ({best_in_range_sil:.4f})"
        )
    else:
        logger.info(
            f"  k=2 rejected: silhouette={sil2:.4f} does not meet 15% "
            f"improvement threshold over best in range ({best_in_range_sil:.4f})"
        )

    best_sil_k = max(
        (k for k in results if results[k]["silhouette"] is not None),
        key=lambda k: results[k]["silhouette"]
    )
    logger.info(f"  k={best_sil_k} | silhouette={results[best_sil_k]['silhouette']:.4f} - BEST")
    return results

def select_optimal_k(k_results: dict) -> int:
    """
    Dynamically select optimal k from find_optimal_k() results.
    Returns k with highest silhouette score.
    """
    best_k = max(
        (k for k in k_results if k_results[k]["silhouette"] is not None),
        key=lambda k: k_results[k]["silhouette"]
    )
    logger.info(f"KMeans optimal k selected: {best_k} (silhouette={k_results[best_k]['silhouette']:.4f})")
    return best_k

def train_kmeans(X: np.ndarray, n_clusters: int = None) -> KMeans:
    """
    Train KMeans with dynamic or specified cluster count.
    n_init is enforced >= 10 from KMEANS_PARAMS.
    Falls back to KMEANS_PARAMS['n_clusters'] only if n_clusters is None
    and no dynamic selection is performed upstream.
    """
    k = n_clusters or KMEANS_PARAMS["n_clusters"]
    assert KMEANS_PARAMS["n_init"] >= 10, "n_init must be >= 10"

    params = {**KMEANS_PARAMS, "n_clusters": k}
    km     = KMeans(**params)
    km.fit(X)

    sil = silhouette_score(X, km.labels_)
    db_score = None
    try:
        from sklearn.metrics import davies_bouldin_score
        db_score = davies_bouldin_score(X, km.labels_)
    except Exception:
        pass

    if db_score is not None:
        logger.info(f"[MODEL: KMeans] k={k} | n_init={KMEANS_PARAMS['n_init']} | Silhouette={sil:.4f} | Davies-Bouldin={db_score:.4f} | Inertia={km.inertia_:.2f} | Status: COMPLETE")
    else:
        logger.info(f"[MODEL: KMeans] k={k} | n_init={KMEANS_PARAMS['n_init']} | Silhouette={sil:.4f} | Inertia={km.inertia_:.2f} | Status: COMPLETE")

    return km

def save_model(model: KMeans) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"KMeans model saved - {MODEL_PATH}")

def load_model() -> KMeans:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info(f"KMeans model loaded - {MODEL_PATH}")
    return model

def get_cluster_labels(df: pd.DataFrame, model: KMeans) -> pd.DataFrame:
    """Adds 'KMeans_Cluster' column. Excludes any existing cluster columns."""
    feature_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in _LABEL_COLS
    ]
    df                   = df.copy()
    df["KMeans_Cluster"] = model.predict(df[feature_cols].values)
    
    counts               = df["KMeans_Cluster"].value_counts().sort_index()
    sizes_str = " | ".join([f"{idx}: {cnt}" for idx, cnt in counts.items()])
    logger.info(f"[KMeans] Cluster sizes: {sizes_str}")
    return df