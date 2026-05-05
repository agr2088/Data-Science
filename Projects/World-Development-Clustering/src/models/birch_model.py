"""
src/models/birch_model.py
World Development Clustering - BIRCH Clustering

Tuning:
  - threshold: candidate values [0.3, 0.5, 0.7, 1.0, 1.5] - selected by silhouette
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from config.config import BIRCH_PARAMS, MODELS_DIR
from src.utils.logger import get_logger
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score

logger = get_logger(__name__)

MODEL_PATH = os.path.join(MODELS_DIR, "birch_model.pkl")

_LABEL_COLS = {
    "KMeans_Cluster", "Hierarchical_Cluster", "GMM_Cluster",
    "Spectral_Cluster", "Birch_Cluster", "GMM_Confidence",
}

_THRESHOLD_CANDIDATES = [0.3, 0.5, 0.7, 1.0, 1.5]

def tune_threshold(X: np.ndarray, n_clusters: int) -> float:
    """
    Tune Birch threshold parameter over candidate values.
    Select by highest silhouette score at the given n_clusters.
    Returns the best threshold value.

    threshold controls the radius of subcluster centroids in the CF tree.
    Lower values -> more granular tree, tighter subclusters.
    """
    logger.info(f"BIRCH threshold search at k={n_clusters}")
    results = {}

    for t in _THRESHOLD_CANDIDATES:
        try:
            model  = Birch(
                n_clusters=n_clusters,
                threshold=t,
                branching_factor=BIRCH_PARAMS["branching_factor"],
            )
            labels = model.fit_predict(X)

            n_unique = len(set(labels))
            if n_unique < 2:
                logger.warning(f"  threshold={t} produced {n_unique} cluster(s) - skipping")
                results[t] = -1.0
                continue

            sil         = silhouette_score(X, labels)
            results[t]  = sil
            logger.info(f"  threshold={t} | silhouette={sil:.4f}")
        except Exception as e:
            logger.warning(f"  threshold={t} failed: {e}")
            results[t] = -1.0

    best_t = max(results, key=results.get)
    logger.info(f"  threshold={best_t} | silhouette={results[best_t]:.4f} - BEST")
    return best_t

def train_birch(
    X: np.ndarray,
    n_clusters: int = None,
    threshold: float = None,
) -> Birch:
    """
    Train BIRCH with tuned threshold.
    If threshold is None, tunes over _THRESHOLD_CANDIDATES and selects
    the one with highest silhouette score.
    """
    k = n_clusters or BIRCH_PARAMS["n_clusters"]

    if threshold is None:
        threshold = tune_threshold(X, k)

    model = Birch(
        n_clusters=k,
        threshold=threshold,
        branching_factor=BIRCH_PARAMS["branching_factor"],
    )
    model.fit(X)

    labels = model.labels_
    n_unique = len(set(labels))

    # Fallback if tuned threshold produces invalid clustering during full train
    if n_unique < 2:
        logger.warning(f"BIRCH produced {n_unique} cluster - using threshold=0.5 fallback")
        model = Birch(n_clusters=k, threshold=0.5)
        model.fit(X)
        labels = model.labels_

    sil = silhouette_score(X, model.labels_)

    try:
        from sklearn.metrics import davies_bouldin_score
        db_score = davies_bouldin_score(X, model.labels_)
        logger.info(f"[MODEL: BIRCH] k={k} | threshold={threshold} | Silhouette={sil:.4f} | Davies-Bouldin={db_score:.4f} | Status: COMPLETE")
    except Exception:
        logger.info(f"[MODEL: BIRCH] k={k} | threshold={threshold} | Silhouette={sil:.4f} | Status: COMPLETE")

    return model

def save_model(model: Birch) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"BIRCH model saved - {MODEL_PATH}")

def load_model() -> Birch:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info(f"BIRCH model loaded - {MODEL_PATH}")
    return model

def get_cluster_labels(df: pd.DataFrame, model: Birch) -> pd.DataFrame:
    """Adds 'Birch_Cluster' column using stored labels_."""
    df                 = df.copy()
    df["Birch_Cluster"] = model.labels_

    counts = df["Birch_Cluster"].value_counts().sort_index()
    sizes_str = " | ".join([f"{idx}: {cnt}" for idx, cnt in counts.items()])
    logger.info(f"[BIRCH] Cluster sizes: {sizes_str}")
    
    return df