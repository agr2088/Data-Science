"""
src/models/spectral_model.py
World Development Clustering - Spectral Clustering

Tuning:
  - affinity: rbf, nearest_neighbors - selected by silhouette score
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from config.config import SPECTRAL_PARAMS, MODELS_DIR, RANDOM_STATE
from src.utils.logger import get_logger
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

logger = get_logger(__name__)

MODEL_PATH = os.path.join(MODELS_DIR, "spectral_model.pkl")

_LABEL_COLS = {
    "KMeans_Cluster", "Hierarchical_Cluster", "GMM_Cluster",
    "Spectral_Cluster", "Birch_Cluster", "GMM_Confidence",
}

_AFFINITIES = ("rbf", "nearest_neighbors")

def tune_affinity(X: np.ndarray, n_clusters: int) -> str:
    """
    Try affinity: rbf, nearest_neighbors.
    Select by highest silhouette score at the given n_clusters.
    Returns the best affinity name.
    """
    logger.info(f"Spectral affinity search at k={n_clusters}")
    results = {}

    for affinity in _AFFINITIES:
        try:
            model = SpectralClustering(
                n_clusters=n_clusters,
                affinity=affinity,
                n_init=10,
                random_state=RANDOM_STATE,
                assign_labels="kmeans",
            )
            labels = model.fit_predict(X)
            sil    = silhouette_score(X, labels)
            results[affinity] = sil
            logger.info(f"  affinity='{affinity}' | silhouette={sil:.4f}")
        except Exception as e:
            logger.warning(f"  affinity='{affinity}' failed: {e}")
            results[affinity] = -1.0

    best_affinity = max(results, key=results.get)
    logger.info(f"  affinity='{best_affinity}' | silhouette={results[best_affinity]:.4f} - BEST")
    return best_affinity


def train_spectral(
    X: np.ndarray,
    n_clusters: int = None,
    affinity: str = None,
) -> SpectralClustering:
    """
    Train Spectral Clustering with tuned affinity parameter.
    If affinity is None, tunes over rbf/nearest_neighbors and selects
    the one with highest silhouette score.
    """
    k = n_clusters or SPECTRAL_PARAMS["n_clusters"]

    if affinity is None:
        affinity = tune_affinity(X, k)

    model = SpectralClustering(
        n_clusters=k,
        affinity=affinity,
        n_init=SPECTRAL_PARAMS["n_init"],
        random_state=RANDOM_STATE,
        assign_labels=SPECTRAL_PARAMS["assign_labels"],
    )
    model.fit(X)

    sil = silhouette_score(X, model.labels_)

    try:
        from sklearn.metrics import davies_bouldin_score
        db_score = davies_bouldin_score(X, model.labels_)
        logger.info(f"[MODEL: Spectral] k={k} | affinity='{affinity}' | Silhouette={sil:.4f} | Davies-Bouldin={db_score:.4f} | Status: COMPLETE")
    except Exception:
        logger.info(f"[MODEL: Spectral] k={k} | affinity='{affinity}' | Silhouette={sil:.4f} | Status: COMPLETE")

    return model

def save_model(model: SpectralClustering) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Spectral model saved - {MODEL_PATH}")

def load_model() -> SpectralClustering:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Spectral model loaded - {MODEL_PATH}")
    return model

def get_cluster_labels(df: pd.DataFrame, model: SpectralClustering) -> pd.DataFrame:
    """Adds 'Spectral_Cluster' column using stored labels_."""
    df                    = df.copy()
    df["Spectral_Cluster"] = model.labels_

    counts = df["Spectral_Cluster"].value_counts().sort_index()
    sizes_str = " | ".join([f"{idx}: {cnt}" for idx, cnt in counts.items()])
    logger.info(f"[Spectral] Cluster sizes: {sizes_str}")

    return df