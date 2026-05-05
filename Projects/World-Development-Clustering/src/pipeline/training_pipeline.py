"""
src/pipeline/training_pipeline.py
World Development Clustering - Training Pipeline Orchestrator

5 Models: KMeans (primary), GMM, Agglomerative, Spectral, BIRCH
No DBSCAN. No PCA before clustering.

Pipeline:
  1.  Data Ingestion
  2.  Data Preprocessing (SimpleImputer median, log1p, StandardScaler)
  3.  Feature Engineering (drop multicollinear, drop |r|>0.90, PCA for viz only)
  4.  Feature Selection  (drop low-variance)
  5.  GMM covariance type comparison (full/tied/diag via BIC)
  6.  Optimal K selection (Elbow + Silhouette, dynamic)
  7.  Train all 5 models with tuning
  8.  Evaluate all 5 models (Silhouette + Davies-Bouldin)
  9.  Print comparison table: Model | Silhouette | Davies-Bouldin | Selected
  10. Select best model (prefer KMeans if within 0.01 of best)
  11. Cluster labels + interpretation (Developed/Developing/Underdeveloped)
  12. GDP proxy validation
  13. Save all artifacts
"""
import os
import sys
from collections import Counter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from config.config import CLEANED_DATA, CLUSTERED_DATA, MODELS_DIR, RANDOM_STATE
from src.data.data_ingestion import run_ingestion
from src.data.data_preprocessing import run_preprocessing
from src.features.feature_engineering import run_feature_engineering
from src.features.feature_selection import run_feature_selection

from src.models.kmeans_model import find_optimal_k, select_optimal_k, train_kmeans, save_model as save_km, get_cluster_labels as km_labels
from src.models.hierarchical_model import compute_linkage_matrix, train_hierarchical, save_model as save_hc, get_cluster_labels as hc_labels
from src.models.gmm_model import find_optimal_components_bic, compare_covariance_types, train_gmm, save_model as save_gmm, get_cluster_labels as gmm_labels
from src.models.spectral_model import train_spectral, save_model as save_spectral, get_cluster_labels as spectral_labels
from src.models.birch_model import train_birch, save_model as save_birch, get_cluster_labels as birch_labels

from src.evaluation.cluster_evaluator import (
    evaluate_model,
    compare_models,
    save_metrics,
    check_cluster_balance,
    compute_cluster_feature_importance,
    validate_clusters_gdp_proxy,
    assign_cluster_labels,
    cluster_profile,
)
from src.utils.logger import get_logger
from src.utils.helpers import save_dataframe, load_dataframe

logger = get_logger(__name__)

_LABEL_COLS = {
    "KMeans_Cluster", "Hierarchical_Cluster", "GMM_Cluster",
    "Spectral_Cluster", "Birch_Cluster", "GMM_Confidence",
}

def _get_feature_matrix(df: pd.DataFrame) -> tuple:
    """Extract numeric feature matrix, excluding any label columns."""
    feature_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in _LABEL_COLS
    ]
    countries = df["Country"].values if "Country" in df.columns else None
    X         = df[feature_cols].values
    return X, feature_cols, countries

def run_training_pipeline(optimal_k: int = None) -> dict:
    """
    Orchestrates the complete 5-model clustering pipeline.
    Args:
        optimal_k: If None, dynamically selected via Silhouette + BIC consensus.
    Returns:
        dict with all models, metrics, comparison table, and clustered DataFrame.
    """
    logger.info("=" * 70)
    logger.info("WORLD DEVELOPMENT CLUSTERING - FULL TRAINING PIPELINE")
    logger.info("Models: KMeans | GMM | Agglomerative | Spectral | BIRCH")
    logger.info("No DBSCAN. No PCA before clustering.")
    logger.info("=" * 70)

    # Stage 1: Data Ingestion
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1 - DATA INGESTION")
    logger.info("# Loading raw data and computing trend features")
    logger.info("=" * 60)
    run_ingestion()

    # Stage 2: Preprocessing
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2 - DATA PREPROCESSING")
    logger.info("# Imputing, scaling, and transforming features")
    logger.info("=" * 60)
    run_preprocessing()

    # Stage 3: Feature Engineering
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3a - FEATURE ENGINEERING")
    logger.info("# Removing redundant and highly correlated features")
    logger.info("=" * 60)
    df_features, df_pca = run_feature_engineering()

    # Stage 4: Feature Selection
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3b - FEATURE SELECTION")
    logger.info("# Dropping low-variance features before training")
    logger.info("=" * 60)
    df_for_clustering = run_feature_selection(df_features)

    X, feature_cols, countries = _get_feature_matrix(df_for_clustering)
    trend_features = [c for c in feature_cols if "_trend" in c]
    logger.info(f"Feature matrix: {X.shape[0]} countries | {X.shape[1]} features | trend features: {len(trend_features)}")

    # Stage 5: GMM Covariance Comparison
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 5 - GMM COVARIANCE TYPE COMPARISON")
    logger.info("# Comparing covariance types via BIC score")
    logger.info("=" * 60)
    cov_comparison = compare_covariance_types(X, k=4)
    best_cov_type  = cov_comparison["recommended"]
    logger.info(f"Selected covariance_type='{best_cov_type}' by BIC")

    # Stage 6: Optimal K Selection
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 6 - OPTIMAL K SELECTION")
    logger.info("# Selecting optimal k via Silhouette and BIC")
    logger.info("=" * 60)
    if optimal_k is None:
        km_k_results  = find_optimal_k(X)
        gmm_k_results = find_optimal_components_bic(X, best_cov_type=best_cov_type)
        best_sil_k = select_optimal_k(km_k_results)
        best_bic_k = min(gmm_k_results, key=lambda k: gmm_k_results[k]["bic"])
        votes     = Counter([best_sil_k, best_bic_k])
        optimal_k = votes.most_common(1)[0][0]
        logger.info(
            f"K selection - Silhouette best: k={best_sil_k} | BIC best: k={best_bic_k} "
            f"| Consensus k={optimal_k}"
        )
    else:
        logger.info(f"Using manually specified k={optimal_k}")

    # Stage 7: Model Training
    logger.info("\n" + "=" * 60)
    logger.info(f"STAGE 7 - TRAINING 5 MODELS (k={optimal_k})")
    logger.info("# Training models with tuned hyperparameters")
    logger.info("=" * 60)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. KMeans (PRIMARY)
    logger.info("Training Model 1/5: KMeans (PRIMARY)")
    km_model = train_kmeans(X, n_clusters=optimal_k)
    save_km(km_model)

    # 2. GMM
    logger.info("Training Model 2/5: GMM")
    gmm_model = train_gmm(X, n_components=optimal_k, covariance_type=best_cov_type)
    save_gmm(gmm_model)

    # 3. Agglomerative (auto-tunes linkage)
    logger.info("Training Model 3/5: Agglomerative")
    compute_linkage_matrix(X)
    hc_model = train_hierarchical(X, n_clusters=optimal_k)
    save_hc(hc_model)

    # 4. Spectral (auto-tunes affinity)
    logger.info("Training Model 4/5: Spectral")
    spectral_model = train_spectral(X, n_clusters=optimal_k)
    save_spectral(spectral_model)

    # 5. BIRCH (auto-tunes threshold)
    logger.info("Training Model 5/5: BIRCH")
    birch_model = train_birch(X, n_clusters=optimal_k)
    save_birch(birch_model)

    logger.info("All 5 models trained successfully")

    # Stage 8: Evaluation
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 8 - MODEL EVALUATION")
    logger.info("# Computing metrics for all 5 models")
    logger.info("=" * 60)
    km_labels_arr       = km_model.predict(X)
    gmm_labels_arr      = gmm_model.predict(X)
    hc_labels_arr       = hc_model.labels_
    spectral_labels_arr = spectral_model.labels_
    birch_labels_arr    = birch_model.labels_

    eval_results = [
        evaluate_model(X, km_labels_arr,       "KMeans"),
        evaluate_model(X, gmm_labels_arr,      f"GMM ({best_cov_type})", gmm_model=gmm_model),
        evaluate_model(X, hc_labels_arr,       "Agglomerative"),
        evaluate_model(X, spectral_labels_arr, "Spectral"),
        evaluate_model(X, birch_labels_arr,    "BIRCH"),
    ]

    metrics_df = compare_models(eval_results)
    save_metrics(eval_results)

    # Determine best model name from comparison table
    selected_rows = metrics_df[metrics_df.get("Selected", pd.Series()) == "Yes"]
    best_model    = selected_rows.iloc[0]["model"] if not selected_rows.empty else "KMeans"
    logger.info(f"Selected model: {best_model}")

    # Cluster size balance check for all models
    logger.info("Cluster Size Balance Checks - All Models")
    check_cluster_balance(km_labels_arr,       "KMeans")
    check_cluster_balance(gmm_labels_arr,      f"GMM ({best_cov_type})")
    check_cluster_balance(hc_labels_arr,       "Agglomerative")
    check_cluster_balance(spectral_labels_arr, "Spectral")
    check_cluster_balance(birch_labels_arr,    "BIRCH")

    # Stage 9: Labels + Interpretation
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 9 - CLUSTER LABELS + INTERPRETATION")
    logger.info("# Assigning development labels to clusters")
    logger.info("=" * 60)
    df_out = df_for_clustering.copy()
    df_out = km_labels(df_out,      km_model)
    df_out = gmm_labels(df_out,     gmm_model)
    df_out = hc_labels(df_out,      hc_model)
    df_out = spectral_labels(df_out, spectral_model)
    df_out = birch_labels(df_out,   birch_model)

    # Map best model name to cluster column
    _model_to_col = {
        "KMeans"       : "KMeans_Cluster",
        "Agglomerative": "Hierarchical_Cluster",
        "Spectral"     : "Spectral_Cluster",
        "BIRCH"        : "Birch_Cluster",
    }
    best_cluster_col = _model_to_col.get(best_model, "GMM_Cluster")
    if best_model.startswith("GMM"):
        best_cluster_col = "GMM_Cluster"

    # Load original-scale cleaned data for interpretation and GDP validation
    df_cleaned = load_dataframe(CLEANED_DATA)

    # Cluster interpretation: assign Developed / Developing / Underdeveloped
    logger.info("Cluster Interpretation - All Models")
    for cluster_col in [
        "KMeans_Cluster", "GMM_Cluster", "Hierarchical_Cluster",
        "Spectral_Cluster", "Birch_Cluster"
    ]:
        if cluster_col in df_out.columns:
            df_out = assign_cluster_labels(df_out, cluster_col, df_cleaned)

    # Feature importance for best model
    logger.info(f"Cluster Feature Importance - Best Model ({best_model})")
    importance_df = compute_cluster_feature_importance(
        df=df_out, cluster_col=best_cluster_col,
        feature_cols=feature_cols, top_n=8,
    )

    # GDP proxy validation for best model
    logger.info(f"GDP-Based Cluster Validation - Best Model ({best_model})")
    gdp_validation = {}
    if "Country" in df_cleaned.columns and "GDP" in df_cleaned.columns:
        df_val = df_cleaned[["Country", "GDP"]].merge(
            df_out[["Country", best_cluster_col]], on="Country", how="inner"
        )
        gdp_validation = validate_clusters_gdp_proxy(
            df=df_val, cluster_col=best_cluster_col,
            gdp_col="GDP", country_col="Country",
        )

    # Cluster profiles for all models
    for cluster_col in [
        "KMeans_Cluster", "GMM_Cluster", "Hierarchical_Cluster",
        "Spectral_Cluster", "Birch_Cluster"
    ]:
        if cluster_col in df_out.columns:
            cluster_profile(df_out, cluster_col, feature_cols)

    # Save final output
    save_dataframe(df_out, CLUSTERED_DATA)

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Optimal K: {optimal_k} | Selected Model: {best_model} | Cluster Col: {best_cluster_col}")
    logger.info(f"GMM cov type: {best_cov_type} | Countries: {len(df_out)} | Features: {len(feature_cols)} (incl. {len(trend_features)} trends)")
    logger.info(f"Output: {CLUSTERED_DATA}")
    logger.info("=" * 70)

    return {
        "optimal_k"         : optimal_k,
        "best_model"        : best_model,
        "best_cluster_col"  : best_cluster_col,
        "best_cov_type"     : best_cov_type,
        "km_model"          : km_model,
        "gmm_model"         : gmm_model,
        "hc_model"          : hc_model,
        "spectral_model"    : spectral_model,
        "birch_model"       : birch_model,
        "metrics"           : eval_results,
        "metrics_df"        : metrics_df,
        "clustered_df"      : df_out,
        "feature_matrix"    : X,
        "feature_cols"      : feature_cols,
        "countries"         : countries,
        "importance_df"     : importance_df,
        "gdp_validation"    : gdp_validation,
        "cov_comparison"    : cov_comparison,
    }

if __name__ == "__main__":
    run_training_pipeline()