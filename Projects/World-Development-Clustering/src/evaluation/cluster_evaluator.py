"""
src/evaluation/cluster_evaluator.py
World Development Clustering - Model Evaluation

Mandatory outputs for ALL 5 models:
  - Silhouette Score (higher = better, range [-1, 1])
  - Davies-Bouldin Score (lower = better)

Comparison table: Model | Silhouette | Davies-Bouldin | Selected (Yes/No)

Final selection logic:
  Step 1: Remove invalid models (any cluster < 5% of total samples)
  Step 2: Rank remaining by: high Silhouette, low Davies-Bouldin
  Step 3: Prefer KMeans if metrics within 0.01 of best; then GMM/Birch

Cluster interpretation:
  - Compute mean values per cluster on CLEANED (original-scale) data
  - Use multiple features: GDP, Life Expectancy, Internet Usage, Infant Mortality
  - Assign human-readable labels: Developed / Developing / Underdeveloped
"""
import os
import sys
import json
from collections import Counter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from config.config import REPORTS_DIR, MIN_CLUSTER_SIZE, MAX_IMBALANCE_RATIO, MIN_CLUSTER_SIZE_PCT
from src.utils.logger import get_logger
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

logger = get_logger(__name__)

# -- Core evaluation ----------------------------------------------------------

def evaluate_model(
    X: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    gmm_model=None,
) -> dict:
    """
    Compute Silhouette Score and Davies-Bouldin Score for a clustering result.
    Also computes Calinski-Harabasz for completeness.
    Passes gmm_model to get BIC/AIC alongside standard metrics.
    """
    mask        = labels != -1
    X_eval      = X[mask]
    labels_eval = labels[mask]
    n_clusters  = len(set(labels_eval))

    if n_clusters < 2:
        logger.warning(f"[{model_name}] Not enough clusters to evaluate (n_clusters={n_clusters})")
        return {
            "model"     : model_name,
            "n_clusters": n_clusters,
            "error"     : "Not enough clusters to evaluate",
        }

    sil   = silhouette_score(X_eval, labels_eval)
    db    = davies_bouldin_score(X_eval, labels_eval)
    ch    = calinski_harabasz_score(X_eval, labels_eval)
    noise = int((labels == -1).sum())

    # Imbalance check: mark invalid if any cluster < MIN_CLUSTER_SIZE_PCT of total
    n_total        = len(labels_eval)
    min_pct_thresh = int(np.ceil(MIN_CLUSTER_SIZE_PCT * n_total))
    counts         = Counter(int(l) for l in labels_eval)
    min_count      = min(counts.values())
    is_invalid     = min_count < min_pct_thresh

    if is_invalid:
        logger.warning(
            f"[{model_name}] Status: INVALID | Reason: min cluster size={min_count} < "
            f"{MIN_CLUSTER_SIZE_PCT:.0%} of {n_total} samples ({min_pct_thresh})"
        )

    result = {
        "model"            : model_name,
        "n_clusters"       : n_clusters,
        "noise_points"     : noise,
        "silhouette_score" : round(sil, 4),
        "davies_bouldin"   : round(db, 4),
        "calinski_harabasz": round(ch, 4),
        "bic"              : round(gmm_model.bic(X_eval), 2) if gmm_model else None,
        "aic"              : round(gmm_model.aic(X_eval), 2) if gmm_model else None,
        "min_cluster_size" : min_count,
        "invalid"          : is_invalid,
    }

    logger.info(
        f"[MODEL: {model_name}] "
        f"k={n_clusters} | Silhouette={sil:.4f} | Davies-Bouldin={db:.4f} | CH={ch:.2f}"
        + (f" | BIC={result['bic']}" if gmm_model else "")
        + (" | Status: INVALID (imbalanced)" if is_invalid else " | Status: COMPLETE")
    )
    return result


def compare_models(results: list) -> pd.DataFrame:
    """
    Build comparison table sorted by Silhouette score descending.
    Table columns: Model | Silhouette | Davies-Bouldin | Selected (Yes/No)
    Selection logic:
      Step 1: Remove models marked invalid (any cluster < 5% of total)
      Step 2: Rank remaining by highest Silhouette, then lowest Davies-Bouldin
      Step 3: Prefer KMeans if within 0.01 of best; then GMM/Birch over others
    """
    df = pd.DataFrame(results)

    # Filter out error rows
    df = df[~df.get("error", pd.Series([None] * len(df))).notna()].copy()

    if "silhouette_score" not in df.columns or df.empty:
        return df

    # Step 1: Remove invalid models (imbalance rule)
    invalid_mask  = df.get("invalid", pd.Series([False] * len(df))).fillna(False)
    df_valid      = df[~invalid_mask].copy()

    n_invalid     = int(invalid_mask.sum())
    if n_invalid > 0:
        invalid_names = df[invalid_mask]["model"].tolist()
        logger.warning(f"Excluding {n_invalid} invalid model(s) from selection (cluster imbalance >5%): {invalid_names}")

    if df_valid.empty:
        logger.warning("All models are invalid by imbalance rule - falling back to full set")
        df_valid = df.copy()

    # Step 2: Rank by silhouette desc, davies-bouldin asc
    df_valid = df_valid.sort_values(
        ["silhouette_score", "davies_bouldin"],
        ascending=[False, True]
    ).reset_index(drop=True)

    best_sil   = df_valid.iloc[0]["silhouette_score"]
    best_model = df_valid.iloc[0]["model"]

    # Step 3: Prefer KMeans if within 0.01 of best (only if not already top-ranked)
    kmeans_rows = df_valid[df_valid["model"].str.contains("KMeans", case=False)]
    if not kmeans_rows.empty:
        km_sil = kmeans_rows.iloc[0]["silhouette_score"]
        if abs(km_sil - best_sil) <= 0.01:
            if kmeans_rows.iloc[0]["model"] != df_valid.iloc[0]["model"]:
                logger.info(
                    f"KMeans preferred over '{df_valid.iloc[0]['model']}': "
                    f"silhouette within 0.01 (KMeans={km_sil:.4f}, best={best_sil:.4f})"
                )
            best_model = kmeans_rows.iloc[0]["model"]

    # If KMeans not within 0.01, prefer GMM or Birch over Agglomerative/Spectral
    elif best_model not in ("KMeans", ) and not best_model.startswith("GMM") and best_model != "BIRCH":
        preferred_rows = df_valid[
            df_valid["model"].str.contains("GMM|BIRCH", case=False, regex=True)
        ]
        if not preferred_rows.empty:
            pref_sil = preferred_rows.iloc[0]["silhouette_score"]
            if abs(pref_sil - best_sil) <= 0.01:
                best_model = preferred_rows.iloc[0]["model"]
                logger.info(
                    f"Preferred model '{best_model}' over '{df_valid.iloc[0]['model']}': "
                    f"silhouette within 0.01"
                )

    # Add Selected column across full df (including invalid rows)
    df["Selected"] = df["model"].apply(lambda m: "Yes" if m == best_model else "No")
    df_valid["Selected"] = df_valid["model"].apply(lambda m: "Yes" if m == best_model else "No")

    # Print formatted comparison table
    display_cols = ["model", "n_clusters", "silhouette_score", "davies_bouldin", "invalid", "Selected"]
    display_cols = [c for c in display_cols if c in df.columns]

    logger.info("\n" + "=" * 70)
    logger.info("MODEL SELECTION")
    logger.info("=" * 70)
    logger.info(df[display_cols].to_string(index=False))
    logger.info("=" * 70)
    logger.info(f"Selected: {best_model}")

    # Log selection reasons for best model
    best_row = df[df["model"] == best_model].iloc[0] if not df[df["model"] == best_model].empty else None
    if best_row is not None:
        reasons = []
        if not best_row.get("invalid", False):
            reasons.append("Balanced clusters")
        sil_val = best_row.get("silhouette_score")
        if sil_val is not None and sil_val >= 0.2:
            reasons.append(f"Good silhouette ({sil_val:.4f})")
        if "KMeans" in best_model:
            reasons.append("Interpretable")
        logger.info(f"Reason: {' | '.join(reasons)}")
    logger.info("=" * 70)
    return df


def save_metrics(results: list) -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "evaluation_metrics.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Evaluation metrics saved - {path}")


# -- Cluster size balance check -----------------------------------------------

def check_cluster_balance(
    labels: np.ndarray,
    model_name: str,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    max_imbalance_ratio: float = MAX_IMBALANCE_RATIO,
) -> dict:
    """Check cluster size balance. Warn if clusters are tiny or heavily imbalanced."""
    label_counts = Counter(int(l) for l in labels if l != -1)
    sizes        = sorted(label_counts.values())
    warnings     = []
    balance_ok   = True

    small_clusters = {k: v for k, v in label_counts.items() if v < min_cluster_size}
    if small_clusters:
        balance_ok = False
        for cluster_id, size in small_clusters.items():
            msg = (
                f"Cluster {cluster_id} has only {size} countries "
                f"(min: {min_cluster_size}). Consider reducing k."
            )
            warnings.append(msg)
            logger.warning(f"[{model_name}] {msg}")

    if sizes:
        ratio = max(sizes) / max(min(sizes), 1)
        if ratio > max_imbalance_ratio:
            balance_ok = False
            msg = (
                f"Size imbalance: {max(sizes)}/{min(sizes)} = {ratio:.1f}x "
                f"(threshold: {max_imbalance_ratio}x)"
            )
            warnings.append(msg)
            logger.warning(f"[{model_name}] {msg}")

    if balance_ok:
        logger.info(
            f"[{model_name}] Cluster sizes balanced | "
            f"Sizes: {dict(sorted(label_counts.items()))}"
        )

    return {
        "model"          : model_name,
        "size_counts"    : dict(sorted(label_counts.items())),
        "min_size"       : min(sizes) if sizes else 0,
        "max_size"       : max(sizes) if sizes else 0,
        "imbalance_ratio": round(max(sizes) / max(min(sizes), 1), 2) if sizes else None,
        "balance_ok"     : balance_ok,
        "warnings"       : warnings,
    }


# -- Cluster interpretation ---------------------------------------------------

def assign_cluster_labels(
    df: pd.DataFrame,
    cluster_col: str,
    df_cleaned: pd.DataFrame,
    country_col: str = "Country",
) -> pd.DataFrame:
    """
    Assign human-readable labels to clusters: Developed / Developing / Underdeveloped.
    Method:
      1. Join cluster assignments with original-scale cleaned data
      2. Compute composite score per cluster using MULTIPLE features:
           - GDP (economic output)
           - Life Expectancy Female (health/longevity proxy)
           - Internet Usage (technology/infrastructure)
           - Infant Mortality Rate (inverted: lower = better development)
      3. Rank clusters by composite score
      4. Assign labels based on rank:
         - Top cluster(s):    Developed
         - Middle cluster(s): Developing
         - Bottom cluster(s): Underdeveloped
    Using multiple features prevents single-feature bias (e.g. GDP-only labeling).
    Returns df with added column: '{cluster_col}_Label'
    """
    logger.info(f"Labeling '{cluster_col}' via multi-feature composite score")
    label_col = f"{cluster_col}_Label"

    if country_col not in df_cleaned.columns:
        logger.warning(f"'{country_col}' not found in cleaned data - skipping label assignment")
        return df

    # Features used for composite development score
    SCORE_FEATURES = {
        "GDP"                   : ("higher_better", 1.0),
        "Life Expectancy Female": ("higher_better", 1.0),
        "Internet Usage"        : ("higher_better", 1.0),
        "Infant Mortality Rate" : ("lower_better",  1.0),  # inverted
    }

    # Find which score features are available in cleaned data
    available = [f for f in SCORE_FEATURES if f in df_cleaned.columns]

    if not available:
        # Fallback to GDP only if none of the multi-feature cols are present
        logger.warning(
            "None of the multi-feature score columns found in cleaned data. "
            "Falling back to GDP-only labeling."
        )
        available = [c for c in ["GDP"] if c in df_cleaned.columns]
        if not available:
            logger.warning("GDP not found either - skipping label assignment")
            return df

    logger.info(f"Composite score features: {available}")

    # Merge cluster assignments with cleaned data
    merge_cols = [country_col] + available
    df_merge = df[[country_col, cluster_col]].merge(
        df_cleaned[merge_cols],
        on=country_col,
        how="left",
    )

    # Normalise each feature to [0,1] range for fair composite scoring
    df_norm = df_merge.copy()
    for feat in available:
        col_data = df_norm[feat]
        col_min  = col_data.min()
        col_max  = col_data.max()
        col_range = col_max - col_min
        if col_range == 0:
            df_norm[feat] = 0.5
        else:
            df_norm[feat] = (col_data - col_min) / col_range

    # Build composite: invert lower_better features
    df_norm["_composite"] = 0.0
    for feat in available:
        direction, weight = SCORE_FEATURES.get(feat, ("higher_better", 1.0))
        if direction == "lower_better":
            df_norm["_composite"] += weight * (1.0 - df_norm[feat])
        else:
            df_norm["_composite"] += weight * df_norm[feat]

    cluster_scores = df_norm.groupby(cluster_col)["_composite"].mean().sort_values()
    n_clusters     = len(cluster_scores)

    # Log per-cluster mean values for traceability (cleaned up)
    logger.info(f"Cluster feature means computed ({', '.join(available)})")

    label_map = {}
    if n_clusters <= 1:
        for c in cluster_scores.index:
            label_map[c] = "Developing"
    elif n_clusters == 2:
        label_map[cluster_scores.index[0]] = "Underdeveloped"
        label_map[cluster_scores.index[1]] = "Developed"
    elif n_clusters == 3:
        label_map[cluster_scores.index[0]] = "Underdeveloped"
        label_map[cluster_scores.index[1]] = "Developing"
        label_map[cluster_scores.index[2]] = "Developed"
    else:
        sorted_clusters = cluster_scores.index.tolist()
        third           = max(1, n_clusters // 3)
        for i, c in enumerate(sorted_clusters):
            if i < third:
                label_map[c] = "Underdeveloped"
            elif i >= n_clusters - third:
                label_map[c] = "Developed"
            else:
                label_map[c] = "Developing"

    df           = df.copy()
    df[label_col] = df[cluster_col].map(label_map)

    label_assignments = " | ".join([
        f"Cluster {cid} -> {lbl}"
        for cid, lbl in sorted(label_map.items())
    ])
    logger.info(f"{label_assignments}")

    for cluster_id, label in sorted(label_map.items()):
        count = int((df[cluster_col] == cluster_id).sum())
        score = cluster_scores.get(cluster_id, float("nan"))
        logger.info(
            f"Cluster {cluster_id} -> '{label}' "
            f"| composite_score={score:.3f} | n={count}"
        )

    return df


# -- Cluster profiling --------------------------------------------------------

def compute_cluster_feature_importance(
    df: pd.DataFrame,
    cluster_col: str,
    feature_cols: list,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Cluster-wise feature importance: (cluster_mean - global_mean) / global_std.
    Identifies which features most differentiate each cluster from the global average.
    """
    logger.info(f"Computing cluster feature importance for '{cluster_col}' (top {top_n})")
    global_mean   = df[feature_cols].mean()
    global_std    = df[feature_cols].std().replace(0, 1)

    cluster_means = df.groupby(cluster_col)[feature_cols].mean()
    importance_df = (cluster_means - global_mean) / global_std

    for cluster_id in importance_df.index:
        row          = importance_df.loc[cluster_id].abs().sort_values(ascending=False)
        top_features = row.head(top_n)
        logger.info(f"Cluster {cluster_id} - Top {top_n} discriminating features:")
        for feat, imp in top_features.items():
            direction = "+" if importance_df.loc[cluster_id, feat] > 0 else "-"
            logger.info(f"  {direction}{imp:.3f}  {feat}")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, f"feature_importance_{cluster_col}.csv")
    importance_df.to_csv(out_path)
    logger.info(f"Feature importance saved - {out_path}")

    return importance_df


def validate_clusters_gdp_proxy(
    df: pd.DataFrame,
    cluster_col: str,
    gdp_col: str = "GDP",
    country_col: str = "Country",
) -> dict:
    """Validate clusters against GDP quartiles as a real-world development proxy."""
    logger.info("Validating clusters against GDP-based income proxy")

    if gdp_col not in df.columns:
        logger.warning(f"Column '{gdp_col}' not found - skipping GDP validation")
        return {"error": f"Column '{gdp_col}' not found."}

    df_val = df[[country_col, cluster_col, gdp_col]].copy().dropna(subset=[gdp_col])
    df_val["GDP_Quartile"] = pd.qcut(
        df_val[gdp_col], q=4, labels=["Q1_Low", "Q2_LowMid", "Q3_UpperMid", "Q4_High"]
    )

    crosstab = pd.crosstab(df_val["GDP_Quartile"], df_val[cluster_col], margins=True)
    logger.info(f"GDP Quartile x Cluster cross-tabulation:\n{crosstab.to_string()}")

    alignment_scores = []
    for quartile in ["Q1_Low", "Q2_LowMid", "Q3_UpperMid", "Q4_High"]:
        if quartile not in crosstab.index:
            continue
        row               = crosstab.loc[quartile].drop("All")
        dominant_fraction = row.max() / row.sum()
        alignment_scores.append(dominant_fraction)
        logger.info(
            f"{quartile}: {dominant_fraction:.1%} in Cluster {row.idxmax()} "
            + ("(OK)" if dominant_fraction > 0.5 else "(POOR)")
        )

    avg_alignment = float(np.mean(alignment_scores)) if alignment_scores else 0.0
    logger.info(f"Average GDP alignment score: {avg_alignment:.3f}")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    crosstab.to_csv(os.path.join(REPORTS_DIR, f"gdp_validation_{cluster_col}.csv"))

    return {
        "crosstab"        : crosstab,
        "alignment_scores": alignment_scores,
        "avg_alignment"   : round(avg_alignment, 4),
        "alignment_ok"    : avg_alignment >= 0.50,
    }


def cluster_profile(
    df: pd.DataFrame,
    cluster_col: str,
    feature_cols: list,
) -> pd.DataFrame:
    """Mean feature values per cluster (for Streamlit profile heatmap)."""
    profile = df.groupby(cluster_col)[feature_cols].mean().round(4)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    profile.to_csv(os.path.join(REPORTS_DIR, f"cluster_profiles_{cluster_col}.csv"))
    logger.info(f"Cluster profile saved for '{cluster_col}'")
    return profile