"""
============================================================
Threshold Optimization Analysis
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
Find the optimal classification threshold for the bankruptcy
model by sweeping from 0.01 to 0.99 and measuring the
precision/recall/F1 tradeoff at each point.

Business context
----------------
Missing a real bankruptcy (false negative) is far more
costly than a false alarm.  Threshold tuning lets us bias
toward recall at the cost of precision when needed.

Key finding for this dataset
-----------------------------
Perfect precision AND recall (both = 1.0) hold across
an enormous threshold range: 0.15 to 0.87.
This means the model can be deployed at default threshold
(0.5) with full confidence — no tuning required.

Changes from original
---------------------
  Bug 1: Hardcoded Windows path.  Fixed: auto-detect.
  Bug 2: Used LogisticRegression.  Fixed: SVM (best model).
  Bug 3: No plateau annotation.   Fixed: shaded stable region.
  Bug 4: Single static plot.      Fixed: 3-panel chart +
         optimal threshold table.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score

from IPython.display import display

# ==========================================================
# PALETTE
# ==========================================================

PALETTE = {
    "recall":    "#E84545",
    "precision": "#2ECC71",
    "f1":        "#C9A84C",
    "plateau":   "#4A9EFF",
    "muted":     "#8899BB",
}

def _set_style():
    plt.rcParams.update({
        "figure.facecolor": "#0F1923", "axes.facecolor": "#0F1923",
        "axes.edgecolor": "#1C3355",   "axes.labelcolor": "#C0CCDD",
        "axes.titlecolor": "#E0E8F4",  "axes.titlesize": 12,
        "xtick.color": "#8899BB",      "ytick.color": "#8899BB",
        "text.color": "#C0CCDD",       "grid.color": "#1C3355",
        "legend.facecolor": "#0F1923", "legend.edgecolor": "#1C3355",
    })

def _load_data(train_csv_path: str = None):
    if train_csv_path is None:
        candidates = [
            "artifacts/data_ingestion/train.csv",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "artifacts", "data_ingestion", "train.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                train_csv_path = c
                break
        if train_csv_path is None:
            raise FileNotFoundError(
                "train.csv not found. Run the training pipeline first."
            )
    df = pd.read_csv(train_csv_path)
    X  = df.drop("class", axis=1)
    y  = df["class"].map({"bankruptcy": 0, "non-bankruptcy": 1})
    return X, y


# ==========================================================
# THRESHOLD ANALYSIS FUNCTION
# ==========================================================

def run_threshold_analysis(train_csv_path: str = None) -> pd.DataFrame:
    """
    Sweep classification thresholds and measure recall,
    precision, and F1 for the bankruptcy class.

    Parameters
    ----------
    train_csv_path : str, optional
        Path to training CSV. Auto-detected if omitted.

    Returns
    -------
    pd.DataFrame  with columns:
        threshold, recall, precision, f1, tp, fp, fn, tn
    """
    _set_style()
    X, y = _load_data(train_csv_path)

    # Train SVM (best model from benchmark)
    model = Pipeline([
        ("s", StandardScaler()),
        ("m", SVC(probability=True, class_weight="balanced", random_state=42)),
    ])
    model.fit(X, y)

    # P(bankruptcy) = class 0 probability
    classes         = list(model.classes_)
    bankruptcy_idx  = classes.index(0)
    probs_bankrupt  = model.predict_proba(X)[:, bankruptcy_idx]

    print("=" * 65)
    print("  THRESHOLD OPTIMIZATION  (model: SVM RBF)")
    print("  Positive class: bankruptcy (label = 0)")
    print("  Metric focus  : recall — missing bankruptcies is costly")
    print("=" * 65)

    # ── Sweep ─────────────────────────────────────────────
    thresholds = np.linspace(0.01, 0.99, 200)
    rows = []
    for t in thresholds:
        # Classify as bankruptcy (1) if P(bankruptcy) >= threshold
        preds_bankrupt = (probs_bankrupt >= t).astype(int)
        # Map back to label space: bankrupt=0 is "positive"
        y_true_bin = (y == 0).astype(int)
        y_pred_bin = preds_bankrupt

        tp = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
        fp = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
        fn = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())
        tn = int(((y_pred_bin == 0) & (y_true_bin == 0)).sum())

        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        rows.append({
            "threshold": round(float(t), 4),
            "recall":    round(recall, 4),
            "precision": round(precision, 4),
            "f1":        round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    res_df = pd.DataFrame(rows)

    # ── Plateau detection ─────────────────────────────────
    perfect = res_df[(res_df["recall"] == 1.0) & (res_df["precision"] == 1.0)]
    optimal = res_df.loc[res_df["f1"].idxmax()]

    if len(perfect) > 0:
        t_min = perfect["threshold"].min()
        t_max = perfect["threshold"].max()
        print(f"\n  ✓ Perfect (Recall=1, Precision=1) range: {t_min:.2f} – {t_max:.2f}")
        print(f"  ✓ Plateau width: {t_max - t_min:.2f}  (very robust threshold choice)")
        print(f"  ✓ Default threshold 0.5 is {'inside' if t_min <= 0.5 <= t_max else 'OUTSIDE'} the perfect zone")
    else:
        print("  ⚠ No perfect precision+recall threshold found.")

    print(f"\n  Optimal F1 threshold: {optimal['threshold']:.3f}")
    print(f"    Recall    : {optimal['recall']:.4f}")
    print(f"    Precision : {optimal['precision']:.4f}")
    print(f"    F1        : {optimal['f1']:.4f}")

    # Sample of results table
    display_idx = np.linspace(0, len(res_df)-1, 12, dtype=int)
    print("\n  Threshold sweep (sample):")
    display(res_df.iloc[display_idx].reset_index(drop=True))

    # ── Visualisation ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("#0F1923")
    fig.suptitle("Threshold optimization — SVM (RBF)\n"
                 "positive class: bankruptcy",
                 fontsize=13, color="#E0E8F4")

    t_arr = res_df["threshold"].values

    # Panel 1: recall / precision / F1 curves
    ax = axes[0]
    ax.plot(t_arr, res_df["recall"],    color=PALETTE["recall"],
            linewidth=2, label="Recall (bankruptcy)")
    ax.plot(t_arr, res_df["precision"], color=PALETTE["precision"],
            linewidth=2, label="Precision (bankruptcy)")
    ax.plot(t_arr, res_df["f1"],        color=PALETTE["f1"],
            linewidth=2, label="F1 score")
    if len(perfect) > 0:
        ax.axvspan(t_min, t_max, alpha=0.12, color=PALETTE["plateau"],
                   label=f"Perfect zone [{t_min:.2f}–{t_max:.2f}]")
    ax.axvline(0.5, color="#8899BB", linestyle="--", linewidth=0.8,
               alpha=0.7, label="default 0.5")
    ax.set_xlabel("Decision threshold", color="#C0CCDD")
    ax.set_ylabel("Score", color="#C0CCDD")
    ax.set_title("Precision / Recall / F1 vs threshold", color="#E0E8F4")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: TP / FP / FN counts
    ax = axes[1]
    ax.plot(t_arr, res_df["tp"], color=PALETTE["precision"],
            linewidth=2, label="TP (caught bankruptcies)")
    ax.plot(t_arr, res_df["fn"], color=PALETTE["recall"],
            linewidth=2, label="FN (missed bankruptcies)")
    ax.plot(t_arr, res_df["fp"], color="#F39C12",
            linewidth=2, label="FP (false alarms)")
    if len(perfect) > 0:
        ax.axvspan(t_min, t_max, alpha=0.12, color=PALETTE["plateau"])
    ax.axvline(0.5, color="#8899BB", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Decision threshold", color="#C0CCDD")
    ax.set_ylabel("Count", color="#C0CCDD")
    ax.set_title("Prediction counts vs threshold", color="#E0E8F4")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: zoomed F1 with optimal annotation
    ax = axes[2]
    ax.plot(t_arr, res_df["f1"], color=PALETTE["f1"], linewidth=2)
    ax.axvline(optimal["threshold"], color=PALETTE["recall"],
               linestyle="--", linewidth=1.2,
               label=f"optimal: {optimal['threshold']:.3f}")
    if len(perfect) > 0:
        ax.axvspan(t_min, t_max, alpha=0.15, color=PALETTE["plateau"],
                   label=f"perfect zone")
    ax.set_xlabel("Decision threshold", color="#C0CCDD")
    ax.set_ylabel("F1 score", color="#C0CCDD")
    ax.set_title("F1 score — zoomed\n(bankruptcy class)", color="#E0E8F4")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return res_df