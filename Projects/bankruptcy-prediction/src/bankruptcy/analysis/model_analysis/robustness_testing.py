"""
============================================================
Model Robustness Testing
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
Stress-test the trained model under controlled perturbations
to confirm it learned real patterns — not data artefacts.

Tests
-----
1. Baseline performance       — repeatedCV AUC (all 5 models)
2. Noise feature injection    — does a random column hurt?
3. Per-feature removal        — leave-one-out importance
4. Label noise (5 / 10 / 20%)— degradation under noisy labels
5. Label shuffle sanity       — random labels must give AUC ≈ 0.5

Changes from original
---------------------
  Bug 1: Hardcoded Windows absolute path.
         Fixed: accepts relative path or auto-detects artifacts/.
  Bug 2: Only tested LogisticRegression.
         Fixed: tests all 5 pipeline models.
  Bug 3: Feature removal hardcoded "industrial_risk".
         Fixed: iterates every feature programmatically.
  Bug 4: No visualisation — print only.
         Fixed: waterfall chart + grouped bar charts.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from IPython.display import display

# ==========================================================
# PALETTE
# ==========================================================

PALETTE = {
    "good":    "#2ECC71",
    "warn":    "#F39C12",
    "bad":     "#E84545",
    "neutral": "#4A9EFF",
    "muted":   "#8899BB",
    "gold":    "#C9A84C",
}

def _set_style():
    plt.rcParams.update({
        "figure.facecolor": "#0F1923", "axes.facecolor": "#0F1923",
        "axes.edgecolor": "#1C3355",   "axes.labelcolor": "#C0CCDD",
        "axes.titlecolor": "#E0E8F4",  "axes.titlesize": 12,
        "axes.labelsize": 10,          "xtick.color": "#8899BB",
        "ytick.color": "#8899BB",      "text.color": "#C0CCDD",
        "grid.color": "#1C3355",       "grid.linewidth": 0.5,
        "legend.facecolor": "#0F1923", "legend.edgecolor": "#1C3355",
    })

# ==========================================================
# MODEL REGISTRY  (matches pipeline models)
# ==========================================================

def _build_models():
    return {
        "LogisticRegression": Pipeline([
            ("s", StandardScaler()),
            ("m", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "SVM (RBF)": Pipeline([
            ("s", StandardScaler()),
            ("m", SVC(probability=True, class_weight="balanced", random_state=42)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "KNN": Pipeline([
            ("s", StandardScaler()),
            ("m", KNeighborsClassifier()),
        ]),
    }

def _cv_auc(model, X, y, cv):
    scores = cross_val_score(model, X, y,
                             cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean()), float(scores.std())

def _load_data(train_csv_path: str = None):
    """Load training data from path or auto-detect artifacts/."""
    if train_csv_path is None:
        # Auto-detect relative to this file or cwd
        candidates = [
            "artifacts/data_ingestion/train.csv",
            os.path.join(os.path.dirname(__file__), "..",
                         "artifacts/data_ingestion/train.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                train_csv_path = c
                break
        if train_csv_path is None:
            raise FileNotFoundError(
                "train.csv not found. Run the training pipeline first, "
                "or pass train_csv_path explicitly."
            )
    df = pd.read_csv(train_csv_path)
    X  = df.drop("class", axis=1)
    y  = df["class"].map({"bankruptcy": 0, "non-bankruptcy": 1})
    return X, y, list(X.columns)


# ==========================================================
# ROBUSTNESS TEST SUITE
# ==========================================================

def run_robustness_tests(train_csv_path: str = None,
                          primary_model: str = "SVM (RBF)") -> pd.DataFrame:
    """
    Run the full robustness test suite.

    Parameters
    ----------
    train_csv_path : str, optional
        Path to training CSV. Auto-detected if omitted.
    primary_model : str
        Which model from the registry to use for per-feature
        removal and label noise tests (default: SVM (RBF)).

    Returns
    -------
    pd.DataFrame  with all test results.
    """
    _set_style()
    X, y, features = _load_data(train_csv_path)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    models = _build_models()
    model  = models[primary_model]

    print("=" * 65)
    print(f"  ROBUSTNESS TESTING  (primary: {primary_model})")
    print("=" * 65)

    results = []

    # ── 1. Baseline — all 5 models ────────────────────────
    print("\n  1 — Baseline AUC (all 5 models)")
    baselines = {}
    for name, m in models.items():
        mean, std = _cv_auc(m, X, y, cv)
        baselines[name] = mean
        results.append({"test": f"Baseline — {name}", "mean_auc": mean,
                        "std": std, "drop": 0.0})
        print(f"     {name:<22}: {mean:.4f} ± {std:.4f}")

    base = baselines[primary_model]

    # ── 2. Noise feature injection ────────────────────────
    print(f"\n  2 — Random noise feature injection ({primary_model})")
    rng     = np.random.default_rng(42)
    X_noise = X.copy()
    X_noise["random_noise"] = rng.standard_normal(len(X))
    m2, s2  = _cv_auc(model, X_noise, y, cv)
    drop2   = base - m2
    results.append({"test": "Noise feature", "mean_auc": m2,
                    "std": s2, "drop": drop2})
    print(f"     AUC: {m2:.4f} ± {s2:.4f}  |  drop: {drop2:+.4f}")
    verdict = "✓ robust" if abs(drop2) < 0.005 else "⚠ sensitive to noise"
    print(f"     {verdict}")

    # ── 3. Per-feature removal ────────────────────────────
    print(f"\n  3 — Leave-one-out feature removal ({primary_model})")
    for f in features:
        Xd = X.drop(columns=[f])
        m3, s3 = _cv_auc(model, Xd, y, cv)
        drop3  = base - m3
        results.append({"test": f"Remove {f}", "mean_auc": m3,
                        "std": s3, "drop": drop3})
        flag = "★ critical" if drop3 > 0.01 else ("○ dispensable" if drop3 < 0 else "")
        print(f"     remove {f:<28}: {m3:.4f} ± {s3:.4f}  drop: {drop3:+.4f}  {flag}")

    # ── 4. Label noise ────────────────────────────────────
    print(f"\n  4 — Label noise injection ({primary_model})")
    for noise_pct in [5, 10, 20]:
        y_noisy  = y.copy()
        n_flip   = int(noise_pct / 100 * len(y))
        flip_idx = rng.choice(len(y), n_flip, replace=False)
        y_noisy.iloc[flip_idx] = 1 - y_noisy.iloc[flip_idx]
        m4, s4  = _cv_auc(model, X, y_noisy, cv)
        drop4   = base - m4
        results.append({"test": f"Label noise {noise_pct}%", "mean_auc": m4,
                        "std": s4, "drop": drop4})
        print(f"     {noise_pct}% noise: {m4:.4f} ± {s4:.4f}  drop: {drop4:+.4f}")

    # ── 5. Label shuffle sanity check ────────────────────
    print(f"\n  5 — Label shuffle sanity check ({primary_model})")
    y_shuffled = pd.Series(rng.permutation(y.values), index=y.index)
    m5, s5     = _cv_auc(model, X, y_shuffled, cv)
    results.append({"test": "Shuffled labels", "mean_auc": m5,
                    "std": s5, "drop": base - m5})
    verdict = "✓ passes sanity" if m5 < 0.6 else "✗ FAIL — possible data leakage"
    print(f"     AUC: {m5:.4f} ± {s5:.4f}  {verdict}")

    # ── Results table ─────────────────────────────────────
    res_df = pd.DataFrame(results)
    print("\n  Full results:")
    display(res_df.round(4))

    # ── Visualisation ─────────────────────────────────────
    _plot_robustness(res_df, base, primary_model, features)

    return res_df


def _plot_robustness(res_df: pd.DataFrame, base: float,
                     model_name: str, features: list):

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.patch.set_facecolor("#0F1923")
    fig.suptitle(f"Robustness analysis — {model_name}",
                 fontsize=13, color="#E0E8F4")

    # Panel 1: baseline all models
    baselines = res_df[res_df["test"].str.startswith("Baseline")]
    names     = [t.replace("Baseline — ", "") for t in baselines["test"]]
    colors    = [PALETTE["gold"] if n == model_name else PALETTE["neutral"]
                 for n in names]
    ax = axes[0]
    bars = ax.barh(names, baselines["mean_auc"], color=colors,
                   height=0.5, edgecolor="#1C3355")
    ax.set_xlim(0.95, 1.01)
    ax.axvline(1.0, color="#8899BB", linestyle="--", linewidth=0.8)
    ax.set_title("Baseline AUC — all models", color="#E0E8F4")
    ax.set_xlabel("ROC-AUC", color="#C0CCDD")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, baselines["mean_auc"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", color="#E0E8F4", fontsize=9)

    # Panel 2: feature removal waterfall
    removals = res_df[res_df["test"].str.startswith("Remove")]
    feat_names = [t.replace("Remove ", "") for t in removals["test"]]
    drops = removals["drop"].values
    colors2 = [PALETTE["bad"] if d > 0.005 else
               PALETTE["warn"] if d > 0 else
               PALETTE["good"] for d in drops]
    ax = axes[1]
    ax.barh(feat_names, drops, color=colors2, height=0.5, edgecolor="#1C3355")
    ax.axvline(0, color="#8899BB", linewidth=1)
    ax.set_title("AUC drop after feature removal\n(red = critical feature)",
                 color="#E0E8F4")
    ax.set_xlabel("ΔAUC (baseline − after removal)", color="#C0CCDD")
    ax.grid(axis="x", alpha=0.3)
    for feat, d in zip(feat_names, drops):
        ax.text(d + (0.0002 if d >= 0 else -0.0002), feat,
                f"{d:+.4f}", ha="left" if d >= 0 else "right",
                va="center", color="#E0E8F4", fontsize=9)

    # Panel 3: label noise degradation
    noise_rows = res_df[res_df["test"].str.contains("noise|shuffle", case=False)]
    ax = axes[2]
    x_labels = noise_rows["test"].tolist()
    colors3   = [PALETTE["warn"] if "shuffle" not in t.lower() else PALETTE["bad"]
                 for t in x_labels]
    bars3 = ax.bar(range(len(x_labels)), noise_rows["mean_auc"],
                   color=colors3, width=0.5, edgecolor="#1C3355")
    ax.axhline(base, color=PALETTE["gold"], linestyle="--",
               linewidth=1, label=f"baseline {base:.4f}")
    ax.axhline(0.5, color="#8899BB", linestyle=":", linewidth=0.8,
               alpha=0.6, label="random classifier")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("AUC under noise / shuffle", color="#E0E8F4")
    ax.set_ylabel("ROC-AUC", color="#C0CCDD")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars3, noise_rows["mean_auc"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.01, f"{val:.3f}",
                ha="center", va="bottom", color="#E0E8F4", fontsize=9)

    plt.tight_layout()
    plt.show()