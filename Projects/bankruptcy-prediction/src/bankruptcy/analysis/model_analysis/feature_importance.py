"""
============================================================
Feature Importance Analysis
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
Explain WHY the model predicts bankruptcy using SHAP
(SHapley Additive exPlanations) — the gold standard for
model explainability in production ML systems.

SHAP answers: "For this specific prediction, how much did
each feature push the probability toward or away from
bankruptcy?"

Why SHAP over simpler alternatives
------------------------------------
• Feature permutation importance shows GLOBAL importance
  but loses per-sample direction.
• Coefficients only work for linear models.
• SHAP works for ANY model, shows direction and magnitude,
  and satisfies mathematical consistency guarantees.

Visualisations produced
-----------------------
1. SHAP summary plot (beeswarm)  — global + directional
2. SHAP bar chart                — mean absolute importance
3. SHAP waterfall (single sample)— individual prediction
4. SHAP dependency plots         — feature interaction
5. Permutation importance        — model-agnostic fallback

Usage
-----
    from bankruptcy.analysis.feature_importance import FeatureImportanceAnalyzer
    fi = FeatureImportanceAnalyzer()
    fi.analyze(df, "class")          # full suite
    fi.shap_summary(df, "class")     # beeswarm only
    fi.shap_waterfall(df, "class", sample_idx=0)  # single prediction
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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from IPython.display import display



def _normalize_shap_values(shap_values):
    """
    Normalize SHAP outputs across versions so we always get
    an array of shape (samples, features).
    """
    import numpy as np

    # Handle Explanation objects (new SHAP API)
    if hasattr(shap_values, "values"):
        sv = shap_values.values
    else:
        sv = shap_values

    # Old SHAP format -> list of arrays
    if isinstance(sv, list):
        sv = sv[1]

    sv = np.array(sv)

    # Handle 3D outputs
    if sv.ndim == 3:
        # Could be (samples, features, classes)
        if sv.shape[1] < sv.shape[2]:
            sv = sv[:, :, 1]
        else:
            sv = sv[:, 1, :]

    return sv

# ==========================================================
# PALETTE
# ==========================================================

PALETTE = {
    "bankruptcy":     "#E84545",
    "non-bankruptcy": "#2ECC71",
    "neutral":        "#4A9EFF",
    "mid":            "#F39C12",
    "muted":          "#8899BB",
    "gold":           "#C9A84C",
}

FEATURE_LABELS = {
    "industrial_risk":       "Industrial Risk",
    "management_risk":       "Management Risk",
    "financial_flexibility": "Financial Flexibility",
    "credibility":           "Credibility",
    "competitiveness":       "Competitiveness",
    "operating_risk":        "Operating Risk",
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

def _safe_numeric(s):
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce")

def _load_data(df: pd.DataFrame, target: str):
    features = [c for c in df.columns if c != target]
    X = df[features].apply(_safe_numeric)
    y = (df[target].astype(str).str.strip() == "bankruptcy").astype(int)
    return X, y, features


# ==========================================================
# FEATURE IMPORTANCE ANALYZER
# ==========================================================

class FeatureImportanceAnalyzer:
    """
    SHAP-based and permutation-based feature importance
    for the bankruptcy prediction model.

    Parameters
    ----------
    target_column : str
    use_rf : bool
        If True, uses RandomForest (SHAP TreeExplainer is
        faster for tree models). If False, uses SVM with
        KernelExplainer (slower but more accurate for SVM).
    """

    def __init__(self, target_column: str = "class",
                 use_rf: bool = True):
        self.target_column = target_column
        self.use_rf        = use_rf
        _set_style()

    # --------------------------------------------------------
    # PUBLIC: full suite
    # --------------------------------------------------------

    def analyze(self, df: pd.DataFrame,
                target_column: str = None) -> None:
        """
        Full feature importance analysis:
        1. Permutation importance (model-agnostic)
        2. SHAP summary + bar chart
        3. SHAP waterfall for a bankrupt and a safe sample
        4. SHAP dependency plots for top 2 features
        """
        tgt = target_column or self.target_column

        print("=" * 65)
        print("  FEATURE IMPORTANCE ANALYSIS")
        print("=" * 65)

        self.permutation_importance(df, tgt)

        try:
            import shap
            self.shap_summary(df, tgt)
            self.shap_waterfall(df, tgt, sample_idx="bankrupt")
            self.shap_waterfall(df, tgt, sample_idx="safe")
            self.shap_dependency(df, tgt)
        except ImportError:
            print("\n  ⚠  SHAP not installed. Install with: pip install shap")
            print("  Showing permutation importance only.")

    # --------------------------------------------------------
    # 1. PERMUTATION IMPORTANCE  (no SHAP dependency)
    # --------------------------------------------------------

    def permutation_importance(self, df: pd.DataFrame,
                                target_column: str = None) -> None:
        """
        Model-agnostic permutation importance using RandomForest.

        Shuffles each feature column and measures the AUC drop.
        Works without SHAP installed.
        """
        tgt = target_column or self.target_column
        X, y, features = _load_data(df, tgt)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )
        model.fit(X_tr, y_tr)

        print("\n  Permutation importance (RandomForest, AUC metric)")

        result = permutation_importance(
            model, X_te, y_te,
            n_repeats=20, random_state=42,
            scoring="roc_auc", n_jobs=-1
        )

        perm_df = pd.DataFrame({
            "Feature":   features,
            "Mean drop": result.importances_mean.round(4),
            "Std":       result.importances_std.round(4),
        }).sort_values("Mean drop", ascending=False).set_index("Feature")
        display(perm_df)

        # ── Also: RF built-in MDI importance ─────────────
        mdi = pd.Series(model.feature_importances_, index=features)
        mdi = mdi.sort_values(ascending=False)

        # Visualise both
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0F1923")
        fig.suptitle("Feature importance — RandomForest",
                     fontsize=13, color="#E0E8F4")

        # Permutation
        sorted_perm = perm_df.sort_values("Mean drop")
        colors = [PALETTE["gold"] if v > 0.01 else
                  PALETTE["neutral"] if v > 0.001 else PALETTE["muted"]
                  for v in sorted_perm["Mean drop"]]
        ax = axes[0]
        ax.barh(sorted_perm.index, sorted_perm["Mean drop"],
                xerr=sorted_perm["Std"], color=colors,
                height=0.5, edgecolor="#1C3355",
                error_kw={"ecolor": PALETTE["muted"], "capsize": 3})
        ax.axvline(0, color="#8899BB", linewidth=0.8)
        ax.set_title("Permutation importance\n(AUC drop when feature is shuffled)",
                     color="#E0E8F4")
        ax.set_xlabel("Mean AUC drop", color="#C0CCDD")
        ax.grid(axis="x", alpha=0.3)
        for feat, val in zip(sorted_perm.index, sorted_perm["Mean drop"]):
            ax.text(val + 0.0005, sorted_perm.index.tolist().index(feat),
                    f"{val:.4f}", va="center", color="#E0E8F4", fontsize=9)

        # MDI
        sorted_mdi = mdi.sort_values()
        colors2 = [PALETTE["gold"] if v > 0.2 else
                   PALETTE["neutral"] if v > 0.1 else PALETTE["muted"]
                   for v in sorted_mdi]
        ax = axes[1]
        ax.barh(sorted_mdi.index, sorted_mdi.values,
                color=colors2, height=0.5, edgecolor="#1C3355")
        ax.set_title("Mean Decrease Impurity\n(RF built-in importance)",
                     color="#E0E8F4")
        ax.set_xlabel("Importance", color="#C0CCDD")
        ax.grid(axis="x", alpha=0.3)
        for feat, val in zip(sorted_mdi.index, sorted_mdi.values):
            ax.text(val + 0.003, sorted_mdi.index.tolist().index(feat),
                    f"{val:.3f}", va="center", color="#E0E8F4", fontsize=9)

        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # 2. SHAP SUMMARY  (beeswarm + bar)
    # --------------------------------------------------------

    def shap_summary(self, df: pd.DataFrame,
                     target_column: str = None) -> None:
        """
        SHAP beeswarm and bar summary plots.

        Each dot in the beeswarm is one sample.
        Colour = feature value (red=high, blue=low).
        x-axis = SHAP value (positive = pushes toward bankruptcy).
        """
        import shap
        tgt = target_column or self.target_column
        X, y, features = _load_data(df, tgt)

        print("\n  Computing SHAP values (TreeExplainer)...")
        model = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )
        model.fit(X, y)

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For binary RF: shap_values is a list [class0, class1]
        # We want class 1 = bankruptcy (encoded as 1 here)
        # Normalize SHAP output
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
            if isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1]

        print("  Mean |SHAP| per feature (global importance):")
        mean_shap = pd.DataFrame({
            "Feature":       features,
            "Mean |SHAP|":   np.abs(sv).mean(axis=0).round(4),
        }).sort_values("Mean |SHAP|", ascending=False).set_index("Feature")
        display(mean_shap)

        # Beeswarm
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor("#0F1923")
        fig.suptitle("SHAP feature importance", fontsize=13, color="#E0E8F4")

        plt.sca(axes[0])
        shap.summary_plot(sv, X, feature_names=features,
                          show=False, plot_size=None)
        axes[0].set_title("SHAP beeswarm\n(red=high feature, positive=→bankruptcy)",
                          color="#E0E8F4")

        plt.sca(axes[1])
        shap.summary_plot(sv, X, feature_names=features,
                          plot_type="bar", show=False, plot_size=None)
        axes[1].set_title("Mean |SHAP| — global importance",
                          color="#E0E8F4")

        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # 3. SHAP WATERFALL  (single prediction)
    # --------------------------------------------------------

    def shap_waterfall(self, df: pd.DataFrame,
                       target_column: str = None,
                       sample_idx = "bankrupt") -> None:
        """
        SHAP waterfall plot for a single sample.

        Shows how each feature pushed the prediction from the
        base rate up or down to the final output.

        Parameters
        ----------
        sample_idx : int | 'bankrupt' | 'safe'
            Row index, or 'bankrupt'/'safe' to auto-select
            a clear example of each class.
        """
        import shap
        tgt = target_column or self.target_column
        X, y, features = _load_data(df, tgt)

        model = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )
        model.fit(X, y)
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
            if isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1]

        if sample_idx == "bankrupt":
            # Find a sample where model is most confident about bankruptcy
            probs = model.predict_proba(X)[:, 1]
            idx   = int(np.argmax(probs * (y == 1)))
            label = "bankruptcy"
        elif sample_idx == "safe":
            probs = model.predict_proba(X)[:, 0]
            idx   = int(np.argmax(probs * (y == 0)))
            label = "non-bankruptcy"
        else:
            idx   = int(sample_idx)
            label = "bankruptcy" if y.iloc[idx] == 1 else "non-bankruptcy"

        print(f"\n  SHAP waterfall — sample {idx}  (true label: {label})")
        print(f"  Feature values: {X.iloc[idx].to_dict()}")

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0F1923")

        sample_sv = sv[idx]
        base_val  = float(explainer.expected_value[1]
                          if isinstance(explainer.expected_value, (list, np.ndarray))
                          else explainer.expected_value)

        # Manual waterfall bar chart
        sorted_idx = np.argsort(np.abs(sample_sv))[::-1]
        labels     = [features[i] for i in sorted_idx]
        values     = [sample_sv[i] for i in sorted_idx]
        cumulative = base_val + np.cumsum([0] + values)

        bar_colors = [PALETTE["bankruptcy"] if v > 0
                      else PALETTE["non-bankruptcy"]
                      for v in values]

        ax.barh(labels, values, color=bar_colors,
                height=0.5, edgecolor="#1C3355")
        ax.axvline(0, color="#8899BB", linewidth=1)
        ax.set_title(
            f"SHAP waterfall — {label}\n"
            f"(red = pushes toward bankruptcy, green = away)",
            color="#E0E8F4"
        )
        ax.set_xlabel("SHAP value", color="#C0CCDD")
        ax.grid(axis="x", alpha=0.3)
        for feat, val in zip(labels, values):
            ax.text(val + (0.005 if val >= 0 else -0.005),
                    labels.index(feat),
                    f"{val:+.3f}",
                    ha="left" if val >= 0 else "right",
                    va="center", color="#E0E8F4", fontsize=9)

        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # 4. SHAP DEPENDENCY  (top 2 features)
    # --------------------------------------------------------

    def shap_dependency(self, df: pd.DataFrame,
                         target_column: str = None) -> None:
        """
        SHAP dependency plots for the top 2 features.

        Shows how SHAP value changes with feature value,
        with interaction effect coloured by a second feature.
        """
        import shap
        tgt = target_column or self.target_column
        X, y, features = _load_data(df, tgt)

        model = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )
        model.fit(X, y)
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
            if isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1]

        mean_abs = np.abs(sv).mean(axis=0)
        top2     = np.argsort(mean_abs)[::-1][:2]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0F1923")
        fig.suptitle("SHAP dependency plots — top 2 features",
                     fontsize=13, color="#E0E8F4")

        for ax, feat_idx in zip(axes, top2):
            feat  = features[feat_idx]
            # Interaction coloured by 2nd top feature
            inter = features[top2[1]] if feat_idx == top2[0] else features[top2[0]]

            scatter = ax.scatter(
                X.iloc[:, feat_idx],
                sv[:, feat_idx],
                c=X[inter],
                cmap=mcolors.LinearSegmentedColormap.from_list(
                    "safe_risky", [PALETTE["non-bankruptcy"],
                                   PALETTE["neutral"],
                                   PALETTE["bankruptcy"]]
                ),
                alpha=0.7, s=40, edgecolors="none"
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(f"{inter}\n(colour)", color="#C0CCDD")
            cbar.ax.yaxis.set_tick_params(color="#8899BB")

            ax.axhline(0, color="#8899BB", linewidth=0.8, linestyle="--")
            ax.set_xlabel(feat, color="#C0CCDD")
            ax.set_ylabel("SHAP value", color="#C0CCDD")
            ax.set_title(
                f"{feat}\n(positive = pushes toward bankruptcy)",
                color="#E0E8F4"
            )
            ax.grid(alpha=0.2)

        plt.tight_layout()
        plt.show()