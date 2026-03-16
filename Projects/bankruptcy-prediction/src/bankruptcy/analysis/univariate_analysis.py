"""
============================================================
Univariate Analysis Module
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
Perform univariate analysis on every feature in the
bankruptcy prevention dataset and produce publication-ready
visualizations suitable for a Jupyter notebook.

Why the original module was wrong
----------------------------------
The original module used sns.histplot + KDE and IQR-based
outlier detection.  Both are inappropriate for this dataset:

• Every numeric feature (industrial_risk, management_risk,
  financial_flexibility, credibility, competitiveness,
  operating_risk) has exactly THREE distinct values:
  0.0 (low), 0.5 (medium), 1.0 (high).

  → These are ORDINAL categorical features, not continuous.
  → KDE draws a fake smooth curve over 3 points.  Misleading.
  → Skewness on 3-value ordinal is technically computable
    but meaningless as a summary statistic.
  → IQR = 0 for features concentrated at one value, so
    every row (or no row) flags as an "outlier".  Useless.

Correct approach for ordinal 3-level features
----------------------------------------------
• Frequency / proportion table   (value_counts)
• Shannon entropy                (how evenly spread are the 3 levels?)
• Gini impurity                  (class diversity measure)
• Mode                           (most common level)
• Class-conditional means        (mean per bankruptcy class)
• Point-biserial correlation     (correct for ordinal vs binary target)
• Cramér's V                     (association strength, chi-square based)
• Stacked bar chart by class     (shows class separation at a glance)
• Side-by-side count chart       (raw counts per level per class)

For the target column ('class'):
• Simple count + proportion bar
• Imbalance ratio

Design
------
Two public methods:

    UnivariateAnalyzer().analyze(df, feature)
        Full single-feature report: stats table + 3 charts.

    UnivariateAnalyzer().analyze_all(df, target_column="class")
        Runs analyze() on every feature, then a combined
        summary comparison chart across all features.
"""

# ==========================================================
# IMPORTS
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from IPython.display import display
from scipy.stats import pointbiserialr, chi2_contingency


# ==========================================================
# COLOUR PALETTE
# ==========================================================
# Consistent across all analysis modules in this project.

PALETTE = {
    "bankruptcy":     "#E84545",   # red
    "non-bankruptcy": "#2ECC71",   # green
    "neutral":        "#4A9EFF",   # blue
    "mid":            "#F39C12",   # amber
    "muted":          "#8899BB",
}

CLASS_PALETTE = [PALETTE["bankruptcy"], PALETTE["non-bankruptcy"]]

LEVEL_COLOURS = {
    0.0: "#4A9EFF",   # low / blue
    0.5: "#F39C12",   # medium / amber
    1.0: "#E84545",   # high / red
}

# Label for each ordinal level — used in chart annotations
LEVEL_LABELS = {0.0: "Low (0.0)", 0.5: "Medium (0.5)", 1.0: "High (1.0)"}


# ==========================================================
# HELPERS
# ==========================================================

def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convert series to numeric, coercing errors to NaN."""
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")


def _is_ordinal_feature(series: pd.Series) -> bool:
    """
    Return True if the series looks like an ordinal 3-level
    feature (only values 0.0, 0.5, 1.0 after numeric conversion).
    """
    num = _safe_numeric(series).dropna()
    return set(num.unique()).issubset({0.0, 0.5, 1.0}) and num.notna().sum() > 0


def _cramers_v(series_a: pd.Series, series_b: pd.Series) -> float:
    """
    Compute Cramér's V association between two categorical series.
    Returns value in [0, 1] where 1 = perfect association.
    """
    ct = pd.crosstab(series_a, series_b)
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    min_dim = min(ct.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def _shannon_entropy(series: pd.Series) -> float:
    """Shannon entropy in bits (log base 2)."""
    probs = series.value_counts(normalize=True)
    return float(-sum(p * np.log2(p) for p in probs if p > 0))


def _gini_impurity(series: pd.Series) -> float:
    """Gini impurity: 0 = pure, 1 = maximally mixed."""
    probs = series.value_counts(normalize=True)
    return float(1 - sum(p ** 2 for p in probs))


def _set_style():
    """Apply consistent dark-style chart theme."""
    plt.rcParams.update({
        "figure.facecolor":  "#0F1923",
        "axes.facecolor":    "#0F1923",
        "axes.edgecolor":    "#1C3355",
        "axes.labelcolor":   "#C0CCDD",
        "axes.titlecolor":   "#E0E8F4",
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "xtick.color":       "#8899BB",
        "ytick.color":       "#8899BB",
        "text.color":        "#C0CCDD",
        "grid.color":        "#1C3355",
        "grid.linewidth":    0.6,
        "legend.facecolor":  "#0F1923",
        "legend.edgecolor":  "#1C3355",
        "legend.fontsize":   10,
        "font.family":       "DejaVu Sans",
    })


# ==========================================================
# UNIVARIATE ANALYZER
# ==========================================================

class UnivariateAnalyzer:
    """
    Production-grade univariate analysis for the bankruptcy
    prevention dataset.

    Handles:
    • ordinal 3-level numeric features  (0 / 0.5 / 1)
    • binary string target column       ('bankruptcy' / 'non-bankruptcy')

    Usage
    -----
    >>> ua = UnivariateAnalyzer()
    >>> ua.analyze(df, "competitiveness")
    >>> ua.analyze_all(df, target_column="class")
    """

    def __init__(self, target_column: str = "class"):
        self.target_column = target_column
        _set_style()

    # ----------------------------------------------------------
    # PUBLIC: single feature
    # ----------------------------------------------------------

    def analyze(self, df: pd.DataFrame, feature: str) -> None:
        """
        Full univariate report for one feature.

        Produces:
        1. Statistics table  (frequency, entropy, Cramér's V, …)
        2. Count chart       (absolute + relative counts per level)
        3. Stacked class bar (shows class split at each level)
        4. Class-conditional means bar

        Parameters
        ----------
        df      : DataFrame containing the feature and target columns.
        feature : Column name to analyse.
        """
        if feature not in df.columns:
            print(f"❌  Feature '{feature}' not found in dataset.")
            return

        series = df[feature]

        # ── Route to correct handler ──────────────────────────
        if feature == self.target_column:
            self._analyze_target(df)
        elif _is_ordinal_feature(series):
            self._analyze_ordinal(df, feature)
        else:
            self._analyze_categorical(df, feature)

    # ----------------------------------------------------------
    # PUBLIC: all features
    # ----------------------------------------------------------

    def analyze_all(self, df: pd.DataFrame,
                    target_column: str = "class") -> None:
        """
        Run analyze() on every column, then draw a combined
        cross-feature comparison chart (Cramér's V + entropy).

        Parameters
        ----------
        df            : Full dataset.
        target_column : Name of the binary target column.
        """
        self.target_column = target_column

        for col in df.columns:
            self.analyze(df, col)

        # Cross-feature summary chart
        self._summary_comparison_chart(df, target_column)

    # ----------------------------------------------------------
    # ORDINAL FEATURE ANALYSIS
    # ----------------------------------------------------------

    def _analyze_ordinal(self, df: pd.DataFrame, feature: str) -> None:

        series   = _safe_numeric(df[feature]).dropna()
        has_tgt  = self.target_column in df.columns

        # ── Stats table ───────────────────────────────────────
        print("\n" + "=" * 62)
        print(f"  UNIVARIATE ANALYSIS — {feature.upper().replace('_', ' ')}")
        print("=" * 62)

        vc      = series.value_counts().sort_index()
        pct     = (vc / len(series) * 100).round(2)
        entropy = _shannon_entropy(series)
        gini    = _gini_impurity(series)
        mode    = series.mode()[0]

        stats_df = pd.DataFrame({
            "Level":       [LEVEL_LABELS[v] for v in vc.index],
            "Count":       vc.values,
            "Proportion":  (pct / 100).round(4),
            "Percent (%)": pct.values,
        })

        display(stats_df.set_index("Level"))

        # Information-theoretic stats
        info_df = pd.DataFrame({
            "Metric": [
                "Mode",
                "Shannon entropy (bits)",
                "Gini impurity",
                "Max entropy (uniform)",
            ],
            "Value": [
                LEVEL_LABELS.get(mode, str(mode)),
                f"{entropy:.4f}",
                f"{gini:.4f}",
                f"{np.log2(3):.4f}  (3-class uniform = 1.585 bits)",
            ],
        })
        display(info_df.set_index("Metric"))

        # Target association (if target present)
        if has_tgt:
            y_bin = (df[self.target_column] == "bankruptcy").astype(int)
            r_pb, p_pb = pointbiserialr(series, y_bin)
            cv          = _cramers_v(
                series.astype(str),
                df[self.target_column].astype(str)
            )
            bk_mean  = df[df[self.target_column] == "bankruptcy"][feature]
            bk_mean  = _safe_numeric(bk_mean).mean()
            nb_mean  = df[df[self.target_column] == "non-bankruptcy"][feature]
            nb_mean  = _safe_numeric(nb_mean).mean()

            assoc_df = pd.DataFrame({
                "Metric": [
                    "Point-biserial r (vs target)",
                    "p-value",
                    "Cramér's V (association strength)",
                    "Mean — bankruptcy companies",
                    "Mean — non-bankruptcy companies",
                    "Mean difference",
                ],
                "Value": [
                    f"{r_pb:.4f}",
                    f"{p_pb:.2e}  {'✓ significant' if p_pb < 0.05 else '✗ not significant'}",
                    f"{cv:.4f}  {'★ strong' if cv > 0.5 else ('◆ moderate' if cv > 0.2 else '○ weak')}",
                    f"{bk_mean:.4f}",
                    f"{nb_mean:.4f}",
                    f"{abs(bk_mean - nb_mean):.4f}",
                ],
            })
            print("\n  Target association:")
            display(assoc_df.set_index("Metric"))

        # ── Charts ────────────────────────────────────────────
        if has_tgt:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            fig.patch.set_facecolor("#0F1923")
            fig.suptitle(
                f"Univariate analysis — {feature.replace('_', ' ').title()}",
                fontsize=14, color="#E0E8F4", y=1.02
            )
            self._plot_count_bar(series, axes[0], feature)
            self._plot_stacked_class_bar(df, feature, axes[1])
            self._plot_class_conditional_means(df, feature, axes[2])
        else:
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            fig.patch.set_facecolor("#0F1923")
            fig.suptitle(
                f"Univariate analysis — {feature.replace('_', ' ').title()}",
                fontsize=14, color="#E0E8F4", y=1.02
            )
            self._plot_count_bar(series, axes[0], feature)
            self._plot_proportion_pie(series, axes[1], feature)

        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------
    # TARGET COLUMN ANALYSIS
    # ----------------------------------------------------------

    def _analyze_target(self, df: pd.DataFrame) -> None:

        print("\n" + "=" * 62)
        print("  UNIVARIATE ANALYSIS — TARGET: CLASS")
        print("=" * 62)

        series  = df[self.target_column].astype(str).str.strip()
        vc      = series.value_counts()
        pct     = (vc / len(series) * 100).round(2)
        ratio   = vc.min() / vc.max()

        dist_df = pd.DataFrame({
            "Class":        vc.index,
            "Count":        vc.values,
            "Percent (%)":  pct.values,
        }).set_index("Class")
        display(dist_df)

        imbalance_df = pd.DataFrame({
            "Metric": [
                "Imbalance ratio (minority / majority)",
                "Assessment",
                "Entropy (bits)",
            ],
            "Value": [
                f"{ratio:.4f}",
                "✓ Mild imbalance (ratio > 0.7 — acceptable)" if ratio > 0.7
                else "⚠ Moderate imbalance — consider class weights",
                f"{_shannon_entropy(series):.4f}",
            ],
        })
        display(imbalance_df.set_index("Metric"))

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.patch.set_facecolor("#0F1923")
        fig.suptitle("Target distribution — class", fontsize=14,
                     color="#E0E8F4", y=1.02)

        # Count bar
        bars = axes[0].bar(
            vc.index, vc.values,
            color=CLASS_PALETTE, width=0.45, edgecolor="#1C3355"
        )
        axes[0].set_title("Absolute counts", color="#E0E8F4")
        axes[0].set_xlabel("Class", color="#C0CCDD")
        axes[0].set_ylabel("Count", color="#C0CCDD")
        axes[0].grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vc.values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1, str(val),
                ha="center", va="bottom", color="#E0E8F4", fontsize=11
            )

        # Donut
        wedges, texts, autotexts = axes[1].pie(
            vc.values, labels=vc.index,
            colors=CLASS_PALETTE,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "#0F1923", "linewidth": 2},
            pctdistance=0.78,
        )
        for t in texts:
            t.set_color("#C0CCDD")
        for at in autotexts:
            at.set_color("#0F1923")
            at.set_fontweight("bold")
        centre = plt.Circle((0, 0), 0.55, color="#0F1923")
        axes[1].add_patch(centre)
        axes[1].text(0, 0, f"n={len(series)}", ha="center", va="center",
                     color="#E0E8F4", fontsize=12, fontweight="bold")
        axes[1].set_title("Proportion", color="#E0E8F4")

        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------
    # CATEGORICAL (non-ordinal) FALLBACK
    # ----------------------------------------------------------

    def _analyze_categorical(self, df: pd.DataFrame, feature: str) -> None:

        print("\n" + "=" * 62)
        print(f"  UNIVARIATE ANALYSIS — {feature.upper().replace('_', ' ')}")
        print("=" * 62)

        series  = df[feature].astype(str).str.strip()
        vc      = series.value_counts()
        pct     = (vc / len(series) * 100).round(2)

        dist_df = pd.DataFrame({
            "Count":        vc.values,
            "Percent (%)":  pct.values,
        }, index=vc.index)
        display(dist_df)

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#0F1923")
        bars = ax.bar(
            vc.index, vc.values,
            color=PALETTE["neutral"], edgecolor="#1C3355"
        )
        ax.set_title(f"Distribution of {feature}", color="#E0E8F4")
        ax.set_xlabel(feature, color="#C0CCDD")
        ax.set_ylabel("Count", color="#C0CCDD")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        for bar, val in zip(bars, vc.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, str(val),
                ha="center", va="bottom", color="#E0E8F4", fontsize=10
            )
        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------
    # CHART HELPERS
    # ----------------------------------------------------------

    def _plot_count_bar(self, series: pd.Series, ax, feature: str) -> None:
        """Absolute count bar — one bar per ordinal level."""
        vc     = series.value_counts().sort_index()
        colors = [LEVEL_COLOURS.get(v, PALETTE["neutral"]) for v in vc.index]
        bars   = ax.bar(
            [LEVEL_LABELS[v] for v in vc.index],
            vc.values, color=colors, width=0.5, edgecolor="#1C3355"
        )
        ax.set_title("Count per level", color="#E0E8F4")
        ax.set_xlabel("Level", color="#C0CCDD")
        ax.set_ylabel("Count", color="#C0CCDD")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vc.values):
            pct = val / series.notna().sum() * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val}\n({pct:.0f}%)",
                ha="center", va="bottom", color="#E0E8F4", fontsize=9
            )

    def _plot_stacked_class_bar(self, df: pd.DataFrame,
                                feature: str, ax) -> None:
        """
        Stacked horizontal bar chart.
        For each ordinal level: proportion of bankruptcy vs non-bankruptcy.
        This is the most informative chart for ordinal vs binary target.
        """
        levels  = [0.0, 0.5, 1.0]
        y_pos   = np.arange(len(levels))
        bk_pcts, nb_pcts, totals = [], [], []

        for lv in levels:
            subset = df[_safe_numeric(df[feature]) == lv]
            total  = len(subset)
            if total == 0:
                bk_pcts.append(0); nb_pcts.append(0); totals.append(0)
                continue
            bk = (subset[self.target_column] == "bankruptcy").sum()
            bk_pcts.append(bk / total * 100)
            nb_pcts.append((total - bk) / total * 100)
            totals.append(total)

        ax.barh(y_pos, bk_pcts,
                color=PALETTE["bankruptcy"], height=0.45,
                label="Bankruptcy", edgecolor="#0F1923")
        ax.barh(y_pos, nb_pcts, left=bk_pcts,
                color=PALETTE["non-bankruptcy"], height=0.45,
                label="Non-bankruptcy", edgecolor="#0F1923")

        ax.set_yticks(y_pos)
        ax.set_yticklabels([LEVEL_LABELS[lv] for lv in levels],
                           color="#C0CCDD")
        ax.set_xlabel("Percentage (%)", color="#C0CCDD")
        ax.set_title("Class split per level", color="#E0E8F4")
        ax.set_xlim(0, 100)
        ax.axvline(50, color="#8899BB", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.grid(axis="x", alpha=0.3)

        # Annotate bankruptcy % inside bar
        for i, (bk, tot) in enumerate(zip(bk_pcts, totals)):
            if tot > 0:
                ax.text(bk / 2, i, f"{bk:.0f}%",
                        ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)

        # Right-side total count labels
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"n={t}" for t in totals], color="#8899BB",
                            fontsize=9)
        ax2.tick_params(axis="y", length=0)

    def _plot_class_conditional_means(self, df: pd.DataFrame,
                                      feature: str, ax) -> None:
        """
        Bar chart of mean feature value per class.
        With error bars showing ±1 std.
        """
        classes = ["bankruptcy", "non-bankruptcy"]
        means, stds = [], []
        for cls in classes:
            vals = _safe_numeric(df[df[self.target_column] == cls][feature])
            means.append(vals.mean())
            stds.append(vals.std())

        bars = ax.bar(
            classes, means,
            color=CLASS_PALETTE, width=0.45, edgecolor="#1C3355",
            yerr=stds, capsize=5,
            error_kw={"ecolor": "#8899BB", "elinewidth": 1.2}
        )
        ax.set_ylim(0, 1.2)
        ax.set_title("Mean value per class (±1 std)", color="#E0E8F4")
        ax.set_ylabel("Mean level", color="#C0CCDD")
        ax.set_xlabel("Class", color="#C0CCDD")
        ax.axhline(0.5, color="#8899BB", linestyle="--",
                   linewidth=0.8, alpha=0.6, label="midpoint")
        ax.grid(axis="y", alpha=0.3)
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + 0.04,
                f"{mean:.3f}",
                ha="center", va="bottom", color="#E0E8F4", fontsize=10,
                fontweight="bold"
            )

    def _plot_proportion_pie(self, series: pd.Series,
                             ax, feature: str) -> None:
        """Simple donut chart — proportion of each level."""
        vc = series.value_counts().sort_index()
        colors = [LEVEL_COLOURS.get(v, PALETTE["neutral"]) for v in vc.index]

        wedges, texts, autotexts = ax.pie(
            vc.values,
            labels=[LEVEL_LABELS[v] for v in vc.index],
            colors=colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "#0F1923", "linewidth": 2},
            pctdistance=0.78,
        )
        for t in texts:
            t.set_color("#C0CCDD"); t.set_fontsize(9)
        for at in autotexts:
            at.set_color("#0F1923"); at.set_fontweight("bold")
        centre = plt.Circle((0, 0), 0.55, color="#0F1923")
        ax.add_patch(centre)
        ax.set_title("Level proportions", color="#E0E8F4")

    # ----------------------------------------------------------
    # SUMMARY COMPARISON CHART (all features)
    # ----------------------------------------------------------

    def _summary_comparison_chart(self, df: pd.DataFrame,
                                   target_column: str) -> None:
        """
        Side-by-side bar charts comparing all features on:
        • Cramér's V (association with target)
        • Shannon entropy (distribution spread)
        • Mean difference between classes
        """
        features = [c for c in df.columns if c != target_column
                    and _is_ordinal_feature(df[c])]
        if not features:
            return

        cramers, entropies, mean_diffs = [], [], []
        for f in features:
            s = _safe_numeric(df[f])
            cramers.append(_cramers_v(s.astype(str), df[target_column].astype(str)))
            entropies.append(_shannon_entropy(s))
            bk_m  = _safe_numeric(df[df[target_column] == "bankruptcy"][f]).mean()
            nb_m  = _safe_numeric(df[df[target_column] == "non-bankruptcy"][f]).mean()
            mean_diffs.append(abs(bk_m - nb_m))

        labels    = [f.replace("_", "\n") for f in features]
        x         = np.arange(len(features))
        width     = 0.26

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.patch.set_facecolor("#0F1923")
        fig.suptitle(
            "Cross-feature summary — all ordinal features",
            fontsize=14, color="#E0E8F4", y=1.02
        )

        for ax, vals, title, ylabel, color, thresh, thresh_label in [
            (axes[0], cramers, "Cramér's V\n(association with target)",
             "Cramér's V", "#C9A84C", 0.5, "strong"),
            (axes[1], entropies, "Shannon entropy\n(distribution spread)",
             "Entropy (bits)", PALETTE["neutral"], np.log2(3), "uniform"),
            (axes[2], mean_diffs, "Mean difference\n(bankruptcy vs non-bankruptcy)",
             "|Δ mean|", PALETTE["mid"], 0.3, "notable"),
        ]:
            bars = ax.bar(x, vals, color=color, width=0.55,
                          edgecolor="#1C3355", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9, color="#C0CCDD")
            ax.set_title(title, color="#E0E8F4", fontsize=11)
            ax.set_ylabel(ylabel, color="#C0CCDD")
            ax.axhline(thresh, color="#8899BB", linestyle="--",
                       linewidth=0.8, alpha=0.7,
                       label=f"{thresh_label} threshold ({thresh:.2f})")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center", va="bottom", color="#E0E8F4", fontsize=9
                )

        plt.tight_layout()
        plt.show()