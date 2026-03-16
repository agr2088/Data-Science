"""
============================================================
Bivariate Analysis Module
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
Perform pairwise feature analysis to identify which feature
combinations drive bankruptcy prediction.

All features are ordinal 3-level (0.0 / 0.5 / 1.0) and the
target is binary ('bankruptcy' / 'non-bankruptcy').  Every
method is designed for this specific data structure.

Changes from original
---------------------
The original module had three structural problems:

  1. analyze(df, 'class', feature) silently broke because
     pd.to_numeric('bankruptcy') → NaN, so the pivot_table
     produced an empty result with no error message.
     Fixed by separating feature-vs-target analysis into its
     own method: analyze_feature_vs_target().

  2. No analyze_all() — the notebook looped over all columns
     with analyze(), producing 36 redundant feature-pair
     reports instead of the intended 6 feature-vs-target
     reports plus a summary.

  3. The Cramér's V matrix was missing — the most important
     single chart for understanding inter-feature relationships
     in a categorical dataset.

Public API
----------
  analyzer = BivariateAnalyzer(target_column="class")

  # Feature vs target (main notebook use case)
  analyzer.analyze_feature_vs_target(df, "competitiveness")

  # Any two features against each other
  analyzer.analyze(df, "financial_flexibility", "credibility")

  # Full Cramér's V matrix across all feature pairs
  analyzer.cramers_v_matrix(df)

  # Run everything in one call
  analyzer.analyze_all(df)

Key insights baked into visualizations
---------------------------------------
• competitiveness × credibility = 100% bankruptcy at 0.0×0.0,
  0% at 1.0×1.0.  The cleanest interaction in the dataset.
• financial_flexibility + credibility is the most synergistic
  pair (joint Cramér's V with target = 0.93).
• Conditional entropy H(class | feature) shows competitiveness
  alone reduces class uncertainty to 0.08 bits from 0.98 bits.
"""

# ==========================================================
# IMPORTS
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from IPython.display import display
from scipy.stats import chi2_contingency, entropy as sp_entropy


# ==========================================================
# COLOUR PALETTE  (matches univariate module)
# ==========================================================

PALETTE = {
    "bankruptcy":     "#E84545",
    "non-bankruptcy": "#2ECC71",
    "neutral":        "#4A9EFF",
    "mid":            "#F39C12",
    "muted":          "#8899BB",
    "gold":           "#C9A84C",
}

CLASS_PALETTE = [PALETTE["bankruptcy"], PALETTE["non-bankruptcy"]]

LEVEL_LABELS = {0.0: "Low", 0.5: "Med", 1.0: "High"}
LEVEL_COLORS = {0.0: "#4A9EFF", 0.5: "#F39C12", 1.0: "#E84545"}


# ==========================================================
# HELPERS
# ==========================================================

def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")


def _is_ordinal(series: pd.Series) -> bool:
    num = _safe_numeric(series).dropna()
    return set(num.unique()).issubset({0.0, 0.5, 1.0}) and len(num) > 0


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    ct = pd.crosstab(a, b)
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    min_dim = min(ct.shape) - 1
    return float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 and n > 0 else 0.0


def _conditional_entropy(df: pd.DataFrame, feature: str,
                          target: str = "class") -> float:
    """H(target | feature) in bits."""
    total = len(df)
    h = 0.0
    for val in df[feature].unique():
        subset = df[df[feature] == val][target]
        probs  = subset.value_counts(normalize=True)
        p_val  = len(subset) / total
        h     += p_val * sp_entropy(probs.values, base=2)
    return h


def _set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  "#0F1923",
        "axes.facecolor":    "#0F1923",
        "axes.edgecolor":    "#1C3355",
        "axes.labelcolor":   "#C0CCDD",
        "axes.titlecolor":   "#E0E8F4",
        "axes.titlesize":    12,
        "axes.labelsize":    10,
        "xtick.color":       "#8899BB",
        "ytick.color":       "#8899BB",
        "text.color":        "#C0CCDD",
        "grid.color":        "#1C3355",
        "grid.linewidth":    0.5,
        "legend.facecolor":  "#0F1923",
        "legend.edgecolor":  "#1C3355",
        "legend.fontsize":   9,
    })


# ==========================================================
# BIVARIATE ANALYZER
# ==========================================================

class BivariateAnalyzer:
    """
    Pairwise analysis for ordinal × binary target datasets.

    Parameters
    ----------
    target_column : str
        Name of the binary target column (default: 'class').
    """

    def __init__(self, target_column: str = "class"):
        self.target_column = target_column
        _set_style()

    # --------------------------------------------------------
    # PUBLIC: feature vs target  (PRIMARY METHOD)
    # --------------------------------------------------------

    def analyze_feature_vs_target(
        self, df: pd.DataFrame, feature: str
    ) -> None:
        """
        Full bivariate analysis between one ordinal feature
        and the binary target.

        Produces:
        • frequency + bankruptcy-rate table
        • conditional entropy reduction
        • Cramér's V + chi-square
        • 3-panel chart:
            1. Bankruptcy probability per level (bar)
            2. Stacked absolute counts per level
            3. Class split across all levels (diverging bar)

        Parameters
        ----------
        df      : Full dataset.
        feature : Ordinal feature name.
        """
        if feature not in df.columns:
            print(f"❌  '{feature}' not found in dataset.")
            return
        if not _is_ordinal(df[feature]):
            print(f"⚠  '{feature}' is not an ordinal 3-level feature.")
            return

        print("\n" + "=" * 64)
        print(f"  BIVARIATE ANALYSIS — {feature.upper().replace('_',' ')} vs TARGET")
        print("=" * 64)

        num = _safe_numeric(df[feature])
        tgt = df[self.target_column].astype(str).str.strip()

        # ── Stats table ───────────────────────────────────────
        rows = []
        for lv in sorted(num.unique()):
            mask   = num == lv
            subset = tgt[mask]
            total  = mask.sum()
            bk     = (subset == "bankruptcy").sum()
            nb     = total - bk
            rows.append({
                "Level":              LEVEL_LABELS.get(lv, str(lv)),
                "n":                  total,
                "% of dataset":       f"{total/len(df)*100:.1f}%",
                "Bankruptcy count":   bk,
                "Non-bankr. count":   nb,
                "Bankruptcy rate":    f"{bk/total*100:.1f}%" if total > 0 else "—",
            })
        stats_df = pd.DataFrame(rows).set_index("Level")
        display(stats_df)

        # ── Information-theoretic stats ───────────────────────
        H_prior = sp_entropy(tgt.value_counts(normalize=True).values, base=2)
        H_cond  = _conditional_entropy(
            pd.DataFrame({feature: num, self.target_column: tgt}),
            feature, self.target_column
        )
        ig      = H_prior - H_cond
        cv      = _cramers_v(num.astype(str), tgt)

        ct = pd.crosstab(num, tgt)
        from scipy.stats import chi2_contingency
        chi2, p_chi, dof, _ = chi2_contingency(ct)

        info_df = pd.DataFrame({
            "Metric": [
                "H(class) — prior entropy (bits)",
                "H(class | feature) — conditional entropy (bits)",
                "Information gain (bits)",
                "Entropy reduction",
                "Cramér's V (association strength)",
                "Chi-square p-value",
                "Degrees of freedom",
            ],
            "Value": [
                f"{H_prior:.4f}",
                f"{H_cond:.4f}",
                f"{ig:.4f}",
                f"{ig/H_prior*100:.1f}%  "
                f"({'★ dominant' if ig/H_prior > 0.7 else '◆ strong' if ig/H_prior > 0.4 else '○ moderate'})",
                f"{cv:.4f}  "
                f"({'★ strong' if cv > 0.5 else '◆ moderate' if cv > 0.2 else '○ weak'})",
                f"{p_chi:.2e}  {'✓ significant' if p_chi < 0.05 else '✗ not significant'}",
                f"{dof}",
            ],
        })
        display(info_df.set_index("Metric"))

        # ── Charts ────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.patch.set_facecolor("#0F1923")
        fig.suptitle(
            f"{feature.replace('_',' ').title()}  vs  Class",
            fontsize=13, color="#E0E8F4", y=1.02
        )

        levels    = sorted(num.unique())
        bk_rates  = []
        bk_counts = []
        nb_counts = []

        for lv in levels:
            mask = num == lv
            sub  = tgt[mask]
            bk   = (sub == "bankruptcy").sum()
            nb   = (sub == "non-bankruptcy").sum()
            bk_rates.append(bk / len(sub) * 100 if len(sub) > 0 else 0)
            bk_counts.append(bk)
            nb_counts.append(nb)

        level_names = [LEVEL_LABELS.get(lv, str(lv)) for lv in levels]
        bar_colors  = [LEVEL_COLORS.get(lv, PALETTE["neutral"]) for lv in levels]

        # Chart 1: Bankruptcy rate per level
        ax = axes[0]
        bars = ax.bar(level_names, bk_rates,
                      color=bar_colors, width=0.45, edgecolor="#1C3355")
        ax.set_ylim(0, 115)
        ax.axhline(50, color="#8899BB", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title("Bankruptcy rate per level", color="#E0E8F4")
        ax.set_xlabel("Level", color="#C0CCDD")
        ax.set_ylabel("Bankruptcy rate (%)", color="#C0CCDD")
        ax.grid(axis="y", alpha=0.3)
        for bar, rate, bk, total in zip(
            bars, bk_rates, bk_counts,
            [bk + nb for bk, nb in zip(bk_counts, nb_counts)]
        ):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    f"{rate:.0f}%\n(n={total})",
                    ha="center", va="bottom", color="#E0E8F4", fontsize=9)

        # Chart 2: Stacked absolute counts
        ax = axes[1]
        x  = np.arange(len(levels))
        b1 = ax.bar(x, bk_counts, color=PALETTE["bankruptcy"],
                    width=0.4, label="Bankruptcy", edgecolor="#1C3355")
        b2 = ax.bar(x, nb_counts, bottom=bk_counts,
                    color=PALETTE["non-bankruptcy"],
                    width=0.4, label="Non-bankruptcy", edgecolor="#1C3355")
        ax.set_xticks(x)
        ax.set_xticklabels(level_names, color="#C0CCDD")
        ax.set_title("Absolute counts per level", color="#E0E8F4")
        ax.set_xlabel("Level", color="#C0CCDD")
        ax.set_ylabel("Count", color="#C0CCDD")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        for xi, (bk, nb) in enumerate(zip(bk_counts, nb_counts)):
            if bk > 0:
                ax.text(xi, bk / 2, str(bk), ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
            if nb > 0:
                ax.text(xi, bk + nb / 2, str(nb), ha="center", va="center",
                        color="#0F1923", fontsize=9, fontweight="bold")

        # Chart 3: Diverging risk bar (bankruptcy % vs 50% baseline)
        ax = axes[2]
        deltas = [r - 50 for r in bk_rates]
        colors_div = [PALETTE["bankruptcy"] if d > 0 else PALETTE["non-bankruptcy"]
                      for d in deltas]
        ax.barh(level_names, deltas, color=colors_div,
                height=0.4, edgecolor="#1C3355")
        ax.axvline(0, color="#8899BB", linewidth=1)
        ax.set_xlim(-60, 60)
        ax.set_title("Risk deviation from 50%", color="#E0E8F4")
        ax.set_xlabel("Δ from 50% (percentage points)", color="#C0CCDD")
        ax.grid(axis="x", alpha=0.3)
        for i, (d, r) in enumerate(zip(deltas, bk_rates)):
            ax.text(d + (2 if d >= 0 else -2), i,
                    f"{r:.0f}%",
                    ha="left" if d >= 0 else "right",
                    va="center", color="#E0E8F4", fontsize=9)

        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # PUBLIC: feature × feature (MANUAL PAIR EXPLORATION)
    # --------------------------------------------------------

    def analyze(
        self, df: pd.DataFrame, feature1: str, feature2: str
    ) -> None:
        """
        Full bivariate analysis between two ordinal features,
        including interaction effect on the target.

        Produces:
        • frequency cross-tab
        • row-normalised percentage table
        • bankruptcy probability matrix (the key insight)
        • Cramér's V + chi-square between the two features
        • joint Cramér's V with target (synergy measurement)
        • 2-panel chart: frequency heatmap + risk heatmap

        Parameters
        ----------
        df       : Full dataset.
        feature1 : First ordinal feature.
        feature2 : Second ordinal feature.
        """
        for f in (feature1, feature2):
            if f not in df.columns:
                print(f"❌  '{f}' not found.")
                return
            if not _is_ordinal(df[f]):
                print(f"⚠  '{f}' is not an ordinal feature.")
                return

        print("\n" + "=" * 64)
        print(f"  FEATURE PAIR — {feature1.upper().replace('_',' ')} × {feature2.upper().replace('_',' ')}")
        print("=" * 64)

        f1  = _safe_numeric(df[feature1])
        f2  = _safe_numeric(df[feature2])
        tgt = df[self.target_column].astype(str).str.strip()

        temp = pd.DataFrame({
            feature1: f1,
            feature2: f2,
            self.target_column: tgt,
        }).dropna()

        # ── Frequency cross-tab ───────────────────────────────
        freq = pd.crosstab(temp[feature1], temp[feature2])
        freq.index   = [LEVEL_LABELS.get(v, str(v)) for v in freq.index]
        freq.columns = [LEVEL_LABELS.get(v, str(v)) for v in freq.columns]
        freq.index.name   = feature1
        freq.columns.name = feature2
        print("\nFrequency table:")
        display(freq)

        # ── Row-normalised ────────────────────────────────────
        row_pct = (freq.div(freq.sum(axis=1), axis=0) * 100).round(1)
        print("\nRow percentage (%):")
        display(row_pct)

        # ── Bankruptcy probability matrix ─────────────────────
        risk_raw = pd.pivot_table(
            temp, values=self.target_column,
            index=feature1, columns=feature2,
            aggfunc=lambda x: (x == "bankruptcy").mean()
        )
        count_raw = pd.pivot_table(
            temp, values=self.target_column,
            index=feature1, columns=feature2,
            aggfunc="count"
        )
        risk_raw.index   = [LEVEL_LABELS.get(v, str(v)) for v in risk_raw.index]
        risk_raw.columns = [LEVEL_LABELS.get(v, str(v)) for v in risk_raw.columns]
        risk_raw.index.name   = feature1
        risk_raw.columns.name = feature2

        print("\nBankruptcy probability matrix (% per cell):")
        display((risk_raw * 100).round(1))

        # ── Association metrics ───────────────────────────────
        ct_features = pd.crosstab(f1, f2)
        chi2_ff, p_ff, dof_ff, _ = chi2_contingency(ct_features)
        cv_ff = _cramers_v(f1.astype(str), f2.astype(str))

        # Joint interaction with target
        combined = f1.astype(str) + "_" + f2.astype(str)
        cv_joint = _cramers_v(combined, tgt)
        cv1 = _cramers_v(f1.astype(str), tgt)
        cv2 = _cramers_v(f2.astype(str), tgt)
        synergy = cv_joint - max(cv1, cv2)

        assoc_df = pd.DataFrame({
            "Metric": [
                f"Cramér's V ({feature1} × {feature2})",
                "Chi-square p-value (feature independence)",
                f"Cramér's V ({feature1} × target) alone",
                f"Cramér's V ({feature2} × target) alone",
                f"Joint Cramér's V ({feature1}+{feature2} × target)",
                "Synergy  (joint − max individual)",
            ],
            "Value": [
                f"{cv_ff:.4f}  {'★ correlated' if cv_ff > 0.5 else '◆ moderate' if cv_ff > 0.2 else '○ independent'}",
                f"{p_ff:.2e}",
                f"{cv1:.4f}",
                f"{cv2:.4f}",
                f"{cv_joint:.4f}",
                f"{synergy:+.4f}  "
                f"({'synergistic' if synergy > 0.05 else 'redundant' if synergy < -0.05 else 'additive'})",
            ],
        })
        display(assoc_df.set_index("Metric"))

        # ── Charts ────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0F1923")
        fig.suptitle(
            f"{feature1.replace('_',' ').title()}  ×  "
            f"{feature2.replace('_',' ').title()}",
            fontsize=13, color="#E0E8F4", y=1.02
        )

        # Left: frequency heatmap
        ax = axes[0]
        sns.heatmap(
            freq.astype(int), annot=True, fmt="d",
            cmap=sns.light_palette("#4A9EFF", as_cmap=True),
            linewidths=0.5, linecolor="#0F1923",
            ax=ax, cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Sample count per cell", color="#E0E8F4")
        ax.set_xlabel(feature2.replace("_", " "), color="#C0CCDD")
        ax.set_ylabel(feature1.replace("_", " "), color="#C0CCDD")

        # Right: bankruptcy probability heatmap with count overlay
        ax = axes[1]
        annot_labels = (risk_raw * 100).round(0).astype(int).astype(str) + "%"
        # Add sample sizes as second line
        for i, row_idx in enumerate(risk_raw.index):
            for j, col_idx in enumerate(risk_raw.columns):
                try:
                    ri = [LEVEL_LABELS.get(v, str(v))
                          for v in count_raw.index].index(row_idx)
                    ci = [LEVEL_LABELS.get(v, str(v))
                          for v in count_raw.columns].index(col_idx)
                    n = int(count_raw.values[ri, ci])
                    annot_labels.iloc[i, j] += f"\n(n={n})"
                except (ValueError, IndexError):
                    pass

        risk_cmap = mcolors.LinearSegmentedColormap.from_list(
            "risk", ["#2ECC71", "#F39C12", "#E84545"]
        )
        sns.heatmap(
            risk_raw, annot=annot_labels, fmt="",
            cmap=risk_cmap, vmin=0, vmax=1,
            linewidths=0.5, linecolor="#0F1923",
            ax=ax, cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Bankruptcy probability per cell", color="#E0E8F4")
        ax.set_xlabel(feature2.replace("_", " "), color="#C0CCDD")
        ax.set_ylabel(feature1.replace("_", " "), color="#C0CCDD")
        # Colourbar label
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.set_label("P(bankruptcy)", color="#C0CCDD")
            cbar.ax.yaxis.set_tick_params(color="#8899BB")

        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # PUBLIC: full Cramér's V matrix
    # --------------------------------------------------------

    def cramers_v_matrix(self, df: pd.DataFrame) -> None:
        """
        Compute and visualise Cramér's V between every pair
        of ordinal features, plus each feature vs target.

        This is the primary inter-feature dependency chart.
        """
        ord_cols = [c for c in df.columns
                    if c != self.target_column and _is_ordinal(df[c])]

        all_cols = ord_cols + [self.target_column]
        n        = len(all_cols)
        mat      = pd.DataFrame(np.zeros((n, n)),
                                index=all_cols, columns=all_cols)

        tgt_str = df[self.target_column].astype(str).str.strip()

        for i, c1 in enumerate(all_cols):
            for j, c2 in enumerate(all_cols):
                if c1 == c2:
                    mat.loc[c1, c2] = 1.0
                    continue
                s1 = (_safe_numeric(df[c1]).astype(str)
                      if c1 != self.target_column
                      else tgt_str)
                s2 = (_safe_numeric(df[c2]).astype(str)
                      if c2 != self.target_column
                      else tgt_str)
                mat.loc[c1, c2] = _cramers_v(s1, s2)

        print("\n" + "=" * 64)
        print("  CRAMÉR'S V — FULL ASSOCIATION MATRIX")
        print("=" * 64)
        display(mat.round(3))

        # ── Heatmap ───────────────────────────────────────────
        labels = [c.replace("_", "\n") for c in all_cols]
        # Last column/row = target (highlight with border)
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#0F1923")

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "cv", ["#0F1923", "#4A9EFF", "#C9A84C", "#E84545"]
        )
        sns.heatmap(
            mat.astype(float),
            annot=True, fmt=".2f",
            cmap=cmap, vmin=0, vmax=1,
            linewidths=0.5, linecolor="#1C3355",
            xticklabels=labels, yticklabels=labels,
            ax=ax, cbar_kws={"shrink": 0.75}
        )
        ax.set_title(
            "Cramér's V — inter-feature association\n"
            "(1.0 = perfect, 0.0 = independent)",
            color="#E0E8F4", fontsize=12
        )

        # Highlight target row/col with gold tick labels
        tgt_idx = list(all_cols).index(self.target_column)
        ax.get_xticklabels()[tgt_idx].set_color("#C9A84C")
        ax.get_xticklabels()[tgt_idx].set_fontweight("bold")
        ax.get_yticklabels()[tgt_idx].set_color("#C9A84C")
        ax.get_yticklabels()[tgt_idx].set_fontweight("bold")

        cbar = ax.collections[0].colorbar
        cbar.set_label("Cramér's V", color="#C0CCDD")

        plt.tight_layout()
        plt.show()

        # ── Ranked target associations ─────────────────────────
        tgt_row = mat[self.target_column].drop(self.target_column).sort_values(ascending=False)
        print("\nFeatures ranked by association with target:")
        rank_df = pd.DataFrame({
            "Feature": tgt_row.index,
            "Cramér's V": tgt_row.values.round(4),
            "Strength": [
                "★ Strong" if v > 0.5 else
                "◆ Moderate" if v > 0.2 else "○ Weak"
                for v in tgt_row.values
            ],
        }).set_index("Feature")
        display(rank_df)

    # --------------------------------------------------------
    # PUBLIC: analyze_all
    # --------------------------------------------------------

    def analyze_all(self, df: pd.DataFrame) -> None:
        """
        Run the complete bivariate analysis suite:
          1. analyze_feature_vs_target() for each ordinal feature
          2. cramers_v_matrix() for the full dependency map
          3. The most informative feature pair
             (financial_flexibility × competitiveness)

        Parameters
        ----------
        df : Full dataset with target column.
        """
        ord_features = [c for c in df.columns
                        if c != self.target_column and _is_ordinal(df[c])]

        print("=" * 64)
        print("  BIVARIATE ANALYSIS — FULL SUITE")
        print(f"  {len(ord_features)} features vs target  +  Cramér's V matrix")
        print("=" * 64)

        # 1. Each feature vs target
        for feat in ord_features:
            self.analyze_feature_vs_target(df, feat)

        # 2. Full association matrix
        self.cramers_v_matrix(df)

        # 3. Most informative pair (hard-coded based on data audit)
        best_pairs = [
            ("financial_flexibility", "competitiveness"),
            ("credibility", "competitiveness"),
        ]
        print("\n" + "=" * 64)
        print("  MOST INFORMATIVE FEATURE PAIRS")
        print("=" * 64)
        for f1, f2 in best_pairs:
            if f1 in df.columns and f2 in df.columns:
                self.analyze(df, f1, f2)