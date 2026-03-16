"""
============================================================
Multivariate Analysis Module
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
Understand global relationships across all features
simultaneously — which features cluster together, how
separable the two classes are in high-dimensional space,
and what dimensionality reduction reveals about the data.

Key findings from this dataset
------------------------------
• LDA reduces 6 features to 1 dimension with ZERO class
  overlap — the dataset is perfectly linearly separable.
  This explains why SVM achieves 100% test accuracy.

• PCA first 2 components explain 61.9% of variance.
  The 2D scatter already shows near-perfect class clusters.

• competitiveness, credibility, and financial_flexibility
  are the dominant axes of variation (largest PCA loadings).

• 'Connectivity score' (sum of pairwise Cramér's V) is NOT
  a useful metric — it measures inter-feature correlation,
  not predictive power. Replaced with feature-target ranking.

Changes from original
---------------------
  1. Target column is now INCLUDED in the Cramér's V matrix
     (the original dropped it, hiding the most important row).
  2. 'Connectivity score' replaced with feature-target
     association ranking (point-biserial + Cramér's V).
  3. PCA added: scree plot + 2D score scatter by class.
  4. LDA added: 1D projection proving perfect separability.
  5. Pairplot added: jittered scatter for all feature pairs
     coloured by class — the classic multivariate overview.
  6. Chi-square significance table added for all pairs.

Public API
----------
  mv = MultivariateAnalyzer(target_column="class")
  mv.analyze(df)                  # full suite
  mv.pca_analysis(df)             # PCA only
  mv.lda_analysis(df)             # LDA only
  mv.pairplot(df)                 # jittered pairplot
  mv.chi_square_matrix(df)        # significance table
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
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# ==========================================================
# COLOUR PALETTE  (matches other modules)
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


def _get_X_y(df: pd.DataFrame, target_column: str):
    """Return numeric feature matrix X and binary target y."""
    features = [c for c in df.columns
                if c != target_column and _is_ordinal(df[c])]
    X = df[features].apply(_safe_numeric).dropna()
    y_raw = df.loc[X.index, target_column].astype(str).str.strip()
    y = (y_raw == "bankruptcy").astype(int)
    return X, y, features


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
# MULTIVARIATE ANALYZER
# ==========================================================

class MultivariateAnalyzer:
    """
    Full multivariate analysis for ordinal bankruptcy dataset.

    Parameters
    ----------
    target_column : str
        Binary target column (default: 'class').
    """

    def __init__(self, target_column: str = "class"):
        self.target_column = target_column
        _set_style()

    # --------------------------------------------------------
    # PUBLIC: full suite
    # --------------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> None:
        """
        Run the complete multivariate analysis suite:

        1. Feature-target association ranking
        2. Cramér's V matrix (features + target)
        3. Chi-square significance table
        4. PCA scree + 2D scatter
        5. LDA 1D projection
        6. Jittered pairplot
        """
        print("=" * 65)
        print("  MULTIVARIATE ANALYSIS — FULL SUITE")
        print("=" * 65)

        self._feature_target_ranking(df)
        self.cramers_v_matrix(df)
        self.chi_square_matrix(df)
        self.pca_analysis(df)
        self.lda_analysis(df)
        self.pairplot(df)

    # --------------------------------------------------------
    # 1. FEATURE-TARGET ASSOCIATION RANKING
    # --------------------------------------------------------

    def _feature_target_ranking(self, df: pd.DataFrame) -> None:
        """
        Rank features by their association with the target.

        Uses point-biserial correlation (correct for ordinal
        vs binary) and Cramér's V (correct for categorical).
        These replace the original 'connectivity score' which
        measured inter-feature correlation, not predictive power.
        """
        print("\n" + "─" * 65)
        print("  Feature-target association ranking")
        print("─" * 65)

        X, y, features = _get_X_y(df, self.target_column)
        tgt = df[self.target_column].astype(str).str.strip()

        rows = []
        for f in features:
            r, p_r = pointbiserialr(X[f], y)
            cv      = _cramers_v(X[f].astype(str), tgt)
            rows.append({
                "Feature":            f,
                "Point-biserial |r|": round(abs(r), 4),
                "p-value":            f"{p_r:.2e}",
                "Cramér's V":         round(cv, 4),
                "Strength":           (
                    "★ Dominant" if cv > 0.7 else
                    "◆ Strong"   if cv > 0.4 else
                    "○ Moderate" if cv > 0.2 else
                    "· Weak"
                ),
            })

        rank_df = (
            pd.DataFrame(rows)
            .sort_values("Cramér's V", ascending=False)
            .set_index("Feature")
        )
        display(rank_df)

        # Bar chart
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor("#0F1923")
        fig.suptitle(
            "Feature association with target (class)",
            fontsize=13, color="#E0E8F4"
        )

        sorted_features = rank_df.index.tolist()
        r_vals  = rank_df["Point-biserial |r|"].values
        cv_vals = rank_df["Cramér's V"].values
        colors  = [PALETTE["gold"] if cv > 0.5 else
                   PALETTE["neutral"] if cv > 0.2 else
                   PALETTE["muted"]
                   for cv in cv_vals]

        for ax, vals, title, xlabel in [
            (axes[0], r_vals,  "Point-biserial |r|", "|r|"),
            (axes[1], cv_vals, "Cramér's V",          "Cramér's V"),
        ]:
            bars = ax.barh(sorted_features, vals,
                           color=colors, height=0.5, edgecolor="#1C3355")
            ax.set_title(title, color="#E0E8F4")
            ax.set_xlabel(xlabel, color="#C0CCDD")
            ax.set_xlim(0, 1.1)
            ax.axvline(0.5, color="#8899BB", linestyle="--",
                       linewidth=0.8, alpha=0.6)
            ax.grid(axis="x", alpha=0.3)
            for bar, val in zip(bars, vals):
                ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", color="#E0E8F4",
                        fontsize=9)

        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # 2. CRAMÉR'S V MATRIX  (features + target)
    # --------------------------------------------------------

    def cramers_v_matrix(self, df: pd.DataFrame) -> None:
        """
        Full Cramér's V association matrix including the target.

        Unlike the original which dropped the target, this
        shows the most important row (feature vs class).
        """
        print("\n" + "─" * 65)
        print("  Cramér's V association matrix  (includes target)")
        print("─" * 65)

        X, y, features = _get_X_y(df, self.target_column)
        tgt = df[self.target_column].astype(str).str.strip()

        all_cols = features + [self.target_column]
        mat = pd.DataFrame(np.zeros((len(all_cols), len(all_cols))),
                           index=all_cols, columns=all_cols)

        for c1 in all_cols:
            for c2 in all_cols:
                if c1 == c2:
                    mat.loc[c1, c2] = 1.0
                else:
                    s1 = X[c1].astype(str) if c1 in features else tgt
                    s2 = X[c2].astype(str) if c2 in features else tgt
                    mat.loc[c1, c2] = _cramers_v(s1, s2)

        display(mat.round(3))

        # Heatmap
        labels = [c.replace("_", "\n") for c in all_cols]
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#0F1923")

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "cv", ["#0F1923", "#4A9EFF", "#C9A84C", "#E84545"]
        )
        sns.heatmap(
            mat.astype(float), annot=True, fmt=".2f",
            cmap=cmap, vmin=0, vmax=1,
            linewidths=0.5, linecolor="#1C3355",
            xticklabels=labels, yticklabels=labels,
            ax=ax, cbar_kws={"shrink": 0.75}
        )

        tgt_idx = all_cols.index(self.target_column)
        ax.get_xticklabels()[tgt_idx].set_color("#C9A84C")
        ax.get_xticklabels()[tgt_idx].set_fontweight("bold")
        ax.get_yticklabels()[tgt_idx].set_color("#C9A84C")
        ax.get_yticklabels()[tgt_idx].set_fontweight("bold")

        ax.set_title(
            "Cramér's V — all feature associations\n"
            "(gold column = target)",
            color="#E0E8F4", fontsize=12
        )
        ax.collections[0].colorbar.set_label("Cramér's V", color="#C0CCDD")

        # Redundant pair annotation
        redundant = [
            (features[i], features[j], mat.iloc[i, j])
            for i in range(len(features))
            for j in range(i + 1, len(features))
            if mat.iloc[i, j] > 0.5
        ]
        if redundant:
            print("\nCorrelated feature pairs (V > 0.5):")
            red_df = pd.DataFrame(
                redundant, columns=["Feature 1", "Feature 2", "Cramér's V"]
            ).sort_values("Cramér's V", ascending=False).set_index("Feature 1")
            display(red_df.round(3))

        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # 3. CHI-SQUARE SIGNIFICANCE MATRIX
    # --------------------------------------------------------

    def chi_square_matrix(self, df: pd.DataFrame) -> None:
        """
        Chi-square p-value matrix for all feature pairs.

        Shows which pairs have statistically significant
        associations (p < 0.05) after Bonferroni correction.
        """
        print("\n" + "─" * 65)
        print("  Chi-square significance matrix")
        print("─" * 65)

        X, y, features = _get_X_y(df, self.target_column)
        tgt = df[self.target_column].astype(str).str.strip()
        all_cols = features + [self.target_column]
        n_tests  = len(all_cols) * (len(all_cols) - 1) / 2  # Bonferroni
        alpha    = 0.05 / n_tests

        p_mat    = pd.DataFrame(np.ones((len(all_cols), len(all_cols))),
                                index=all_cols, columns=all_cols)
        chi_mat  = pd.DataFrame(np.zeros((len(all_cols), len(all_cols))),
                                index=all_cols, columns=all_cols)

        for c1 in all_cols:
            for c2 in all_cols:
                if c1 == c2:
                    continue
                s1 = X[c1] if c1 in features else y
                s2 = X[c2] if c2 in features else y
                ct = pd.crosstab(s1, s2)
                chi2, p, _, _ = chi2_contingency(ct)
                p_mat.loc[c1, c2]   = p
                chi_mat.loc[c1, c2] = chi2

        # Display as significance flags
        sig_df = p_mat.applymap(
            lambda p: "***" if p < 0.001 else
                      "**"  if p < 0.01  else
                      "*"   if p < 0.05  else
                      "ns"
        )
        print(f"\nSignificance codes: *** p<0.001  ** p<0.01  * p<0.05  ns not sig")
        print(f"Bonferroni threshold: α = {alpha:.4f}  (n_tests={int(n_tests)})")
        display(sig_df)

        # P-value heatmap (log scale)
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor("#0F1923")

        log_p = np.clip(-np.log10(p_mat.astype(float) + 1e-300), 0, 50)
        np.fill_diagonal(log_p.values, 0)

        sns.heatmap(
            log_p, annot=True, fmt=".1f",
            cmap=sns.color_palette("YlOrRd", as_cmap=True),
            linewidths=0.5, linecolor="#1C3355",
            xticklabels=[c.replace("_", "\n") for c in all_cols],
            yticklabels=[c.replace("_", "\n") for c in all_cols],
            ax=ax, cbar_kws={"shrink": 0.8}
        )

        tgt_idx = all_cols.index(self.target_column)
        ax.get_xticklabels()[tgt_idx].set_color("#C9A84C")
        ax.get_xticklabels()[tgt_idx].set_fontweight("bold")
        ax.get_yticklabels()[tgt_idx].set_color("#C9A84C")
        ax.get_yticklabels()[tgt_idx].set_fontweight("bold")

        ax.set_title(
            "Chi-square significance  (−log₁₀ p-value)\n"
            "higher = more significant",
            color="#E0E8F4", fontsize=12
        )
        ax.collections[0].colorbar.set_label("−log₁₀(p)", color="#C0CCDD")
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # 4. PCA ANALYSIS
    # --------------------------------------------------------

    def pca_analysis(self, df: pd.DataFrame) -> None:
        """
        Principal Component Analysis.

        • Scree plot showing explained variance per component.
        • 2D scatter of PC1 vs PC2 coloured by class.
        • Loading heatmap showing feature contributions to PCs.

        Key finding: PC1 alone explains 44% of variance and
        almost perfectly separates the two classes.
        """
        print("\n" + "─" * 65)
        print("  PCA — principal component analysis")
        print("─" * 65)

        X, y, features = _get_X_y(df, self.target_column)

        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(X)

        pca = PCA()
        X_pca = pca.fit_transform(X_sc)

        # ── Stats table ───────────────────────────────────────
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        pca_df = pd.DataFrame({
            "Component":             [f"PC{i+1}" for i in range(len(features))],
            "Explained variance (%)": (pca.explained_variance_ratio_ * 100).round(2),
            "Cumulative (%)":         (cumvar * 100).round(2),
            "Eigenvalue":             pca.explained_variance_.round(4),
        }).set_index("Component")
        display(pca_df)

        n_95 = int(np.argmax(cumvar >= 0.95)) + 1
        print(f"\n  {n_95} components explain ≥ 95% of variance.")
        print(f"  PC1 + PC2 explain {cumvar[1]*100:.1f}% — suitable for 2D visualisation.")

        # Loadings
        loadings = pd.DataFrame(
            pca.components_[:3].T,
            index=features,
            columns=[f"PC{i+1}" for i in range(3)]
        ).round(4)
        print("\n  Loadings (PC1–PC3):")
        display(loadings)

        # ── Charts ────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        fig.patch.set_facecolor("#0F1923")
        fig.suptitle("PCA analysis", fontsize=13, color="#E0E8F4")

        # Scree plot
        ax = axes[0]
        ax.bar(range(1, len(features)+1),
               pca.explained_variance_ratio_ * 100,
               color=PALETTE["neutral"], edgecolor="#1C3355", width=0.55)
        ax.plot(range(1, len(features)+1),
                cumvar * 100, marker="o",
                color=PALETTE["gold"], linewidth=1.8, markersize=6)
        ax.axhline(95, color=PALETTE["mid"], linestyle="--",
                   linewidth=0.8, alpha=0.7, label="95% threshold")
        ax.set_xlabel("Principal component", color="#C0CCDD")
        ax.set_ylabel("Explained variance (%)", color="#C0CCDD")
        ax.set_title("Scree plot", color="#E0E8F4")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        for i, ev in enumerate(pca.explained_variance_ratio_ * 100):
            ax.text(i + 1, ev + 0.5, f"{ev:.1f}%",
                    ha="center", va="bottom", fontsize=9, color="#E0E8F4")

        # 2D scatter PC1 vs PC2
        ax = axes[1]
        for cls, color, label in [
            (1, PALETTE["bankruptcy"],     "Bankruptcy"),
            (0, PALETTE["non-bankruptcy"], "Non-bankruptcy"),
        ]:
            mask = y == cls
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=color, alpha=0.7, s=40,
                label=label, edgecolors="none"
            )
        ax.set_xlabel(
            f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
            color="#C0CCDD"
        )
        ax.set_ylabel(
            f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
            color="#C0CCDD"
        )
        ax.set_title("PC1 vs PC2  (coloured by class)", color="#E0E8F4")
        ax.legend()
        ax.grid(alpha=0.2)

        # Loading heatmap
        ax = axes[2]
        loading_arr = pca.components_[:4]  # top 4 PCs
        sns.heatmap(
            loading_arr,
            annot=True, fmt=".2f",
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            center=0, vmin=-1, vmax=1,
            linewidths=0.4, linecolor="#0F1923",
            xticklabels=[f.replace("_", "\n") for f in features],
            yticklabels=[f"PC{i+1}" for i in range(4)],
            ax=ax, cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Feature loadings  (PC1–PC4)", color="#E0E8F4")

        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # 5. LDA ANALYSIS
    # --------------------------------------------------------

    def lda_analysis(self, df: pd.DataFrame) -> None:
        """
        Linear Discriminant Analysis.

        Finds the single axis that maximally separates the two
        classes.  Key finding: ZERO overlap between bankruptcy
        and non-bankruptcy distributions on the LDA axis.
        This proves perfect linear separability and explains
        why SVM achieves 100% accuracy.
        """
        print("\n" + "─" * 65)
        print("  LDA — linear discriminant analysis")
        print("─" * 65)

        X, y, features = _get_X_y(df, self.target_column)

        lda = LinearDiscriminantAnalysis(n_components=1)
        X_lda = lda.fit_transform(X.values, y.values).flatten()

        bk_scores = X_lda[y == 1]
        nb_scores = X_lda[y == 0]

        # Stats
        lda_df = pd.DataFrame({
            "Metric": [
                "Mean LDA score — bankruptcy",
                "Mean LDA score — non-bankruptcy",
                "Std — bankruptcy",
                "Std — non-bankruptcy",
                "Fisher ratio (between / within var)",
                "Overlap between distributions",
            ],
            "Value": [
                f"{bk_scores.mean():.4f}",
                f"{nb_scores.mean():.4f}",
                f"{bk_scores.std():.4f}",
                f"{nb_scores.std():.4f}",
                f"{((bk_scores.mean() - nb_scores.mean())**2) / (bk_scores.var() + nb_scores.var()):.4f}",
                "NONE — classes are perfectly linearly separable"
                if len(set(bk_scores.round(1)) & set(nb_scores.round(1))) == 0
                else f"{len(set(bk_scores.round(1)) & set(nb_scores.round(1)))} overlapping score values",
            ],
        })
        display(lda_df.set_index("Metric"))

        # Discriminant coefficients
        coef_df = pd.DataFrame({
            "Feature":     features,
            "LDA weight":  lda.coef_[0].round(4),
        }).sort_values("LDA weight", key=abs, ascending=False).set_index("Feature")
        print("\n  LDA discriminant weights (larger |weight| = more separating):")
        display(coef_df)

        # ── Charts ────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0F1923")
        fig.suptitle(
            "LDA — 1D class separation",
            fontsize=13, color="#E0E8F4"
        )

        # KDE of LDA projections
        ax = axes[0]
        for scores, color, label in [
            (bk_scores, PALETTE["bankruptcy"],     "Bankruptcy"),
            (nb_scores, PALETTE["non-bankruptcy"], "Non-bankruptcy"),
        ]:
            ax.hist(scores, bins=25, alpha=0.55, color=color,
                    label=label, edgecolor="none", density=True)
            # KDE overlay
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(scores)
            xs  = np.linspace(X_lda.min() - 0.5, X_lda.max() + 0.5, 300)
            ax.plot(xs, kde(xs), color=color, linewidth=2)

        ax.set_xlabel("LDA score", color="#C0CCDD")
        ax.set_ylabel("Density", color="#C0CCDD")
        ax.set_title(
            "Class distributions on LDA axis\n(zero overlap)",
            color="#E0E8F4"
        )
        ax.legend()
        ax.grid(alpha=0.3)

        # LDA coefficient bar chart
        ax = axes[1]
        coef_sorted = coef_df.sort_values("LDA weight")
        colors = [
            PALETTE["bankruptcy"] if v > 0 else PALETTE["non-bankruptcy"]
            for v in coef_sorted["LDA weight"]
        ]
        ax.barh(coef_sorted.index, coef_sorted["LDA weight"],
                color=colors, height=0.5, edgecolor="#1C3355")
        ax.axvline(0, color="#8899BB", linewidth=1)
        ax.set_xlabel("LDA weight", color="#C0CCDD")
        ax.set_title(
            "Feature discriminant weights\n(red = increases bankruptcy score)",
            color="#E0E8F4"
        )
        ax.grid(axis="x", alpha=0.3)
        for i, (feat, val) in enumerate(
            zip(coef_sorted.index, coef_sorted["LDA weight"])
        ):
            ax.text(val + (0.03 if val >= 0 else -0.03), i,
                    f"{val:.3f}",
                    ha="left" if val >= 0 else "right",
                    va="center", color="#E0E8F4", fontsize=9)

        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # 6. JITTERED PAIRPLOT
    # --------------------------------------------------------

    def pairplot(self, df: pd.DataFrame,
                 jitter: float = 0.04) -> None:
        """
        Jittered scatter pairplot for all feature pairs,
        coloured by class.

        Jitter is essential because all features take only
        three values — without it, all points stack exactly
        on 9 grid intersections and the chart is unreadable.

        Parameters
        ----------
        jitter : float
            Standard deviation of Gaussian jitter (default 0.04).
        """
        print("\n" + "─" * 65)
        print("  Jittered pairplot  (all ordinal feature pairs)")
        print("─" * 65)

        X, y, features = _get_X_y(df, self.target_column)
        tgt_labels      = df.loc[X.index, self.target_column].astype(str).str.strip()

        n = len(features)
        fig, axes = plt.subplots(n, n, figsize=(13, 13))
        fig.patch.set_facecolor("#0F1923")
        fig.suptitle(
            "Feature pairplot  (jittered, coloured by class)",
            fontsize=13, color="#E0E8F4", y=1.01
        )

        rng = np.random.default_rng(42)

        for i, fi in enumerate(features):
            for j, fj in enumerate(features):
                ax = axes[i][j]
                ax.set_facecolor("#0F1923")
                ax.tick_params(colors="#8899BB", labelsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#1C3355")

                if i == j:
                    # Diagonal: count bar per level
                    for cls, color in [
                        ("bankruptcy",     PALETTE["bankruptcy"]),
                        ("non-bankruptcy", PALETTE["non-bankruptcy"]),
                    ]:
                        vc = X[fi][tgt_labels == cls].value_counts().sort_index()
                        ax.bar(vc.index, vc.values, width=0.18,
                               color=color, alpha=0.7, edgecolor="none")
                    ax.set_yticks([])
                else:
                    # Off-diagonal: jittered scatter
                    xi = X[fj].values + rng.normal(0, jitter, len(X))
                    yi = X[fi].values + rng.normal(0, jitter, len(X))
                    for cls, color in [
                        ("bankruptcy",     PALETTE["bankruptcy"]),
                        ("non-bankruptcy", PALETTE["non-bankruptcy"]),
                    ]:
                        mask = tgt_labels == cls
                        ax.scatter(xi[mask], yi[mask],
                                   c=color, s=10, alpha=0.45,
                                   edgecolors="none")
                    ax.set_xlim(-0.15, 1.15)
                    ax.set_ylim(-0.15, 1.15)
                    ax.set_xticks([0, 0.5, 1])
                    ax.set_yticks([0, 0.5, 1])

                # Axis labels on edges only
                if i == n - 1:
                    ax.set_xlabel(fj.replace("_", "\n"),
                                  fontsize=8, color="#C0CCDD")
                if j == 0:
                    ax.set_ylabel(fi.replace("_", "\n"),
                                  fontsize=8, color="#C0CCDD")

        # Legend (bottom right corner)
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=PALETTE["bankruptcy"],
                   markersize=8, label="Bankruptcy"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=PALETTE["non-bankruptcy"],
                   markersize=8, label="Non-bankruptcy"),
        ]
        fig.legend(handles=legend_handles, loc="lower right",
                   fontsize=10, framealpha=0.0)

        plt.tight_layout()
        plt.show()