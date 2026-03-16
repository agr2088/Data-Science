"""
============================================================
Data Inspection Module
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
First-pass inspection of the raw bankruptcy dataset before
any EDA or modelling begins.  Designed to run top-to-bottom
in a notebook and answer the question:

    "What exactly is in this file and is it usable?"

Design pattern
--------------
Strategy pattern — swap inspection strategies without
changing the orchestrating DataInspector context class.
This makes it easy to add new inspection types later.

Changes from original
---------------------
  1. SummaryStatisticsInspectionStrategy used
     describe(include=['object']) — this shows stats for
     the target string column only and SKIPS all 6 numeric
     features.  Fixed to run describe() on numeric columns
     and include a value_counts table (essential for
     ordinal 3-level features).

  2. BasicStructureInspectionStrategy only reported the
     duplicate count.  Added:
       • duplicate % of total
       • class distribution of duplicate rows vs unique rows
       • which rows appear most often

  3. No data quality checklist — added DataQualityReport
     which gives a single pass/fail scorecard with every
     check a reviewer needs to sign off on before modelling.

  4. No ordinal value distribution — added
     OrdinalDistributionStrategy which produces a
     value_counts table and bar chart for every feature,
     confirming every column really is 0 / 0.5 / 1.

  5. DataStructureFixer is kept for notebook convenience
     (the raw Excel is semicolon-delimited in one column).
     It replicates logic from data_ingestion.py intentionally
     so the notebook is self-contained.
"""

# ==========================================================
# IMPORTS
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABC, abstractmethod
from IPython.display import display


# ==========================================================
# COLOUR PALETTE  (matches all analysis modules)
# ==========================================================

PALETTE = {
    "bankruptcy":     "#E84545",
    "non-bankruptcy": "#2ECC71",
    "neutral":        "#4A9EFF",
    "mid":            "#F39C12",
    "muted":          "#8899BB",
    "gold":           "#C9A84C",
}


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
# DATA STRUCTURE FIXER
# ==========================================================

class DataStructureFixer:
    """
    Fix the malformed Excel file where all columns are
    packed into one semicolon-delimited string per row.

    Note: This replicates the fix in data_ingestion.py
    so the notebook is fully self-contained without
    depending on the pipeline components.
    """

    EXPECTED_COLUMNS = [
        "industrial_risk",
        "management_risk",
        "financial_flexibility",
        "credibility",
        "competitiveness",
        "operating_risk",
        "class",
    ]

    @staticmethod
    def fix(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and repair a single-column semicolon dataset.

        Also standardises column names (lowercase, stripped).

        Returns
        -------
        pd.DataFrame
            Clean, properly structured DataFrame.
        """
        # ── Repair single-column format ───────────────────────
        if df.shape[1] == 1:
            print("⚠  Single-column format detected — splitting on ';'...")
            df = df.iloc[:, 0].astype(str).str.split(";", expand=True)

            if df.shape[1] == len(DataStructureFixer.EXPECTED_COLUMNS):
                df.columns = DataStructureFixer.EXPECTED_COLUMNS
                print(f"  Columns assigned: {DataStructureFixer.EXPECTED_COLUMNS}")
            else:
                raise ValueError(
                    f"Expected {len(DataStructureFixer.EXPECTED_COLUMNS)} columns "
                    f"after split, got {df.shape[1]}."
                )

        # ── Standardise column names ──────────────────────────
        df.columns = df.columns.astype(str).str.strip().str.lower()

        # ── Convert numeric feature columns ──────────────────
        # This also NaN-ifies the header row (if present) because
        # pd.to_numeric('industrial_risk') = NaN.
        numeric_cols = [c for c in df.columns if c != "class"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip(), errors="coerce"
            )

        # ── Drop header row if it survived as all-NaN ─────────
        # When pandas reads the raw Excel with header=None the first
        # row contains the column name strings (e.g. 'industrial_risk').
        # After to_numeric() that row becomes all NaN for every feature
        # column but keeps the string 'class' in the target column.
        # Leaving it in breaks pointbiserialr (shape mismatch after
        # dropna) and mutual_info_classif (NaN not accepted).
        before = len(df)
        df = df.dropna(subset=numeric_cols, how="all").reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            print(f"  Dropped {dropped} header/empty row(s).")

        # ── Strip target column ───────────────────────────────
        if "class" in df.columns:
            df["class"] = df["class"].astype(str).str.strip()

        print(f"  Final shape: {df.shape}")
        return df


# ==========================================================
# STRATEGY INTERFACE
# ==========================================================

class DataInspectionStrategy(ABC):
    """Abstract base for all inspection strategies."""

    @abstractmethod
    def inspect(self, df: pd.DataFrame) -> None:
        pass


# ==========================================================
# STRATEGY 1: BASIC STRUCTURE
# ==========================================================

class BasicStructureInspectionStrategy(DataInspectionStrategy):
    """
    Inspect fundamental dataset structure.

    Reports:
    • shape (rows × columns)
    • column names and dtypes
    • first 5 rows
    • duplicate analysis with class breakdown
    • memory usage
    """

    def inspect(self, df: pd.DataFrame) -> None:
        _set_style()

        print("=" * 64)
        print("  DATASET STRUCTURE")
        print("=" * 64)

        # Shape
        print(f"\n  Rows    : {df.shape[0]:,}")
        print(f"  Columns : {df.shape[1]}")

        # Columns summary
        col_df = pd.DataFrame({
            "Column":   df.columns,
            "Dtype":    [str(df[c].dtype) for c in df.columns],
            "Non-null": [df[c].notna().sum() for c in df.columns],
            "Unique":   [df[c].nunique() for c in df.columns],
        }).set_index("Column")
        display(col_df)

        # First rows
        print("\n  First 5 rows:")
        display(df.head())

        # ── Duplicate analysis ────────────────────────────────
        total      = len(df)
        dup_mask   = df.duplicated(keep=False)   # mark ALL copies
        dup_count  = df.duplicated().sum()        # count extras only
        dup_pct    = dup_count / total * 100

        print(f"\n  Duplicate rows   : {dup_count:,}  ({dup_pct:.1f}% of dataset)")
        print(f"  Unique rows      : {total - dup_count:,}")

        if dup_count > 0 and "class" in df.columns:
            # Class breakdown of duplicated rows
            dup_class = df[dup_mask]["class"].value_counts()
            all_class = df["class"].value_counts()
            dup_pct_df = pd.DataFrame({
                "Duplicated rows":  dup_class,
                "All rows":         all_class,
                "Dup % of class":   (dup_class / all_class * 100).round(1),
            })
            print("\n  Duplicate class breakdown:")
            display(dup_pct_df)

            # Most duplicated rows
            dup_groups = (
                df[dup_mask]
                .groupby(df.columns.tolist())
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(5)
            )
            print("\n  Most repeated rows (top 5):")
            display(dup_groups)

        # Memory
        mem_kb = df.memory_usage(deep=True).sum() / 1024
        print(f"\n  Memory usage: {mem_kb:.1f} KB")


# ==========================================================
# STRATEGY 2: DATA TYPES
# ==========================================================

class DataTypesInspectionStrategy(DataInspectionStrategy):
    """
    Inspect column data types and null counts.

    Also checks for type consistency — e.g. a 'numeric'
    column that contains strings after loading.
    """

    def inspect(self, df: pd.DataFrame) -> None:
        print("=" * 64)
        print("  DATA TYPES AND NULL COUNTS")
        print("=" * 64)

        info_rows = []
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct   = null_count / len(df) * 100
            info_rows.append({
                "Column":       col,
                "Dtype":        str(df[col].dtype),
                "Null count":   null_count,
                "Null %":       f"{null_pct:.1f}%",
                "Sample value": str(df[col].dropna().iloc[0]) if df[col].notna().any() else "—",
            })

        display(pd.DataFrame(info_rows).set_index("Column"))

        total_nulls = df.isnull().sum().sum()
        if total_nulls == 0:
            print("\n  ✓  Zero missing values across all columns.")
        else:
            print(f"\n  ⚠  {total_nulls} total missing values detected.")


# ==========================================================
# STRATEGY 3: SUMMARY STATISTICS  (FIXED)
# ==========================================================

class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    """
    Generate descriptive statistics for all columns.

    Bug fixed: original used describe(include=['object'])
    which only showed stats for the string target column
    and completely skipped all 6 numeric feature columns.

    Now produces:
    • Numeric describe() for feature columns
    • Value counts for every column (essential for ordinal)
    • Class distribution chart
    """

    def inspect(self, df: pd.DataFrame) -> None:
        _set_style()

        print("=" * 64)
        print("  SUMMARY STATISTICS")
        print("=" * 64)

        # Numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            print("\n  Numeric feature statistics:")
            display(df[numeric_cols].describe().round(4))

        # String / categorical columns
        obj_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if obj_cols:
            print("\n  Categorical column statistics:")
            display(df[obj_cols].describe())

        # Value counts per column
        print("\n  Value distribution per column:")
        for col in df.columns:
            vc  = df[col].value_counts().sort_index()
            pct = (vc / len(df) * 100).round(1)
            vc_df = pd.DataFrame({
                "Count":   vc.values,
                "Pct (%)": pct.values,
            }, index=vc.index)
            vc_df.index.name = col
            display(vc_df)

        # Class distribution chart (if target present)
        if "class" in df.columns:
            vc  = df["class"].value_counts()
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            fig.patch.set_facecolor("#0F1923")
            fig.suptitle("Target class distribution",
                         fontsize=12, color="#E0E8F4")

            # Count bar
            colors = [
                PALETTE["bankruptcy"] if "bankrupt" in str(c).lower()
                else PALETTE["non-bankruptcy"]
                for c in vc.index
            ]
            bars = axes[0].bar(vc.index, vc.values, color=colors,
                               width=0.45, edgecolor="#1C3355")
            axes[0].set_title("Absolute counts", color="#E0E8F4")
            axes[0].set_ylabel("Count", color="#C0CCDD")
            axes[0].grid(axis="y", alpha=0.3)
            for bar, val in zip(bars, vc.values):
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, str(val),
                    ha="center", va="bottom",
                    color="#E0E8F4", fontsize=11
                )

            # Donut
            wedges, texts, autotexts = axes[1].pie(
                vc.values, labels=vc.index, colors=colors,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "#0F1923", "linewidth": 2},
                pctdistance=0.78,
            )
            for t in texts:
                t.set_color("#C0CCDD")
            for at in autotexts:
                at.set_color("#0F1923"); at.set_fontweight("bold")
            centre = plt.Circle((0, 0), 0.55, color="#0F1923")
            axes[1].add_patch(centre)
            axes[1].text(0, 0, f"n={len(df)}", ha="center", va="center",
                         color="#E0E8F4", fontsize=12, fontweight="bold")
            axes[1].set_title("Proportion", color="#E0E8F4")

            plt.tight_layout()
            plt.show()


# ==========================================================
# STRATEGY 4: ORDINAL DISTRIBUTION  (NEW)
# ==========================================================

class OrdinalDistributionStrategy(DataInspectionStrategy):
    """
    Confirm that every numeric feature is truly ordinal
    with values {0.0, 0.5, 1.0} — and nothing else.

    Produces a value_counts table and a grid of count bars,
    one per feature.  Any unexpected values show up clearly.
    """

    EXPECTED_VALUES = {0.0, 0.5, 1.0}
    LEVEL_COLORS    = {0.0: "#4A9EFF", 0.5: "#F39C12", 1.0: "#E84545"}
    LEVEL_LABELS    = {0.0: "Low (0.0)", 0.5: "Med (0.5)", 1.0: "High (1.0)"}

    def inspect(self, df: pd.DataFrame) -> None:
        _set_style()

        print("=" * 64)
        print("  ORDINAL FEATURE DISTRIBUTION")
        print("=" * 64)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print("  No numeric columns found.")
            return

        # Validation table
        rows = []
        for col in numeric_cols:
            unique_vals = set(df[col].dropna().unique())
            is_ordinal  = unique_vals.issubset(self.EXPECTED_VALUES)
            unexpected  = unique_vals - self.EXPECTED_VALUES
            rows.append({
                "Feature":          col,
                "Unique values":    sorted(unique_vals),
                "Is ordinal":       "✓" if is_ordinal else "✗",
                "Unexpected vals":  sorted(unexpected) if unexpected else "—",
            })

        validation_df = pd.DataFrame(rows).set_index("Feature")
        display(validation_df)

        non_ordinal = [r["Feature"] for r in rows if r["Is ordinal"] == "✗"]
        if non_ordinal:
            print(f"\n  ⚠  Non-ordinal columns: {non_ordinal}")
        else:
            print("\n  ✓  All numeric features confirmed ordinal (0 / 0.5 / 1)")

        # Grid of count bars
        n      = len(numeric_cols)
        ncols  = 3
        nrows  = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(14, 4 * nrows))
        axes = np.array(axes).flatten()
        fig.patch.set_facecolor("#0F1923")
        fig.suptitle("Ordinal feature distributions",
                     fontsize=13, color="#E0E8F4", y=1.01)

        for idx, col in enumerate(numeric_cols):
            ax  = axes[idx]
            vc  = df[col].value_counts().sort_index()
            pct = (vc / len(df) * 100)
            colors = [self.LEVEL_COLORS.get(v, PALETTE["neutral"])
                      for v in vc.index]
            bars = ax.bar(
                [self.LEVEL_LABELS.get(v, str(v)) for v in vc.index],
                vc.values, color=colors, width=0.45, edgecolor="#1C3355"
            )
            ax.set_title(col.replace("_", " ").title(), color="#E0E8F4")
            ax.set_ylabel("Count", color="#C0CCDD")
            ax.grid(axis="y", alpha=0.3)
            for bar, val, p in zip(bars, vc.values, pct):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val}\n({p:.0f}%)",
                    ha="center", va="bottom",
                    color="#E0E8F4", fontsize=9
                )

        # Hide unused axes
        for ax in axes[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()


# ==========================================================
# DATA QUALITY REPORT  (NEW)
# ==========================================================

class DataQualityReport:
    """
    Single-pass data quality checklist.

    Runs every quality check and prints a PASS / WARN / FAIL
    scorecard — everything a reviewer needs before modelling.

    Usage
    -----
    DataQualityReport().run(df)
    """

    EXPECTED_COLUMNS = [
        "industrial_risk", "management_risk", "financial_flexibility",
        "credibility", "competitiveness", "operating_risk", "class",
    ]
    EXPECTED_ORDINAL = {0.0, 0.5, 1.0}
    EXPECTED_CLASSES = {"bankruptcy", "non-bankruptcy"}

    def run(self, df: pd.DataFrame) -> None:
        print("=" * 64)
        print("  DATA QUALITY REPORT")
        print("=" * 64)

        checks = []

        # 1. Schema
        missing_cols = [c for c in self.EXPECTED_COLUMNS
                        if c not in df.columns]
        extra_cols   = [c for c in df.columns
                        if c not in self.EXPECTED_COLUMNS]
        checks.append((
            "Schema — expected columns present",
            "PASS" if not missing_cols else "FAIL",
            f"Missing: {missing_cols}" if missing_cols else "All 7 columns present",
        ))
        if extra_cols:
            checks.append((
                "Schema — unexpected columns",
                "WARN",
                f"Extra columns: {extra_cols}",
            ))

        # 2. Null values
        total_nulls = int(df.isnull().sum().sum())
        checks.append((
            "Missing values",
            "PASS" if total_nulls == 0 else "FAIL",
            f"{total_nulls} null values" if total_nulls else "Zero missing values",
        ))

        # 3. Ordinal values
        feat_cols = [c for c in self.EXPECTED_COLUMNS if c != "class"
                     and c in df.columns]
        bad_feat  = []
        for c in feat_cols:
            unique = set(df[c].dropna().unique())
            if not unique.issubset(self.EXPECTED_ORDINAL):
                bad_feat.append(f"{c}: {unique - self.EXPECTED_ORDINAL}")
        checks.append((
            "Ordinal values — features are {0, 0.5, 1}",
            "PASS" if not bad_feat else "FAIL",
            "All features confirmed {0.0, 0.5, 1.0}" if not bad_feat
            else f"Unexpected: {bad_feat}",
        ))

        # 4. Target classes
        if "class" in df.columns:
            found_classes = set(df["class"].astype(str).str.strip().unique())
            unexpected    = found_classes - self.EXPECTED_CLASSES
            checks.append((
                "Target classes — {bankruptcy, non-bankruptcy}",
                "PASS" if not unexpected else "FAIL",
                f"Classes: {found_classes}" if not unexpected
                else f"Unexpected: {unexpected}",
            ))

        # 5. Dataset size
        n = len(df)
        checks.append((
            "Dataset size (rows)",
            "PASS" if n >= 100 else "WARN",
            f"{n:,} rows  ({'adequate' if n >= 100 else 'small — use cross-validation'})",
        ))

        # 6. Class balance
        if "class" in df.columns:
            vc    = df["class"].astype(str).str.strip().value_counts()
            ratio = vc.min() / vc.max() if len(vc) >= 2 else 1.0
            checks.append((
                "Class balance (minority / majority ratio)",
                "PASS" if ratio > 0.7 else "WARN" if ratio > 0.4 else "FAIL",
                f"Ratio = {ratio:.4f}  "
                f"({'balanced' if ratio > 0.7 else 'mild imbalance' if ratio > 0.4 else 'severe imbalance'})",
            ))

        # 7. Duplicates
        dup_count = df.duplicated().sum()
        dup_pct   = dup_count / n * 100
        checks.append((
            "Duplicate rows",
            "PASS" if dup_pct < 10 else "WARN" if dup_pct < 50 else "WARN",
            f"{dup_count:,} duplicates ({dup_pct:.1f}%) — "
            f"{'acceptable' if dup_pct < 10 else 'high — investigate impact on model'}",
        ))

        # ── Print scorecard ───────────────────────────────────
        status_icons = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}
        status_colors = {"PASS": "\033[92m", "WARN": "\033[93m",
                         "FAIL": "\033[91m", "RESET": "\033[0m"}

        print()
        for check, status, detail in checks:
            icon = status_icons.get(status, "?")
            print(f"  {icon}  [{status:<4}]  {check}")
            print(f"          {detail}")
            print()

        # Summary
        n_pass = sum(1 for _, s, _ in checks if s == "PASS")
        n_warn = sum(1 for _, s, _ in checks if s == "WARN")
        n_fail = sum(1 for _, s, _ in checks if s == "FAIL")
        total  = len(checks)

        print("─" * 64)
        print(f"  Result: {n_pass}/{total} PASS  "
              f"{n_warn} WARN  {n_fail} FAIL")
        if n_fail == 0 and n_warn == 0:
            print("  ✓  Dataset is production-ready for modelling.")
        elif n_fail == 0:
            print("  ⚠  Dataset has warnings — review before modelling.")
        else:
            print("  ✗  Dataset has failures — fix before modelling.")


# ==========================================================
# CONTEXT CLASS  (unchanged API)
# ==========================================================

class DataInspector:
    """
    Execute data inspection strategies.

    Allows swapping strategies at runtime without
    changing the inspection logic.

    Usage
    -----
    inspector = DataInspector(BasicStructureInspectionStrategy())
    inspector.execute(df)

    inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    inspector.execute(df)
    """

    def __init__(self, strategy: DataInspectionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy) -> None:
        self._strategy = strategy

    def execute(self, df: pd.DataFrame) -> None:
        self._strategy.inspect(df)