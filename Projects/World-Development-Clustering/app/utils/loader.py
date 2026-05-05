"""
utils/loader.py
Cached data loading — no redundant I/O on re-renders.

BUGS FIXED:
  - load_cluster_profiles: set_index check was fragile when first column
    name doesn't contain "Cluster"; now uses positional check.
  - load_metrics: JSON cluster_sizes keys are strings; kept as-is (callers cast).
  - Added graceful fallback if Label_Agreement column values are > 1.0
    (some pipelines store raw count instead of ratio).
"""

import os
import json
import streamlit as st
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.config import CLUSTERED_DATA, REPORTS_DIR, PROCESSED_DIR


@st.cache_data(show_spinner=False)
def load_clustered_data() -> pd.DataFrame:
    df = pd.read_csv(CLUSTERED_DATA)

    # ── Coerce cluster columns to int ───────────────────────────────
    cluster_cols = [c for c in df.columns if c.endswith("_Cluster")]
    for col in cluster_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # ── Ensure Label_Agreement is a 0-1 float ───────────────────────
    if "Label_Agreement" not in df.columns:
        label_cols = [c for c in df.columns if c.endswith("_Label")]
        if label_cols:
            df["Label_Agreement"] = df[label_cols].apply(
                lambda row: row.value_counts().iloc[0] / len(row), axis=1
            )
        else:
            df["Label_Agreement"] = 1.0
    else:
        df["Label_Agreement"] = pd.to_numeric(df["Label_Agreement"], errors="coerce").fillna(1.0)
        # Normalise if stored as count (e.g. 3 out of 5 models)
        max_val = df["Label_Agreement"].max()
        if max_val > 1.0:
            df["Label_Agreement"] = df["Label_Agreement"] / max_val

    # ── Ensure Majority_Label ───────────────────────────────────────
    if "Majority_Label" not in df.columns or df["Majority_Label"].isna().all():
        label_cols = [c for c in df.columns if c.endswith("_Label")]
        if label_cols:
            df["Majority_Label"] = df[label_cols].mode(axis=1).iloc[:, 0]
        else:
            df["Majority_Label"] = "Unknown"
    else:
        df["Majority_Label"] = df["Majority_Label"].fillna("Unknown")

    return df


@st.cache_data(show_spinner=False)
def load_pca_data() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "pca_data.csv")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_metrics() -> list:
    path = os.path.join(REPORTS_DIR, "evaluation_metrics.json")
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_cluster_profiles(model_col: str) -> pd.DataFrame:
    """
    Load cluster_profiles_{model_col}.csv.
    Returns a DataFrame indexed by cluster id (int).
    The first column is assumed to be the cluster ID column.
    """
    fname = f"cluster_profiles_{model_col}.csv"
    path  = os.path.join(REPORTS_DIR, fname)
    df    = pd.read_csv(path)

    # Set the first column as index (it's always the cluster-ID column)
    first_col = df.columns[0]
    df = df.set_index(first_col)
    df.index = df.index.astype(int)
    return df