"""
layers/layer_2_intelligence.py
LAYER 2 — INTELLIGENCE STRIP
6 KPI cards + mini silhouette bar chart.
"""
import streamlit as st
import pandas as pd

from utils.theme import render_section_header, render_gold_divider, render_kpi_card
from utils.charts import build_model_mini_bar


def render_intelligence(df: pd.DataFrame, metrics: list, state: dict) -> None:
    render_section_header(
        "◆ INTELLIGENCE",
        "Key Performance Indicators — Cross-Model Overview",
    )

    active_model = state["active_model"]
    active_clusters = state["active_clusters"]

    # ── Filter to current view
    df_view = df.copy()
    if active_clusters:
        df_view = df_view[df_view[active_model].isin(active_clusters)]

    # ── Compute KPIs
    n_countries = len(df_view)
    n_clusters = int(df_view[active_model].nunique())

    valid_metrics = [m for m in metrics if not m.get("invalid", False)]
    if valid_metrics:
        best_model_dict = max(valid_metrics, key=lambda m: m["silhouette_score"])
        best_model_name = best_model_dict["model"]
        best_sil = best_model_dict["silhouette_score"]
        best_db = min(m["davies_bouldin"] for m in valid_metrics)
    else:
        # Fallback: use all metrics
        best_model_dict = max(metrics, key=lambda m: m["silhouette_score"])
        best_model_name = best_model_dict["model"]
        best_sil = best_model_dict["silhouette_score"]
        best_db = min(m["davies_bouldin"] for m in metrics)

    # Label agreement: fraction of rows where all 5 model labels agree
    if "Label_Agreement" in df.columns:
        agreement = float(df["Label_Agreement"].mean()) * 100
    else:
        agreement = 0.0

    # ── 6 KPI cards
    kpi_data = [
        ("🌍", "COUNTRIES",       str(n_countries),        "in view"),
        ("◈",  "CLUSTERS",        str(n_clusters),         "active"),
        ("★",  "BEST MODEL",      best_model_name,         "by silhouette"),
        ("◆",  "SILHOUETTE",      f"{best_sil:.4f}",       "higher = better"),
        ("◇",  "DAVIES-BOULDIN",  f"{best_db:.4f}",        "lower = better"),
        ("◉",  "LABEL AGREEMENT", f"{agreement:.1f}%",     "cross-model"),
    ]

    cols = st.columns(6)
    for col, (icon, label, value, sub) in zip(cols, kpi_data):
        with col:
            render_kpi_card(icon, label, value, sub)

    # ── Mini model silhouette bar
    st.markdown('<div class="intel-chart-wrap">', unsafe_allow_html=True)
    fig = build_model_mini_bar(metrics)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    render_gold_divider()
