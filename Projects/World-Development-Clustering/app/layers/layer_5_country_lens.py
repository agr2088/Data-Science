"""
layers/layer_5_country_lens.py
LAYER 5 — COUNTRY LENS
Conditional render: only shown when a country query is entered.
Country banner + feature comparison bar + cross-model membership table.
"""
import streamlit as st
import pandas as pd

from utils.theme import render_section_header, render_gold_divider
from utils.charts import build_country_feature_bars

CLUSTER_COLORS = {
    0: "#D4A017",
    1: "#7EC8A4",
    2: "#6EB3D4",
    3: "#C97B5E",
}

MODEL_COLS = {
    "KMeans ★":      ("KMeans_Cluster",       "KMeans_Cluster_Label"),
    "GMM":           ("GMM_Cluster",           "GMM_Cluster_Label"),
    "Agglomerative": ("Hierarchical_Cluster",  "Hierarchical_Cluster_Label"),
    "Spectral":      ("Spectral_Cluster",       "Spectral_Cluster_Label"),
    "BIRCH":         ("Birch_Cluster",          "Birch_Cluster_Label"),
}


def _render_model_membership(row: pd.Series) -> None:
    """Render cross-model cluster assignment table."""
    rows_html = ""
    for model_name, (col, lcol) in MODEL_COLS.items():
        if col not in row.index or lcol not in row.index:
            continue
        try:
            cid = int(row[col])
        except (ValueError, TypeError):
            cid = 0
        label = row.get(lcol, "—")
        color = CLUSTER_COLORS.get(cid, "#5A5040")
        rows_html += (
            f"<tr>"
            f'<td class="mm-model">{model_name}</td>'
            f"<td>"
            f'<span class="mm-badge" style="background:{color}22;border:1px solid {color};color:{color}">'
            f"{cid} · {label}"
            f"</span>"
            f"</td>"
            f"</tr>"
        )

    majority = row.get("Majority_Label", "—")
    agreement_raw = row.get("Label_Agreement", None)
    if agreement_raw is not None:
        try:
            agreement_pct = f"{float(agreement_raw):.0%}"
        except (ValueError, TypeError):
            agreement_pct = "—"
    else:
        agreement_pct = "—"

    st.markdown(
        f"""
        <div class="membership-wrap">
          <p class="section-micro">CROSS-MODEL ASSIGNMENT</p>
          <table class="membership-table">{rows_html}</table>
          <div class="majority-label">
            Majority: <strong>{majority}</strong>
            &nbsp;·&nbsp; Agreement: <strong>{agreement_pct}</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_country_lens(df: pd.DataFrame, state: dict) -> None:
    """Rendered only when country_query is non-empty."""
    query = state["country_query"]
    if not query:
        return

    render_section_header(
        "◉ COUNTRY LENS",
        f'Drill-Down — "{query}"',
    )

    # Fuzzy match
    matches = df[df["Country"].str.contains(query, case=False, na=False, regex=False)]
    if matches.empty:
        st.markdown(
            f'<div class="no-result">⚠ No country matching "<em>{query}</em>" found.</div>',
            unsafe_allow_html=True,
        )
        render_gold_divider()
        return

    country_row = matches.iloc[0]
    active_model = state["active_model"]
    active_label_col = state["active_label_col"]

    try:
        cid = int(country_row[active_model])
    except (ValueError, TypeError, KeyError):
        cid = 0

    label = country_row.get(active_label_col, "—")
    color = CLUSTER_COLORS.get(cid, "#D4A017")

    agreement_raw = country_row.get("Label_Agreement", None)
    if agreement_raw is not None:
        try:
            agreement_str = f"{float(agreement_raw):.0%}"
        except (ValueError, TypeError):
            agreement_str = "—"
    else:
        agreement_str = "—"

    # ── Country Banner
    st.markdown(
        f"""
        <div class="country-banner" style="border-left: 4px solid {color}; border-radius: 8px;
             background: #111118; padding: 18px 24px; margin-bottom: 16px;
             display:flex; align-items:center; gap:24px; flex-wrap:wrap;">
          <span class="cb-country" style="font-family:'Cinzel',serif;
                font-size:22px; color:{color}; letter-spacing:0.1em;">
            {country_row["Country"]}
          </span>
          <span class="cb-cluster" style="color:{color}; font-family:'JetBrains Mono',monospace;
                font-size:12px; background:{color}18; border:1px solid {color};
                border-radius:100px; padding:3px 12px;">
            Cluster {cid} · {label}
          </span>
          <span class="cb-agree" style="color:#B8A87A; font-family:'JetBrains Mono',monospace;
                font-size:11px;">
            Label Agreement: {agreement_str}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_features = state["selected_features"]
    left, right = st.columns([1.6, 1])

    with left:
        if selected_features:
            fig = build_country_feature_bars(
                country_row, df, active_model, selected_features
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("Select features in the Command Deck to display the comparison chart.")

    with right:
        _render_model_membership(country_row)

    render_gold_divider()
