"""
layers/layer_3_world_map.py
LAYER 3 — WORLD MAP
Full-width choropleth, single-model or 5-model small-multiples toggle.
"""
import streamlit as st
import pandas as pd

from utils.theme import render_section_header, render_gold_divider
from utils.charts import build_choropleth_single, build_choropleth_mini

CLUSTER_COLORS = {
    0: "#D4A017",
    1: "#7EC8A4",
    2: "#6EB3D4",
    3: "#C97B5E",
}

MODEL_COLS = [
    ("KMeans_Cluster",       "KMeans ★"),
    ("GMM_Cluster",          "GMM"),
    ("Hierarchical_Cluster", "Agglomerative ⚠ invalid"),
    ("Spectral_Cluster",     "Spectral"),
    ("Birch_Cluster",        "BIRCH"),
]


def render_world_map(df: pd.DataFrame, state: dict) -> None:
    render_section_header(
        "🌍 WORLD MAP",
        "Choropleth — Countries Coloured by Cluster Assignment",
    )

    left_ctrl, right_ctrl = st.columns([1, 3])
    with left_ctrl:
        view_toggle = st.radio(
            "map_view",
            ["Single Model", "All 5 Models"],
            horizontal=False,
            label_visibility="collapsed",
        )

    if view_toggle == "Single Model":
        active_model = state["active_model"]
        # Filter if cluster filter is active
        df_map = df.copy()
        if state["active_clusters"]:
            # Grey out non-selected clusters by assigning NaN placeholder
            # Better: just show full map but highlight via the colorscale
            pass  # choropleth always shows all countries
        fig = build_choropleth_single(df_map, active_model, CLUSTER_COLORS)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        # 5-model small-multiples (2 rows × 3 cols, last slot empty)
        row1 = st.columns(3)
        row2 = st.columns(3)
        all_cols = row1 + row2

        for i, (col_name, title) in enumerate(MODEL_COLS):
            if col_name not in df.columns:
                continue
            with all_cols[i]:
                fig = build_choropleth_mini(df, col_name, title, CLUSTER_COLORS)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

    # Cluster colour legend
    st.markdown(
        """
        <div class="map-legend">
          <span class="legend-item" style="color:#D4A017">■ Cluster 0 — High-Income Developed</span>
          <span class="legend-item" style="color:#7EC8A4">■ Cluster 1 — Upper-Middle Income</span>
          <span class="legend-item" style="color:#6EB3D4">■ Cluster 2 — Lower-Middle Income</span>
          <span class="legend-item" style="color:#C97B5E">■ Cluster 3 — Low-Income / Fragile</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_gold_divider()
