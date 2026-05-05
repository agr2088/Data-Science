"""
layers/layer_4_cluster_atlas.py
LAYER 4 — CLUSTER ATLAS
Left: PCA scatter | Right: Cluster radar
Bottom: 4 cluster profile cards
"""
import streamlit as st
import pandas as pd

from utils.theme import render_section_header, render_gold_divider
from utils.charts import build_pca_scatter, build_cluster_radar
from utils.loader import load_cluster_profiles

CLUSTER_NAMES = {
    0: "High-Income Developed",
    1: "Upper-Middle Income",
    2: "Lower-Middle Income",
    3: "Low-Income / Fragile States",
}

CLUSTER_COLORS = {
    0: "#D4A017",
    1: "#7EC8A4",
    2: "#6EB3D4",
    3: "#C97B5E",
}


def _render_cluster_card(cluster_id: int, stats: pd.Series) -> None:
    """Renders a single cluster profile card."""
    color = CLUSTER_COLORS.get(cluster_id, "#D4A017")
    name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")

    # Safely extract stats — handle both scaled and raw profile CSVs
    def _get(key: str, fmt: str = ".2f") -> str:
        if key in stats.index:
            try:
                val = float(stats[key])
                return format(val, fmt)
            except (ValueError, TypeError):
                return "—"
        return "—"

    count_val = _get("count", ".0f") if "count" in stats.index else "—"
    gdp_val = _get("GDP", ",.4f") if "GDP" in stats.index else "—"
    internet_val = _get("Internet Usage", ".4f") if "Internet Usage" in stats.index else "—"
    infant_val = _get("Infant Mortality Rate", ".4f") if "Infant Mortality Rate" in stats.index else "—"

    st.markdown(
        f"""
        <div class="cluster-card" style="border-top: 2px solid {color}">
          <div class="cc-id" style="color:{color}">CLUSTER {cluster_id}</div>
          <div class="cc-name">{name}</div>
          <div class="cc-stats">
            <div class="cc-stat">
              <span class="cc-stat-label">Countries</span>
              <span class="cc-stat-value">{count_val}</span>
            </div>
            <div class="cc-stat">
              <span class="cc-stat-label">Avg GDP (std)</span>
              <span class="cc-stat-value">{gdp_val}</span>
            </div>
            <div class="cc-stat">
              <span class="cc-stat-label">Internet Usage</span>
              <span class="cc-stat-value">{internet_val}</span>
            </div>
            <div class="cc-stat">
              <span class="cc-stat-label">Infant Mortality</span>
              <span class="cc-stat-value">{infant_val}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_cluster_atlas(
    df: pd.DataFrame,
    pca_df: pd.DataFrame,
    state: dict,
) -> None:
    render_section_header(
        "◈ CLUSTER ATLAS",
        "PCA Topology + Cluster Profiles",
    )

    active_model = state["active_model"]
    active_label_col = state["active_label_col"]
    selected_features = state["selected_features"]

    # ── Merge PCA coords with cluster labels
    merge_cols = ["Country", active_model, active_label_col, "Majority_Label"]
    merge_cols = [c for c in merge_cols if c in df.columns]
    df_pca_merged = pca_df.merge(df[merge_cols], on="Country", how="left")

    left, right = st.columns([1.3, 1])

    with left:
        fig_scatter = build_pca_scatter(df_pca_merged, active_model)
        st.plotly_chart(
            fig_scatter,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right:
        cluster_id = st.selectbox(
            "Profile Cluster",
            [0, 1, 2, 3],
            format_func=lambda x: f"Cluster {x} — {CLUSTER_NAMES[x]}",
            label_visibility="collapsed",
            key="atlas_cluster_select",
        )
        if selected_features:
            fig_radar = build_cluster_radar(df, active_model, cluster_id, selected_features)
            st.plotly_chart(
                fig_radar,
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("Select features in the Command Deck to display the radar chart.")

    # ── 4 Cluster profile cards
    render_gold_divider()

    try:
        profiles = load_cluster_profiles(active_model)
        # profiles index should be integer cluster ids 0-3
        # Add count column from df
        count_series = df.groupby(active_model).size().rename("count")
        if "count" not in profiles.columns:
            profiles = profiles.join(count_series, how="left")

        cols = st.columns(4)
        for col, cid in zip(cols, range(4)):
            with col:
                if cid in profiles.index:
                    _render_cluster_card(cid, profiles.loc[cid])
                else:
                    st.markdown(
                        f'<div class="cluster-card" style="border-top:2px solid {CLUSTER_COLORS[cid]}">'
                        f'<div class="cc-id" style="color:{CLUSTER_COLORS[cid]}">CLUSTER {cid}</div>'
                        f'<div class="cc-name">{CLUSTER_NAMES[cid]}</div>'
                        f'<p style="color:#5A5040;font-size:11px">No profile data</p>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
    except Exception as exc:
        st.warning(f"Could not load cluster profiles: {exc}")

    render_gold_divider()
