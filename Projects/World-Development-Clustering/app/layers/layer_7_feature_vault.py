"""
layers/layer_7_feature_vault.py
LAYER 7 — FEATURE VAULT
Feature importance heatmap · Correlation matrix · Violin plots.
All inside an expandable section with tabs.
"""
import streamlit as st
import pandas as pd

from utils.theme import render_section_header, render_gold_divider
from utils.charts import build_importance_heatmap, build_correlation_matrix, build_violin


def render_feature_vault(df: pd.DataFrame, state: dict) -> None:
    render_section_header(
        "◇ FEATURE VAULT",
        "Feature Importance · Correlation · Distribution",
    )

    selected_features = state["selected_features"]
    active_model = state["active_model"]

    with st.expander("▸ Expand Feature Analysis", expanded=False):

        if not selected_features:
            st.info("Select features in the Command Deck to explore the Feature Vault.")
            return

        tab1, tab2, tab3 = st.tabs(["Importance", "Correlation", "Distribution"])

        # ── Tab 1: Feature Importance heatmap (z-score deviation per cluster)
        with tab1:
            st.markdown(
                '<p class="section-micro">Z-Score deviation of each cluster mean vs global mean</p>',
                unsafe_allow_html=True,
            )
            avail = [f for f in selected_features if f in df.columns]
            if avail:
                fig = build_importance_heatmap(df, active_model, avail)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.warning("None of the selected features exist in the dataset.")

        # ── Tab 2: Correlation matrix
        with tab2:
            avail = [f for f in selected_features if f in df.columns]
            if len(avail) >= 2:
                fig = build_correlation_matrix(df, avail)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.info("Select at least 2 features to display the correlation matrix.")

        # ── Tab 3: Violin plots per cluster
        with tab3:
            avail = [f for f in selected_features if f in df.columns]
            if avail:
                feature_for_violin = st.selectbox(
                    "Select feature for distribution",
                    avail,
                    label_visibility="collapsed",
                    key="violin_feature_select",
                )
                fig = build_violin(df, active_model, feature_for_violin)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.warning("None of the selected features exist in the dataset.")

    render_gold_divider()
