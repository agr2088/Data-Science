"""
layers/layer_1_command.py
LAYER 1 — COMMAND DECK
No sidebar — all global controls live here.
Returns a state dict consumed by every downstream layer.
"""
import streamlit as st
import pandas as pd


MODEL_MAP: dict[str, str] = {
    "KMeans ★":      "KMeans_Cluster",
    "GMM":           "GMM_Cluster",
    "Agglomerative": "Hierarchical_Cluster",
    "Spectral":      "Spectral_Cluster",
    "BIRCH":         "Birch_Cluster",
}

LABEL_MAP: dict[str, str] = {
    "KMeans_Cluster":       "KMeans_Cluster_Label",
    "GMM_Cluster":          "GMM_Cluster_Label",
    "Hierarchical_Cluster": "Hierarchical_Cluster_Label",
    "Spectral_Cluster":     "Spectral_Cluster_Label",
    "Birch_Cluster":        "Birch_Cluster_Label",
}

CLUSTER_NAMES: dict[int, str] = {
    0: "High-Income Developed",
    1: "Upper-Middle Income",
    2: "Lower-Middle Income",
    3: "Low-Income / Fragile",
}

FEATURE_COLS: list[str] = [
    "Birth Rate",
    "Business Tax Rate",
    "CO2 Emissions",
    "Days to Start Business",
    "GDP",
    "Health Exp % GDP",
    "Health Exp/Capita",
    "Hours to do Tax",
    "Infant Mortality Rate",
    "Internet Usage",
    "Lending Interest",
    "Mobile Phone Usage",
    "Population 65+",
    "Population Total",
    "Population Urban",
    "Tourism Inbound",
    "Internet Usage_trend",
    "CO2 Emissions_trend",
    "Life Expectancy Female_trend",
    "Mobile Phone Usage_trend",
]


def render_command(df: pd.DataFrame) -> dict:
    """
    Renders the sticky command deck with all global controls.

    Returns
    -------
    state : dict
        active_model       : column name, e.g. "KMeans_Cluster"
        active_label_col   : matching label column
        active_clusters    : list[int] — empty means "All"
        selected_features  : list[str]
        country_query      : str — empty means no search
    """
    st.markdown('<div class="command-deck">', unsafe_allow_html=True)

    col_m, col_c, col_f, col_s = st.columns([2.5, 2, 3, 2])

    with col_m:
        st.markdown('<p class="cmd-label">MODEL</p>', unsafe_allow_html=True)
        active_model_label = st.radio(
            "model",
            list(MODEL_MAP.keys()),
            horizontal=True,
            label_visibility="collapsed",
        )
        active_model = MODEL_MAP[active_model_label]

    with col_c:
        st.markdown('<p class="cmd-label">CLUSTER FILTER</p>', unsafe_allow_html=True)
        cluster_opts = ["All"] + [f"{i} · {CLUSTER_NAMES[i]}" for i in range(4)]
        selected_c = st.selectbox(
            "cluster",
            cluster_opts,
            label_visibility="collapsed",
        )
        active_clusters: list[int] = [] if selected_c == "All" else [int(selected_c[0])]

    with col_f:
        st.markdown('<p class="cmd-label">FEATURES</p>', unsafe_allow_html=True)
        # Only offer features that actually exist in the dataframe
        available_features = [f for f in FEATURE_COLS if f in df.columns]
        default_features = [
            f for f in [
                "GDP",
                "Internet Usage",
                "Infant Mortality Rate",
                "Health Exp % GDP",
                "Life Expectancy Female_trend",
            ]
            if f in available_features
        ]
        selected_features = st.multiselect(
            "features",
            available_features,
            default=default_features,
            max_selections=6,
            label_visibility="collapsed",
        )

    with col_s:
        st.markdown('<p class="cmd-label">COUNTRY SEARCH</p>', unsafe_allow_html=True)
        country_query = st.text_input(
            "country",
            placeholder="e.g. India, Nigeria…",
            label_visibility="collapsed",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    return {
        "active_model":      active_model,
        "active_label_col":  LABEL_MAP[active_model],
        "active_clusters":   active_clusters,
        "selected_features": selected_features,
        "country_query":     country_query.strip(),
    }
