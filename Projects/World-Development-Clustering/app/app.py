"""
app.py — World Development Intelligence Dashboard
Royal Black + Gold | No Sidebar | 7-Layer Architecture
"""
import streamlit as st

# ── Page config (MUST be first st call)
st.set_page_config(
    page_title = "World Development Intelligence",
    page_icon  = "🌍",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

# ── CSS injection
from utils.theme import inject_global_css
inject_global_css()   # loads full CSS from theme.py

# ── Data loading
from utils.loader import load_clustered_data, load_pca_data, load_metrics
df      = load_clustered_data()
pca_df  = load_pca_data()
metrics = load_metrics()

# ── Layers
from layers.layer_0_hero         import render_hero
from layers.layer_1_command      import render_command
from layers.layer_2_intelligence import render_intelligence
from layers.layer_3_world_map    import render_world_map
from layers.layer_4_cluster_atlas import render_cluster_atlas
from layers.layer_5_country_lens  import render_country_lens
from layers.layer_6_model_arena   import render_model_arena
from layers.layer_7_feature_vault import render_feature_vault

# ── Render stack
render_hero()
state = render_command(df)          # ← returns global state dict
render_intelligence(df, metrics, state)
render_world_map(df, state)
render_cluster_atlas(df, pca_df, state)
render_country_lens(df, state)      # ← conditional on state["country_query"]
render_model_arena(metrics)
render_feature_vault(df, state)