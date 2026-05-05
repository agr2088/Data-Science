"""
layers/layer_0_hero.py
LAYER 0 — HERO BANNER
Animated gold title, tagline, model pills, stat badges.
"""
import streamlit as st


def render_hero():
    st.markdown(
        """
        <div class="hero-wrap">
          <div class="hero-crown">✦</div>
          <h1 class="hero-title">WORLD DEVELOPMENT INTELLIGENCE</h1>
          <p class="hero-tagline">Unsupervised Intelligence · 198 Nations · 5 Clustering Models</p>
          <div class="hero-pills">
            <span class="pill">KMeans ★</span>
            <span class="pill">GMM</span>
            <span class="pill">Agglomerative</span>
            <span class="pill">Spectral</span>
            <span class="pill">BIRCH</span>
          </div>
          <div class="hero-badges">
            <span class="badge">k = 4 Clusters</span>
            <span class="badge">198 Countries</span>
            <span class="badge">19 Features</span>
            <span class="badge">4 Trend Signals</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
