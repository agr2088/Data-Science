"""
layers/layer_6_model_arena.py
LAYER 6 — MODEL ARENA
Full 5-model benchmark comparison inside an expander.
Metrics table + validity badges + cluster-size mini bars.
"""
import streamlit as st

from utils.theme import render_section_header, render_gold_divider
from utils.charts import build_cluster_size_bar


def _render_model_row(m: dict, is_selected: bool) -> None:
    """Single model row: name, metrics, status badge, winner crown."""
    is_invalid = m.get("invalid", False)

    status_html = (
        '<span class="badge-invalid">⚠ INVALID</span>'
        if is_invalid
        else '<span class="badge-valid">✓ VALID</span>'
    )
    selected_html = (
        '<span class="badge-winner">★ SELECTED</span>' if is_selected else "—"
    )
    row_class = "model-row"
    if is_selected:
        row_class += " model-row--winner"
    if is_invalid:
        row_class += " model-row--invalid"

    sil = f"{m['silhouette_score']:.4f}"
    db  = f"{m['davies_bouldin']:.4f}"
    ch  = f"{m['calinski_harabasz']:.2f}"

    st.markdown(
        f"""
        <div class="{row_class}">
          <span class="mr-name">{m["model"]}</span>
          <span class="mr-metric">{sil}</span>
          <span class="mr-metric">{db}</span>
          <span class="mr-metric">{ch}</span>
          <span class="mr-status">{status_html}</span>
          <span class="mr-selected">{selected_html}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_arena(metrics: list) -> None:
    render_section_header(
        "★ MODEL ARENA",
        "5-Model Benchmark Comparison — Silhouette · Davies-Bouldin · Calinski-Harabasz",
    )

    # Determine the overall winner (best silhouette among valid models)
    valid = [m for m in metrics if not m.get("invalid", False)]
    winner_name = max(valid, key=lambda m: m["silhouette_score"])["model"] if valid else ""

    with st.expander("▸ Expand Model Benchmarks", expanded=False):

        # ── Table header
        st.markdown(
            """
            <div class="model-row model-row--header">
              <span class="mr-name">MODEL</span>
              <span class="mr-metric">SILHOUETTE ↑</span>
              <span class="mr-metric">DAVIES-BOULDIN ↓</span>
              <span class="mr-metric">CALINSKI-HARABASZ ↑</span>
              <span class="mr-status">STATUS</span>
              <span class="mr-selected">SELECTED</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Sort: best silhouette first
        for m in sorted(metrics, key=lambda x: -x["silhouette_score"]):
            is_selected = m["model"] == winner_name
            _render_model_row(m, is_selected)

        render_gold_divider()

        # ── Cluster size distribution — one mini bar per model
        st.markdown(
            '<p class="section-micro" style="margin-bottom:8px">CLUSTER SIZE DISTRIBUTION</p>',
            unsafe_allow_html=True,
        )
        size_cols = st.columns(len(metrics))
        for col, m in zip(size_cols, metrics):
            with col:
                fig = build_cluster_size_bar(m)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

    render_gold_divider()
