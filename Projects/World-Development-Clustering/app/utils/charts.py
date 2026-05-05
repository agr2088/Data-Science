"""
utils/charts.py
All Plotly figure builders — Royal Black + Gold themed.

BUGS FIXED:
  - Duplicate `fillcolor` assignment in build_cluster_radar() (line 249/250)
  - DARK_LAYOUT xaxis/yaxis passed incorrectly via **extra (keys clash); now
    done with fig.update_layout separately where needed
  - build_country_feature_bars used .get() on pd.Series which always returns
    scalar — replaced with safe bracket + .at access
  - build_cluster_size_bar: cluster_sizes keys may be strings; cast to int safely
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Shared constants ─────────────────────────────────────────────────────────

CLUSTER_COLORS = {
    0: "#D4A017",
    1: "#7EC8A4",
    2: "#6EB3D4",
    3: "#C97B5E",
}

CLUSTER_NAMES = {
    0: "High-Income Developed",
    1: "Upper-Middle Income",
    2: "Lower-Middle Income",
    3: "Low-Income / Fragile States",
}

GOLD_DIVERGING = [
    [0.0,  "#0A0A0F"],
    [0.25, "#5C4300"],
    [0.5,  "#A67C00"],
    [0.75, "#D4A017"],
    [1.0,  "#FFD966"],
]

DARK_LAYOUT = dict(
    paper_bgcolor="#050507",
    plot_bgcolor="#0A0A0F",
    font=dict(family="Inter, sans-serif", color="#B8A87A", size=11),
    title_font=dict(family="Cinzel, serif", color="#FFD966", size=13),
    xaxis=dict(gridcolor="#18181F", linecolor="#2A2415", tickcolor="#5A5040", zeroline=False),
    yaxis=dict(gridcolor="#18181F", linecolor="#2A2415", tickcolor="#5A5040", zeroline=False),
    legend=dict(bgcolor="#111118", bordercolor="#2A2415", borderwidth=1),
    margin=dict(l=8, r=8, t=36, b=8),
    hoverlabel=dict(
        bgcolor="#18181F",
        bordercolor="#A67C00",
        font_color="#F5ECD7",
        font_family="Inter, sans-serif",
    ),
)

_AXIS_DEFAULTS = dict(gridcolor="#18181F", linecolor="#2A2415", tickcolor="#5A5040", zeroline=False)


def _apply_dark(fig: go.Figure, **extra) -> go.Figure:
    """Apply DARK_LAYOUT then any extra overrides (height, title_text, etc.)."""
    layout = dict(
        paper_bgcolor=DARK_LAYOUT["paper_bgcolor"],
        plot_bgcolor=DARK_LAYOUT["plot_bgcolor"],
        font=DARK_LAYOUT["font"],
        title_font=DARK_LAYOUT["title_font"],
        legend=DARK_LAYOUT["legend"],
        margin=DARK_LAYOUT["margin"],
        hoverlabel=DARK_LAYOUT["hoverlabel"],
    )
    layout.update(extra)
    fig.update_layout(**layout)
    # Always ensure axis defaults unless overridden in extra
    if "xaxis" not in extra:
        fig.update_xaxes(**_AXIS_DEFAULTS)
    if "yaxis" not in extra:
        fig.update_yaxes(**_AXIS_DEFAULTS)
    return fig


def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert #RRGGBB to rgba(r,g,b,alpha)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Mini model silhouette bar ─────────────────────────────────────────────────

def build_model_mini_bar(metrics: list) -> go.Figure:
    """Horizontal bar comparing silhouette scores; invalid models greyed out."""
    models   = [m["model"] for m in metrics]
    scores   = [m["silhouette_score"] for m in metrics]
    invalids = [m["invalid"] for m in metrics]

    colors = ["#5A5040" if inv else "#D4A017" for inv in invalids]
    valid_scores = [s for s, inv in zip(scores, invalids) if not inv]
    best = max(valid_scores) if valid_scores else max(scores)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=models,
        x=scores,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s:.4f}" + (" ⚠" if inv else "") for s, inv in zip(scores, invalids)],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=10, color="#B8A87A"),
        hovertemplate="<b>%{y}</b><br>Silhouette: %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(
        x=best,
        line=dict(color="#FFD966", width=1, dash="dot"),
        annotation_text=f"Best: {best:.4f}",
        annotation_font=dict(color="#FFD966", size=10, family="JetBrains Mono, monospace"),
        annotation_position="top right",
    )
    _apply_dark(fig, height=180, title_text="Silhouette Score by Model")
    fig.update_xaxes(**_AXIS_DEFAULTS, range=[0, max(scores) * 1.25])
    fig.update_layout(margin=dict(l=8, r=60, t=36, b=8))
    return fig


# ── Choropleth — single model ─────────────────────────────────────────────────

def build_choropleth_single(df: pd.DataFrame, cluster_col: str, colors: dict) -> go.Figure:
    label_col   = cluster_col + "_Label"
    custom_cols = [c for c in [label_col, "GDP", "Internet Usage"] if c in df.columns]
    customdata  = df[custom_cols].values if custom_cols else None

    hover = "<b>%{location}</b><br>Cluster: %{z}<br>"
    idx_map = {col: i for i, col in enumerate(custom_cols)}
    if label_col in idx_map:
        hover += f"Label: %{{customdata[{idx_map[label_col]}]}}<br>"
    if "GDP" in idx_map:
        hover += f"GDP: %{{customdata[{idx_map['GDP']}]:,.2f}}<br>"
    if "Internet Usage" in idx_map:
        hover += f"Internet Usage: %{{customdata[{idx_map['Internet Usage']}]:.1f}}%"
    hover += "<extra></extra>"

    colorscale = [[i / max(len(colors) - 1, 1), c] for i, c in enumerate(colors.values())]

    fig = go.Figure(go.Choropleth(
        locations=df["Country"],
        locationmode="country names",
        z=df[cluster_col].astype(float),
        colorscale=colorscale,
        showscale=False,
        hovertemplate=hover,
        customdata=customdata,
    ))
    fig.update_layout(
        geo=dict(
            bgcolor="#050507",
            landcolor="#111118",
            oceancolor="#08080E",
            showocean=True,
            coastlinecolor="#2A2415",
            showframe=False,
            showlakes=False,
            lakecolor="#08080E",
        ),
        paper_bgcolor="#050507",
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
    )
    return fig


def build_choropleth_mini(df: pd.DataFrame, cluster_col: str, title: str, colors: dict) -> go.Figure:
    colorscale = [[i / max(len(colors) - 1, 1), c] for i, c in enumerate(colors.values())]
    fig = go.Figure(go.Choropleth(
        locations=df["Country"],
        locationmode="country names",
        z=df[cluster_col].astype(float),
        colorscale=colorscale,
        showscale=False,
        hovertemplate="<b>%{location}</b><extra></extra>",
    ))
    fig.update_layout(
        title_text=title,
        title_font=dict(family="Cinzel, serif", color="#FFD966", size=11),
        geo=dict(
            bgcolor="#050507",
            landcolor="#111118",
            oceancolor="#08080E",
            showocean=True,
            coastlinecolor="#2A2415",
            showframe=False,
            showlakes=False,
        ),
        paper_bgcolor="#050507",
        margin=dict(l=0, r=0, t=28, b=0),
        height=220,
    )
    return fig


# ── PCA Scatter ───────────────────────────────────────────────────────────────

def build_pca_scatter(df_merged: pd.DataFrame, cluster_col: str) -> go.Figure:
    pc_cols = [c for c in df_merged.columns if c.startswith("PC")]
    if len(pc_cols) < 2:
        fig = go.Figure()
        _apply_dark(fig, title_text="PCA data unavailable")
        return fig

    pc1, pc2 = pc_cols[0], pc_cols[1]
    fig = go.Figure()

    for cid, color in CLUSTER_COLORS.items():
        mask = df_merged[cluster_col] == cid
        sub  = df_merged[mask]
        if sub.empty:
            continue

        if "Country" in sub.columns:
            hover_text = (
                "<b>" + sub["Country"].astype(str) + "</b><br>"
                + "PC1: " + sub[pc1].round(3).astype(str) + "<br>"
                + "PC2: " + sub[pc2].round(3).astype(str)
            ).tolist()
            hoverinfo = "text"
        else:
            hover_text = None
            hoverinfo  = "skip"

        fig.add_trace(go.Scatter(
            x=sub[pc1],
            y=sub[pc2],
            mode="markers",
            name=f"Cluster {cid} — {CLUSTER_NAMES.get(cid, str(cid))}",
            marker=dict(color=color, size=7, opacity=0.85,
                        line=dict(width=0.5, color="#050507")),
            hovertext=hover_text,
            hoverinfo=hoverinfo,
        ))

    _apply_dark(fig, height=380, title_text="PCA Projection — PC1 × PC2")
    fig.update_xaxes(**_AXIS_DEFAULTS, title_text="PC1")
    fig.update_yaxes(**_AXIS_DEFAULTS, title_text="PC2")
    return fig


# ── Cluster Radar ─────────────────────────────────────────────────────────────

def build_cluster_radar(df: pd.DataFrame, cluster_col: str, cluster_id: int,
                         features: list) -> go.Figure:
    avail = [f for f in features if f in df.columns]
    if not avail:
        fig = go.Figure()
        _apply_dark(fig, title_text="No features selected")
        return fig

    cluster_mean = df[df[cluster_col] == cluster_id][avail].mean()
    global_mean  = df[avail].mean()
    global_std   = df[avail].std().replace(0, 1)

    cluster_z = ((cluster_mean - global_mean) / global_std).values
    global_z  = np.zeros(len(avail))

    z_min  = min(float(cluster_z.min()), -0.5)
    z_max  = max(float(cluster_z.max()),  0.5)
    rng    = z_max - z_min + 1e-9

    def scale(z):
        return (z - z_min) / rng

    cats         = avail + [avail[0]]
    cluster_vals = list(scale(cluster_z)) + [scale(cluster_z)[0]]
    global_vals  = list(scale(global_z))  + [scale(global_z)[0]]

    color      = CLUSTER_COLORS.get(cluster_id, "#D4A017")
    fill_color = _hex_to_rgba(color, 0.15)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=cluster_vals, theta=cats,
        fill="toself",
        name=f"Cluster {cluster_id}",
        line=dict(color=color, width=2),
        fillcolor=fill_color,          # BUG FIXED: was duplicated
    ))
    fig.add_trace(go.Scatterpolar(
        r=global_vals, theta=cats,
        fill="toself",
        name="Global Mean",
        line=dict(color="#5A5040", width=1, dash="dot"),
        fillcolor="rgba(90,80,64,0.06)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0A0A0F",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#18181F",
                            tickcolor="#2A2415", color="#5A5040", showticklabels=False),
            angularaxis=dict(gridcolor="#18181F", linecolor="#2A2415",
                             tickfont=dict(family="JetBrains Mono, monospace",
                                          size=9, color="#B8A87A")),
        ),
        paper_bgcolor="#050507",
        font=dict(family="Inter, sans-serif", color="#B8A87A"),
        legend=dict(bgcolor="#111118", bordercolor="#2A2415", borderwidth=1,
                    font=dict(size=10)),
        title_text=f"Cluster {cluster_id} Profile",
        title_font=dict(family="Cinzel, serif", color="#FFD966", size=13),
        height=370,
        margin=dict(l=40, r=40, t=50, b=20),
        hoverlabel=dict(bgcolor="#18181F", bordercolor="#A67C00",
                        font_color="#F5ECD7", font_family="Inter, sans-serif"),
    )
    return fig


# ── Country Feature Bars ──────────────────────────────────────────────────────

def build_country_feature_bars(country_row: pd.Series, df: pd.DataFrame,
                                 cluster_col: str, features: list) -> go.Figure:
    avail = [f for f in features if f in df.columns and f in country_row.index]
    if not avail:
        fig = go.Figure()
        _apply_dark(fig, title_text="No features available")
        return fig

    cid          = int(country_row[cluster_col])
    cluster_mean = df[df[cluster_col] == cid][avail].mean()
    global_mean  = df[avail].mean()

    # BUG FIXED: pd.Series.get() returns scalar correctly, but we use direct
    # index access with a fallback to 0 to avoid KeyError on missing features
    country_vals = [float(country_row[f]) if f in country_row.index else 0.0 for f in avail]
    cluster_vals = [float(cluster_mean[f]) if f in cluster_mean.index else 0.0 for f in avail]
    global_vals  = [float(global_mean[f])  if f in global_mean.index  else 0.0 for f in avail]

    color       = CLUSTER_COLORS.get(cid, "#D4A017")
    short_feats = [f[:18] + "…" if len(f) > 18 else f for f in avail]
    country_name = str(country_row["Country"]) if "Country" in country_row.index else "Country"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=country_name,
        x=short_feats, y=country_vals,
        marker_color=color,
        opacity=0.9,
    ))
    fig.add_trace(go.Bar(
        name=f"Cluster {cid} Mean",
        x=short_feats, y=cluster_vals,
        marker_color="#5A5040",
        opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        name="Global Mean",
        x=short_feats, y=global_vals,
        mode="markers",
        marker=dict(color="#FFD966", size=8, symbol="diamond"),
    ))
    _apply_dark(fig, height=340, title_text="Feature Comparison (Standardised)")
    fig.update_layout(barmode="group")
    fig.update_xaxes(**_AXIS_DEFAULTS, tickangle=-35)
    fig.update_yaxes(**_AXIS_DEFAULTS, title_text="Std. Value")
    return fig


# ── Cluster size bar for model arena ──────────────────────────────────────────

def build_cluster_size_bar(m: dict) -> go.Figure:
    sizes = m.get("cluster_sizes", {})
    # BUG FIXED: keys may be strings from JSON; cast to int safely
    labels = [f"C{k}" for k in sizes]
    values = list(sizes.values())
    colors = [CLUSTER_COLORS.get(int(k), "#5A5040") for k in sizes]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=colors),
        text=values, textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=9, color="#B8A87A"),
    ))
    _apply_dark(fig, height=180, title_text=m["model"])
    fig.update_layout(margin=dict(l=4, r=4, t=30, b=4))
    return fig


# ── Feature Importance Heatmap ────────────────────────────────────────────────

def build_importance_heatmap(df: pd.DataFrame, cluster_col: str,
                               features: list) -> go.Figure:
    avail = [f for f in features if f in df.columns]
    if not avail:
        fig = go.Figure()
        _apply_dark(fig, title_text="No features selected")
        return fig

    clusters     = sorted(df[cluster_col].unique())
    global_mean  = df[avail].mean()
    global_std   = df[avail].std().replace(0, 1)

    z        = []
    y_labels = []
    for cid in clusters:
        cluster_mean = df[df[cluster_col] == cid][avail].mean()
        z_row        = ((cluster_mean - global_mean) / global_std).values
        z.append(z_row.tolist())
        y_labels.append(f"Cluster {cid}")

    short_feats = [f[:14] + "…" if len(f) > 14 else f for f in avail]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=short_feats,
        y=y_labels,
        colorscale=GOLD_DIVERGING,
        zmid=0,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(family="JetBrains Mono, monospace", size=9),
        hovertemplate="Feature: %{x}<br>Cluster: %{y}<br>Z-Score: %{z:.3f}<extra></extra>",
    ))
    _apply_dark(fig, height=260, title_text="Feature Importance (Z-Score vs Global Mean)")
    fig.update_xaxes(**_AXIS_DEFAULTS, tickangle=-35)
    fig.update_layout(margin=dict(l=80, r=8, t=50, b=80))
    return fig


# ── Correlation Matrix ────────────────────────────────────────────────────────

def build_correlation_matrix(df: pd.DataFrame, features: list) -> go.Figure:
    avail = [f for f in features if f in df.columns]
    if len(avail) < 2:
        fig = go.Figure()
        _apply_dark(fig, title_text="Select at least 2 features")
        return fig

    corr  = df[avail].corr()
    short = [f[:12] + "…" if len(f) > 12 else f for f in avail]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=short,
        y=short,
        colorscale=GOLD_DIVERGING,
        zmid=0,
        zmin=-1, zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(family="JetBrains Mono, monospace", size=9),
        hovertemplate="%{x} × %{y}: %{z:.3f}<extra></extra>",
    ))
    _apply_dark(fig, height=420, title_text="Feature Correlation Matrix")
    fig.update_xaxes(**_AXIS_DEFAULTS, tickangle=-35)
    fig.update_layout(margin=dict(l=80, r=8, t=50, b=80))
    return fig


# ── Violin plots ──────────────────────────────────────────────────────────────

def build_violin(df: pd.DataFrame, cluster_col: str, feature: str) -> go.Figure:
    if feature not in df.columns:
        fig = go.Figure()
        _apply_dark(fig, title_text=f"Feature '{feature}' not found")
        return fig

    fig = go.Figure()
    for cid, color in CLUSTER_COLORS.items():
        mask = df[cluster_col] == cid
        sub  = df[mask][feature].dropna()
        if sub.empty:
            continue
        fig.add_trace(go.Violin(
            y=sub,
            name=f"C{cid} — {CLUSTER_NAMES.get(cid, str(cid))}",
            box_visible=True,
            meanline_visible=True,
            line_color=color,
            fillcolor=_hex_to_rgba(color, 0.2),
            points="outliers",
            marker=dict(color=color, size=3),
        ))
    _apply_dark(fig, height=380, title_text=f"Distribution: {feature}")
    fig.update_yaxes(**_AXIS_DEFAULTS, title_text="Value")
    return fig