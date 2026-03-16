"""
============================================================
Bankruptcy Risk Intelligence System — Streamlit Dashboard
============================================================

Pages
-----
1. Risk Predictor     — animated gauge + probability bar
2. Model Performance  — metrics, confusion matrix, class report
3. EDA Explorer       — dataset distributions, correlations
4. MLflow Tracker     — experiment run comparison table

Run
---
    streamlit run app.py

From project root (so config paths resolve correctly).
"""

import os
import sys
import json
import sqlite3
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# ══════════════════════════════════════════════════════════════════════════════
# THEME CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

NAVY      = "#0A1628"
NAVY_MID  = "#0F2040"
NAVY_CARD = "#132238"
NAVY_BORDER = "#1C3355"
GOLD      = "#C9A84C"
GOLD_LIGHT = "#E8C76A"
GOLD_DIM  = "#8A6F32"
WHITE     = "#F0F4FF"
MUTED     = "#8899BB"
RED_RISK  = "#E84545"
GREEN_SAFE = "#2ECC71"
AMBER_MED  = "#F39C12"

FEATURES = [
    "industrial_risk",
    "management_risk",
    "financial_flexibility",
    "credibility",
    "competitiveness",
    "operating_risk",
]

FEATURE_LABELS = {
    "industrial_risk":       "Industrial Risk",
    "management_risk":       "Management Risk",
    "financial_flexibility": "Financial Flexibility",
    "credibility":           "Credibility",
    "competitiveness":       "Competitiveness",
    "operating_risk":        "Operating Risk",
}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Bankruptcy Risk Intelligence",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── root ── */
:root {{
    --navy: {NAVY};
    --navy-mid: {NAVY_MID};
    --navy-card: {NAVY_CARD};
    --navy-border: {NAVY_BORDER};
    --gold: {GOLD};
    --gold-light: {GOLD_LIGHT};
    --gold-dim: {GOLD_DIM};
    --white: {WHITE};
    --muted: {MUTED};
}}

/* ── global body ── */
.stApp {{
    background-color: {NAVY};
    font-family: 'DM Sans', sans-serif;
    color: {WHITE};
}}

/* ── sidebar ── */
[data-testid="stSidebar"] {{
    background: {NAVY_MID} !important;
    border-right: 1px solid {NAVY_BORDER} !important;
}}
[data-testid="stSidebar"] .stRadio label {{
    color: {WHITE} !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.92rem;
    padding: 6px 0;
}}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {{
    color: {WHITE} !important;
}}

/* ── headings ── */
h1, h2, h3 {{
    font-family: 'Playfair Display', serif !important;
    color: {WHITE} !important;
}}

/* ── metric cards ── */
[data-testid="stMetric"] {{
    background: {NAVY_CARD};
    border: 1px solid {NAVY_BORDER};
    border-top: 2px solid {GOLD};
    border-radius: 8px;
    padding: 16px 20px !important;
}}
[data-testid="stMetricLabel"] {{
    color: {MUTED} !important;
    font-size: 0.78rem !important;
    font-family: 'DM Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
[data-testid="stMetricValue"] {{
    color: {GOLD} !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.9rem !important;
}}

/* ── sliders ── */
[data-testid="stSlider"] .st-emotion-cache-1dx1gwv {{
    background: {GOLD} !important;
}}
.stSlider [data-baseweb="slider"] [data-testid="stTickBar"] {{
    color: {MUTED};
}}

/* ── dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {NAVY_BORDER};
    border-radius: 6px;
}}

/* ── buttons ── */
.stButton > button {{
    background: linear-gradient(135deg, {GOLD_DIM}, {GOLD});
    color: {NAVY};
    border: none;
    border-radius: 6px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 10px 32px;
    letter-spacing: 0.05em;
    transition: all 0.2s;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, {GOLD}, {GOLD_LIGHT});
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(201, 168, 76, 0.35);
}}

/* ── divider ── */
hr {{
    border-color: {NAVY_BORDER} !important;
}}

/* ── code / mono ── */
code {{
    background: {NAVY_MID};
    color: {GOLD_LIGHT};
    font-family: 'DM Mono', monospace;
    border-radius: 3px;
    padding: 1px 5px;
}}

/* ── tabs ── */
[data-testid="stTabs"] [role="tab"] {{
    font-family: 'DM Sans', sans-serif;
    color: {MUTED};
    border-bottom: 2px solid transparent;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: {GOLD} !important;
    border-bottom: 2px solid {GOLD} !important;
}}

/* ── selectbox ── */
[data-baseweb="select"] {{
    background: {NAVY_CARD} !important;
    border-color: {NAVY_BORDER} !important;
}}

/* ── progress bar ── */
[data-testid="stProgress"] > div > div {{
    background: linear-gradient(90deg, {GOLD_DIM}, {GOLD}) !important;
}}

/* ── scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {NAVY}; }}
::-webkit-scrollbar-thumb {{ background: {NAVY_BORDER}; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS  (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_dataset():
    train = pd.read_csv(os.path.join(ROOT, "artifacts/data_ingestion/train.csv"))
    test  = pd.read_csv(os.path.join(ROOT, "artifacts/data_ingestion/test.csv"))
    return pd.concat([train, test], ignore_index=True)

@st.cache_data
def load_trainer_metrics():
    path = os.path.join(ROOT, "artifacts/model_trainer/metrics.json")
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_eval_metrics():
    path = os.path.join(ROOT, "artifacts/model_evaluation/evaluation_metrics.json")
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_confusion_matrix():
    path = os.path.join(ROOT, "artifacts/model_evaluation/confusion_matrix.json")
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_classification_report():
    path = os.path.join(ROOT, "artifacts/model_evaluation/evaluation_report.txt")
    with open(path) as f:
        return f.read()

@st.cache_data
def load_mlflow_runs():
    db_path = os.path.join(ROOT, "mlflow.db")
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    runs = conn.execute(
        "SELECT run_uuid, status, start_time FROM runs ORDER BY start_time DESC"
    ).fetchall()
    params_rows  = conn.execute("SELECT run_uuid, key, value FROM params").fetchall()
    metrics_rows = conn.execute(
        "SELECT run_uuid, key, value FROM metrics"
    ).fetchall()
    conn.close()

    params_df  = pd.DataFrame(params_rows,  columns=["run_uuid", "key", "value"])
    metrics_df = pd.DataFrame(metrics_rows, columns=["run_uuid", "key", "value"])
    runs_df    = pd.DataFrame(
        [(r["run_uuid"], r["status"], r["start_time"]) for r in runs],
        columns=["run_uuid", "status", "start_time"],
    )

    params_pivot  = params_df.pivot_table(
        index="run_uuid", columns="key", values="value", aggfunc="first"
    ).reset_index()
    metrics_pivot = metrics_df.pivot_table(
        index="run_uuid", columns="key", values="value", aggfunc="first"
    ).reset_index()

    merged = runs_df.merge(params_pivot,  on="run_uuid", how="left")
    merged = merged.merge(metrics_pivot, on="run_uuid", how="left")
    merged["start_time"] = pd.to_datetime(merged["start_time"], unit="ms")
    merged["run_short"]  = merged["run_uuid"].str[:8].str.upper()
    return merged

@st.cache_resource
def load_model():
    import joblib
    path = os.path.join(ROOT, "artifacts/model_trainer/model.pkl")
    return joblib.load(path)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=NAVY_CARD,
    font_family="DM Sans",
    font_color=WHITE,
    title_font_family="Playfair Display",
    title_font_color=WHITE,
    colorway=[GOLD, "#4A9EFF", GREEN_SAFE, RED_RISK, AMBER_MED, "#A855F7"],
    xaxis=dict(gridcolor=NAVY_BORDER, zerolinecolor=NAVY_BORDER, color=MUTED),
    yaxis=dict(gridcolor=NAVY_BORDER, zerolinecolor=NAVY_BORDER, color=MUTED),
    margin=dict(l=40, r=20, t=50, b=40),
)

def apply_theme(fig, title=""):
    fig.update_layout(**PLOTLY_LAYOUT)
    if title:
        fig.update_layout(title=title, title_font_size=15)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"""
    <div style="padding:20px 0 28px 0;">
        <div style="font-family:'Playfair Display',serif;font-size:1.3rem;
                    color:{GOLD};letter-spacing:0.04em;line-height:1.3;">
            ⚖ Bankruptcy Risk<br>
            <span style="color:{MUTED};font-size:0.85rem;
                         font-family:'DM Sans',sans-serif;font-style:italic;">
            Intelligence System
            </span>
        </div>
        <div style="height:1px;background:linear-gradient(90deg,{GOLD_DIM},transparent);
                    margin-top:16px;"></div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["⚡  Risk Predictor",
         "📊  Model Performance",
         "🔍  EDA Explorer",
         "🧪  MLflow Tracker"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Model status badge
    try:
        m = load_trainer_metrics()
        model_name = m.get("best_model", "unknown").replace("_", " ").title()
        cv_auc     = m.get("cv_auc_mean", 0)
        st.markdown(f"""
        <div style="background:{NAVY_CARD};border:1px solid {NAVY_BORDER};
                    border-left:3px solid {GOLD};border-radius:6px;
                    padding:14px 16px;font-size:0.82rem;">
            <div style="color:{MUTED};text-transform:uppercase;
                        letter-spacing:0.08em;font-family:'DM Mono',monospace;
                        font-size:0.72rem;margin-bottom:8px;">Active Model</div>
            <div style="color:{GOLD};font-weight:600;margin-bottom:4px;">
                {model_name}
            </div>
            <div style="color:{MUTED};">CV AUC&nbsp;
                <span style="color:{GREEN_SAFE};font-family:'DM Mono',monospace;">
                    {cv_auc:.4f}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.warning("Run training pipeline first.")

    st.markdown(f"""
    <div style="position:fixed;bottom:20px;left:0;width:220px;
                padding:0 20px;font-size:0.72rem;color:{GOLD_DIM};
                font-family:'DM Mono',monospace;">
        v1.0.0 · MLOps Production
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
#
# Three bugs fixed vs original:
#
# Bug 1 — Predict on load (main issue in screenshot)
#   Streamlit reruns the entire script on every slider interaction.
#   The old code ran pipeline.predict() unconditionally, so the gauge
#   updated the moment any slider moved — before the button was pressed.
#   Fix: store result in st.session_state["pred_result"] only when the
#   button is clicked. The gauge renders from session_state, so it stays
#   blank until the user explicitly triggers a prediction.
#
# Bug 2 — Wrong slider bar-color semantics
#   financial_flexibility, credibility, competitiveness are PROTECTIVE:
#   higher value = SAFER company. The old code coloured high values red
#   for ALL features, making a company with high credibility look risky.
#   Fix: invert the colour logic for protective features so
#   high = green (good) and low = red (bad).
#
# Bug 3 — Misleading hint text
#   Old hints said "0 = illiquid" for financial_flexibility but still
#   described it as a "risk factor", confusing the user.
#   Fix: each feature is clearly tagged [RISK FACTOR] or [PROTECTIVE]
#   with a coloured badge so users understand the direction.

if "Risk Predictor" in page:

    # ── Initialise session state ──────────────────────────────────────────────
    # pred_result is only written when the button is clicked.
    # It is never written at page-load time, so no ghost predictions appear.
    if "pred_result" not in st.session_state:
        st.session_state["pred_result"] = None
    if "pred_inputs" not in st.session_state:
        st.session_state["pred_inputs"] = None

    st.markdown(f"""
    <h1 style="font-size:2rem;margin-bottom:4px;">
        Company Bankruptcy Risk Predictor
    </h1>
    <p style="color:{MUTED};font-size:0.95rem;margin-bottom:8px;">
        Set the six company indicators, then click <b style="color:{GOLD};">
        Analyse Risk</b> to run the model.
    </p>
    """, unsafe_allow_html=True)

    # ── How to read this tool ─────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{NAVY_CARD};border:1px solid {NAVY_BORDER};
                border-left:3px solid {GOLD};border-radius:8px;
                padding:16px 20px;margin-bottom:24px;font-size:0.83rem;line-height:1.7;">
        <div style="color:{GOLD};font-family:'DM Mono',monospace;font-size:0.72rem;
                    text-transform:uppercase;letter-spacing:0.09em;margin-bottom:10px;">
            How to read the indicators
        </div>
        <div style="display:flex;gap:28px;flex-wrap:wrap;">
            <div>
                <span style="background:#2E0D0D;border:1px solid {RED_RISK};
                             color:{RED_RISK};padding:2px 8px;border-radius:3px;
                             font-size:0.70rem;font-family:'DM Mono',monospace;">
                    ▲ RISK FACTOR
                </span>
                <span style="color:{MUTED};margin-left:8px;">
                    <b style="color:{WHITE};">Higher = more dangerous.</b>
                    Set to <b style="color:{GREEN_SAFE};">0</b> = safe.
                </span>
            </div>
            <div>
                <span style="background:#0D2E1A;border:1px solid {GREEN_SAFE};
                             color:{GREEN_SAFE};padding:2px 8px;border-radius:3px;
                             font-size:0.70rem;font-family:'DM Mono',monospace;">
                    ▼ PROTECTIVE
                </span>
                <span style="color:{MUTED};margin-left:8px;">
                    <b style="color:{WHITE};">Higher = safer.</b>
                    Set to <b style="color:{RED_RISK};">0</b> = company has
                    <b style="color:{RED_RISK};">none</b> of this protection.
                </span>
            </div>
        </div>
        <div style="color:{MUTED};margin-top:10px;font-size:0.78rem;">
            ⚠ <b style="color:{AMBER_MED};">All zeros = maximum risk</b> —
            zero financial flexibility + zero credibility + zero competitiveness
            = a company with no protective factors at all.<br>
            ✓ <b style="color:{GREEN_SAFE};">All ones = very safe</b> —
            protective factors (credibility 1.0 + competitiveness 1.0) dominate.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature metadata ──────────────────────────────────────────────────────
    # is_risk=True  → higher value = more bankrupt
    # is_risk=False → higher value = safer (protective factor)
    SLIDER_META = {
        "industrial_risk": {
            "icon": "🏭", "is_risk": True,
            "hint": "Industry-level exposure to economic shocks",
            "low": "stable sector", "high": "volatile sector",
        },
        "management_risk": {
            "icon": "👔", "is_risk": True,
            "hint": "Quality and stability of company leadership",
            "low": "strong governance", "high": "poor governance",
        },
        "financial_flexibility": {
            "icon": "💰", "is_risk": False,
            "hint": "Ability to raise capital and manage liquidity",
            "low": "illiquid / constrained", "high": "highly flexible",
        },
        "credibility": {
            "icon": "🏦", "is_risk": False,
            "hint": "Market and lender confidence in the company",
            "low": "poor credit standing", "high": "excellent reputation",
        },
        "competitiveness": {
            "icon": "📈", "is_risk": False,
            "hint": "Strength of market position relative to peers",
            "low": "weak market position", "high": "market leader",
        },
        "operating_risk": {
            "icon": "⚙", "is_risk": True,
            "hint": "Operational volatility and cost structure risk",
            "low": "stable operations", "high": "high exposure",
        },
    }

    # ── Sliders ───────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2, gap="large")
    inputs = {}

    for i, feat in enumerate(FEATURES):
        col  = col_a if i < 3 else col_b
        meta = SLIDER_META[feat]
        badge_bg    = "#2E0D0D" if meta["is_risk"] else "#0D2E1A"
        badge_color = RED_RISK  if meta["is_risk"] else GREEN_SAFE
        badge_text  = "▲ RISK" if meta["is_risk"] else "▼ PROTECTIVE"

        with col:
            st.markdown(f"""
            <div style="margin-bottom:3px;display:flex;
                        align-items:center;gap:8px;">
                <span style="font-size:0.9rem;font-weight:600;color:{WHITE};">
                    {meta['icon']} {FEATURE_LABELS[feat]}
                </span>
                <span style="background:{badge_bg};border:1px solid {badge_color};
                             color:{badge_color};font-size:0.67rem;
                             padding:1px 7px;border-radius:3px;
                             font-family:'DM Mono',monospace;">
                    {badge_text}
                </span>
            </div>
            <div style="font-size:0.72rem;color:{MUTED};margin-bottom:2px;">
                {meta['hint']}
            </div>
            <div style="font-size:0.70rem;color:{MUTED};
                        margin-bottom:6px;font-family:'DM Mono',monospace;">
                0.0 = {meta['low']} &nbsp;·&nbsp; 1.0 = {meta['high']}
            </div>
            """, unsafe_allow_html=True)

            val = st.select_slider(
                FEATURE_LABELS[feat],
                options=[0.0, 0.5, 1.0],
                value=0.5,
                key=f"slider_{feat}",
                label_visibility="collapsed",
            )
            inputs[feat] = val

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── Analyse button ────────────────────────────────────────────────────────
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        clicked = st.button("⚡  Analyse Risk", use_container_width=True)

    # ── Write to session_state ONLY on button click ───────────────────────────
    # This is the fix for "predict on load". The prediction block below reads
    # exclusively from st.session_state["pred_result"], which is None until
    # the button is pressed. Moving a slider does not set pred_result.
    if clicked:
        try:
            from bankruptcy.pipeline.prediction_pipeline import PredictionPipeline
            _pipeline = PredictionPipeline(
                model_path=os.path.join(ROOT, "artifacts/model_trainer/model.pkl")
            )
            st.session_state["pred_result"] = _pipeline.predict(inputs)
            st.session_state["pred_inputs"]  = dict(inputs)
        except FileNotFoundError:
            st.error("Model not found — run `python main.py` first.")
        except Exception as ex:
            st.error(f"Prediction error: {ex}")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Result panel — only shown after button click ──────────────────────────
    result = st.session_state.get("pred_result")

    if result is None:
        # Awaiting first prediction — show placeholder
        st.markdown(f"""
        <div style="background:{NAVY_CARD};border:1px solid {NAVY_BORDER};
                    border-radius:10px;padding:40px;text-align:center;">
            <div style="font-size:2.5rem;margin-bottom:12px;">⚖</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.2rem;
                        color:{MUTED};margin-bottom:8px;">
                Awaiting Analysis
            </div>
            <div style="font-size:0.85rem;color:{GOLD_DIM};">
                Set the indicators above and click
                <b style="color:{GOLD};">Analyse Risk</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        saved_inputs = st.session_state.get("pred_inputs", inputs)
        label     = result["prediction"]
        bk_prob   = result["bankruptcy_probability"]
        safe_prob = 1.0 - bk_prob

        is_bankrupt = label == "bankruptcy"
        risk_color  = RED_RISK  if is_bankrupt else GREEN_SAFE
        risk_text   = "HIGH RISK" if is_bankrupt else "LOW RISK"

        # ── Gauge ──────────────────────────────────────────────────────────
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(bk_prob * 100, 1),
            delta={
                "reference": 50,
                "increasing": {"color": RED_RISK},
                "decreasing": {"color": GREEN_SAFE},
            },
            number={
                "suffix": "%",
                "font": {"size": 44, "family": "Playfair Display",
                         "color": WHITE},
            },
            title={
                "text": (
                    f"Bankruptcy Probability<br>"
                    f"<span style='font-size:0.85em;color:{risk_color}'>"
                    f"{risk_text}</span>"
                ),
                "font": {"size": 16, "family": "DM Sans", "color": WHITE},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": MUTED,
                    "tickfont":  {"color": MUTED, "size": 11},
                },
                "bar":        {"color": risk_color, "thickness": 0.28},
                "bgcolor":    NAVY_CARD,
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  35],  "color": "#0D2E1A"},
                    {"range": [35, 65],  "color": "#2A2410"},
                    {"range": [65, 100], "color": "#2E0D0D"},
                ],
                "threshold": {
                    "line":      {"color": GOLD, "width": 3},
                    "thickness": 0.85,
                    "value":     50,
                },
            },
        ))
        gauge_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=20, r=20, t=40, b=10),
        )

        res_left, res_right = st.columns([1, 1], gap="large")

        with res_left:
            st.plotly_chart(gauge_fig, use_container_width=True,
                            config={"displayModeBar": False})

        with res_right:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            # Verdict banner
            banner_bg = "#2E0D0D" if is_bankrupt else "#0D2E1A"
            st.markdown(f"""
            <div style="background:{banner_bg};border:1px solid {risk_color};
                        border-radius:10px;padding:22px 26px;margin-bottom:18px;">
                <div style="font-family:'Playfair Display',serif;
                            font-size:1.5rem;color:{risk_color};
                            margin-bottom:6px;">
                    {'⚠ Bankruptcy Risk Detected' if is_bankrupt
                     else '✓ Company Appears Solvent'}
                </div>
                <div style="color:{MUTED};font-size:0.86rem;line-height:1.6;">
                    {'Elevated financial distress signals detected. Immediate review recommended.'
                     if is_bankrupt else
                     'Healthy financial indicators across all evaluated dimensions.'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Dual probability bars
            for prob, bar_label, color in [
                (bk_prob,   "Bankruptcy Risk", RED_RISK),
                (safe_prob, "Solvency Score",  GREEN_SAFE),
            ]:
                pct = round(prob * 100, 1)
                st.markdown(f"""
                <div style="margin-bottom:14px;">
                    <div style="display:flex;justify-content:space-between;
                                margin-bottom:5px;font-size:0.84rem;">
                        <span style="color:{WHITE};">{bar_label}</span>
                        <span style="color:{color};
                                     font-family:'DM Mono',monospace;
                                     font-weight:600;">{pct}%</span>
                    </div>
                    <div style="background:{NAVY_BORDER};border-radius:4px;
                                height:10px;">
                        <div style="background:{color};width:{pct}%;
                                    height:100%;border-radius:4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Input profile bars — colour direction aware
            st.markdown(f"""
            <div style="margin-top:16px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.70rem;
                            text-transform:uppercase;letter-spacing:0.09em;
                            color:{MUTED};margin-bottom:10px;">
                    Input Profile
                </div>
            """, unsafe_allow_html=True)

            for feat in FEATURES:
                v       = saved_inputs.get(feat, 0.5)
                v_pct   = int(v * 100)
                meta    = SLIDER_META[feat]
                # Correct colour direction:
                # RISK factor:      high value → red, low → green
                # PROTECTIVE factor: high value → green, low → red
                if meta["is_risk"]:
                    bar_c = RED_RISK if v >= 0.8 else (AMBER_MED if v >= 0.5 else GREEN_SAFE)
                else:
                    bar_c = GREEN_SAFE if v >= 0.8 else (AMBER_MED if v >= 0.5 else RED_RISK)

                badge_c = RED_RISK if meta["is_risk"] else GREEN_SAFE
                tag     = "▲" if meta["is_risk"] else "▼"

                st.markdown(f"""
                <div style="display:flex;align-items:center;
                            gap:8px;margin-bottom:7px;font-size:0.79rem;">
                    <span style="color:{badge_c};width:14px;
                                 font-size:0.65rem;">{tag}</span>
                    <span style="color:{MUTED};width:148px;
                                 white-space:nowrap;">{FEATURE_LABELS[feat]}</span>
                    <div style="flex:1;background:{NAVY_BORDER};
                                border-radius:3px;height:7px;">
                        <div style="background:{bar_c};
                                    width:{max(v_pct, 3)}%;height:100%;
                                    border-radius:3px;"></div>
                    </div>
                    <span style="color:{WHITE};width:28px;text-align:right;
                                 font-family:'DM Mono',monospace;">
                        {v:.1f}
                    </span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif "Model Performance" in page:

    st.markdown(f"""
    <h1 style="font-size:2rem;margin-bottom:4px;">Model Performance Dashboard</h1>
    <p style="color:{MUTED};font-size:0.95rem;margin-bottom:32px;">
        Evaluation metrics, confusion matrix and per-class classification report
        from the held-out test set (50 samples).
    </p>
    """, unsafe_allow_html=True)

    try:
        ev  = load_eval_metrics()
        cm  = load_confusion_matrix()
        rep = load_classification_report()
        trm = load_trainer_metrics()

        # ── Top metric strip ──────────────────────────────────────────────────
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        strip = [
            (c1, "Accuracy",       f"{ev['accuracy']*100:.1f}%"),
            (c2, "ROC-AUC",        f"{ev['roc_auc']:.4f}"),
            (c3, "F1 Macro",       f"{ev['f1_macro']:.4f}"),
            (c4, "MCC",            f"{ev['matthews_corrcoef']:.4f}"),
            (c5, "False Neg Rate", f"{ev['false_negative_rate']*100:.1f}%"),
            (c6, "False Pos Rate", f"{ev['false_positive_rate']*100:.1f}%"),
        ]
        for col, label, val in strip:
            with col:
                st.metric(label, val)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        left, right = st.columns([1, 1], gap="large")

        # ── Confusion matrix ──────────────────────────────────────────────────
        with left:
            tn = cm["true_negative"]
            fp = cm["false_positive"]
            fn = cm["false_negative"]
            tp = cm["true_positive"]

            z    = [[tn, fp], [fn, tp]]
            text = [[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]]

            cm_fig = go.Figure(go.Heatmap(
                z=z,
                x=["Predicted\nNon-Bankruptcy", "Predicted\nBankruptcy"],
                y=["Actual\nNon-Bankruptcy", "Actual\nBankruptcy"],
                text=text,
                texttemplate="%{text}",
                textfont={"size": 18, "family": "Playfair Display", "color": WHITE},
                colorscale=[[0, NAVY_MID], [0.5, GOLD_DIM], [1.0, GOLD]],
                showscale=False,
                hoverongaps=False,
            ))
            cm_fig = apply_theme(cm_fig, "Confusion Matrix")
            cm_fig.update_layout(height=320)
            cm_fig.update_xaxes(side="bottom")
            st.plotly_chart(cm_fig, use_container_width=True, config={"displayModeBar": False})

        # ── Model comparison radar ────────────────────────────────────────────
        with right:
            all_models = trm.get("all_models", {})
            if len(all_models) >= 2:
                cats   = ["CV AUC", "Test AUC", "Accuracy", "Stability"]
                traces = []
                for name, vals in all_models.items():
                    stability = max(0, 1 - vals.get("cv_std", 0) * 10)
                    r_vals = [
                        vals.get("cv_auc",   0),
                        vals.get("test_auc", 0),
                        vals.get("test_acc", 0),
                        stability,
                    ]
                    traces.append(go.Scatterpolar(
                        r=r_vals + [r_vals[0]],
                        theta=cats + [cats[0]],
                        name=name.replace("_", " ").title(),
                        fill="toself",
                        fillcolor="rgba(201,168,76,0.12)",
                        line_color=GOLD,
                    ))

                rad_fig = go.Figure(traces)
                rad_fig.update_layout(
                    **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "xaxis" and k != "yaxis"},
                    polar=dict(
                        bgcolor=NAVY_CARD,
                        radialaxis=dict(
                            visible=True, range=[0.95, 1.0],
                            gridcolor=NAVY_BORDER, color=MUTED,
                            tickfont={"size": 9},
                        ),
                        angularaxis=dict(
                            gridcolor=NAVY_BORDER, color=WHITE,
                            tickfont={"size": 11},
                        ),
                    ),
                    showlegend=True,
                    legend=dict(
                        font=dict(color=MUTED, size=10),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    height=320,
                    title="Model Comparison Radar",
                )
                st.plotly_chart(rad_fig, use_container_width=True, config={"displayModeBar": False})
            else:
                # Single model — show metric bars
                metrics_fig = go.Figure()
                metric_vals = {
                    "Accuracy": ev["accuracy"],
                    "ROC-AUC": ev["roc_auc"],
                    "F1 Macro": ev["f1_macro"],
                    "Precision": ev["precision_macro"],
                    "Recall": ev["recall_macro"],
                    "MCC": ev["matthews_corrcoef"],
                    "Cohen κ": ev["cohen_kappa"],
                    "Balanced Acc": ev["balanced_accuracy"],
                }
                metrics_fig.add_trace(go.Bar(
                    x=list(metric_vals.keys()),
                    y=list(metric_vals.values()),
                    marker_color=GOLD,
                    marker_line_color=GOLD_LIGHT,
                    marker_line_width=1,
                ))
                metrics_fig = apply_theme(metrics_fig, "All Evaluation Metrics")
                metrics_fig.update_layout(height=320, yaxis_range=[0.9, 1.01])
                st.plotly_chart(metrics_fig, use_container_width=True, config={"displayModeBar": False})

        # ── Classification report ─────────────────────────────────────────────
        st.markdown(f"""
        <div style="margin-top:8px;">
            <div style="font-family:'DM Mono',monospace;font-size:0.72rem;
                        text-transform:uppercase;letter-spacing:0.09em;
                        color:{MUTED};margin-bottom:10px;">
                Classification Report
            </div>
            <pre style="background:{NAVY_CARD};border:1px solid {NAVY_BORDER};
                        border-radius:8px;padding:20px 24px;
                        font-family:'DM Mono',monospace;font-size:0.85rem;
                        color:{WHITE};line-height:1.7;overflow-x:auto;">
{rep}</pre>
        </div>
        """, unsafe_allow_html=True)

        # ── Cross-validation strip ────────────────────────────────────────────
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        cv_col1, cv_col2, cv_col3, cv_col4 = st.columns(4)
        with cv_col1:
            st.metric("Best Model",
                      trm.get("best_model","—").replace("_"," ").title())
        with cv_col2:
            st.metric("CV AUC Mean", f"{trm.get('cv_auc_mean',0):.6f}")
        with cv_col3:
            st.metric("CV AUC Std (fold stability)", f"{trm.get('cv_auc_std',0):.6f}")
        with cv_col4:
            st.metric("Test AUC", f"{trm.get('test_auc',0):.4f}")

    except FileNotFoundError:
        st.info("Artifacts not found — run `python main.py` to train the pipeline first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EDA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

elif "EDA Explorer" in page:

    st.markdown(f"""
    <h1 style="font-size:2rem;margin-bottom:4px;">Exploratory Data Analysis</h1>
    <p style="color:{MUTED};font-size:0.95rem;margin-bottom:32px;">
        250 companies · 6 risk indicators · binary bankruptcy label
    </p>
    """, unsafe_allow_html=True)

    try:
        df = load_dataset()
        feat_df = df[FEATURES]

        # ── Dataset overview strip ────────────────────────────────────────────
        ov1, ov2, ov3, ov4 = st.columns(4)
        class_counts = df["class"].value_counts()
        with ov1: st.metric("Total Samples",   len(df))
        with ov2: st.metric("Features",        len(FEATURES))
        with ov3: st.metric("Bankruptcies",    class_counts.get("bankruptcy", 0))
        with ov4: st.metric("Non-Bankruptcies",class_counts.get("non-bankruptcy", 0))

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs([
            "  Distribution  ", "  Class vs Features  ",
            "  Correlation  ", "  Raw Data  "
        ])

        # ── Tab 1: Feature distributions ─────────────────────────────────────
        with tab1:
            dist_fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[FEATURE_LABELS[f] for f in FEATURES],
                vertical_spacing=0.18,
                horizontal_spacing=0.08,
            )
            colors = [GOLD, "#4A9EFF", GREEN_SAFE, RED_RISK, AMBER_MED, "#A855F7"]
            for idx, feat in enumerate(FEATURES):
                r, c = divmod(idx, 3)
                vc = df[feat].value_counts().sort_index()
                dist_fig.add_trace(
                    go.Bar(
                        x=vc.index.astype(str),
                        y=vc.values,
                        marker_color=colors[idx],
                        marker_opacity=0.85,
                        name=FEATURE_LABELS[feat],
                        showlegend=False,
                    ),
                    row=r + 1, col=c + 1,
                )

            dist_fig.update_layout(
                **{k: v for k, v in PLOTLY_LAYOUT.items()
                   if k not in ("xaxis", "yaxis")},
                height=440,
                title="Feature Value Distributions (0 = Low, 0.5 = Med, 1 = High)",
            )
            for i in range(1, 7):
                dist_fig.update_xaxes(
                    gridcolor=NAVY_BORDER, color=MUTED, row=(i-1)//3+1, col=(i-1)%3+1
                )
                dist_fig.update_yaxes(
                    gridcolor=NAVY_BORDER, color=MUTED, row=(i-1)//3+1, col=(i-1)%3+1
                )
            st.plotly_chart(dist_fig, use_container_width=True,
                            config={"displayModeBar": False})

            # Class balance donut
            _, donut_col, _ = st.columns([1, 1, 1])
            with donut_col:
                donut_fig = go.Figure(go.Pie(
                    labels=class_counts.index.tolist(),
                    values=class_counts.values.tolist(),
                    hole=0.58,
                    marker_colors=[RED_RISK, GREEN_SAFE],
                    textinfo="label+percent",
                    textfont=dict(family="DM Sans", color=WHITE, size=13),
                ))
                donut_fig.update_layout(
                    **{k: v for k, v in PLOTLY_LAYOUT.items()
                       if k not in ("xaxis","yaxis")},
                    title="Class Balance",
                    height=300,
                    showlegend=False,
                    annotations=[dict(
                        text=f"<b>n={len(df)}</b>",
                        x=0.5, y=0.5, font_size=16,
                        font_color=WHITE, showarrow=False,
                    )],
                )
                st.plotly_chart(donut_fig, use_container_width=True,
                                config={"displayModeBar": False})

        # ── Tab 2: Class vs features ──────────────────────────────────────────
        with tab2:
            box_fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[FEATURE_LABELS[f] for f in FEATURES],
                vertical_spacing=0.18,
                horizontal_spacing=0.08,
            )
            for idx, feat in enumerate(FEATURES):
                r, c = divmod(idx, 3)
                for cls, color in [("bankruptcy", RED_RISK), ("non-bankruptcy", GREEN_SAFE)]:
                    vals = df[df["class"] == cls][feat]
                    box_fig.add_trace(
                        go.Violin(
                            y=vals,
                            name=cls,
                            box_visible=True,
                            meanline_visible=True,
                            fillcolor=color,
                            opacity=0.55,
                            line_color=color,
                            showlegend=(idx == 0),
                        ),
                        row=r + 1, col=c + 1,
                    )

            box_fig.update_layout(
                **{k: v for k, v in PLOTLY_LAYOUT.items()
                   if k not in ("xaxis","yaxis")},
                height=460,
                title="Feature Distribution by Class",
                violingap=0.15,
                violingroupgap=0.1,
                legend=dict(
                    font=dict(color=MUTED, size=11),
                    bgcolor="rgba(0,0,0,0)",
                    orientation="h", y=1.05,
                ),
            )
            for i in range(1, 7):
                box_fig.update_xaxes(
                    gridcolor=NAVY_BORDER, color=MUTED, row=(i-1)//3+1, col=(i-1)%3+1
                )
                box_fig.update_yaxes(
                    gridcolor=NAVY_BORDER, color=MUTED, row=(i-1)//3+1, col=(i-1)%3+1
                )
            st.plotly_chart(box_fig, use_container_width=True,
                            config={"displayModeBar": False})

        # ── Tab 3: Correlation heatmap ────────────────────────────────────────
        with tab3:
            df_enc = df.copy()
            df_enc["class_num"] = (df["class"] == "bankruptcy").astype(int)
            corr_cols = FEATURES + ["class_num"]
            corr = df_enc[corr_cols].corr()

            labels = [FEATURE_LABELS.get(c, c) for c in corr_cols]
            labels[-1] = "Bankruptcy"

            heat_fig = go.Figure(go.Heatmap(
                z=corr.values,
                x=labels,
                y=labels,
                colorscale=[[0, "#4A9EFF"], [0.5, NAVY_CARD], [1, GOLD]],
                zmid=0,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                textfont={"size": 11, "color": WHITE},
                hoverongaps=False,
            ))
            heat_fig = apply_theme(heat_fig, "Feature Correlation Matrix")
            heat_fig.update_layout(height=460)
            heat_fig.update_xaxes(tickangle=-35)
            st.plotly_chart(heat_fig, use_container_width=True,
                            config={"displayModeBar": False})

            # Top correlations with target
            target_corr = (
                corr["class_num"]
                .drop("class_num")
                .abs()
                .sort_values(ascending=False)
            )
            st.markdown(f"""
            <div style="margin-top:12px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;
                            text-transform:uppercase;letter-spacing:0.09em;
                            color:{MUTED};margin-bottom:12px;">
                    Absolute Correlation with Bankruptcy Label
                </div>
            """, unsafe_allow_html=True)
            for feat, corr_val in target_corr.items():
                bar_w = int(corr_val * 100)
                bar_color = RED_RISK if corr_val > 0.4 else (AMBER_MED if corr_val > 0.2 else GOLD_DIM)
                lbl = FEATURE_LABELS.get(feat, feat)
                st.markdown(f"""
                <div style="display:flex;align-items:center;
                            gap:10px;margin-bottom:8px;font-size:0.83rem;">
                    <span style="color:{MUTED};width:175px;">{lbl}</span>
                    <div style="flex:1;background:{NAVY_BORDER};
                                border-radius:3px;height:8px;">
                        <div style="background:{bar_color};
                                    width:{max(bar_w,2)}%;height:100%;
                                    border-radius:3px;"></div>
                    </div>
                    <span style="color:{WHITE};width:38px;text-align:right;
                                 font-family:'DM Mono',monospace;font-size:0.80rem;">
                        {corr_val:.3f}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Tab 4: Raw data ───────────────────────────────────────────────────
        with tab4:
            sort_col = st.selectbox(
                "Sort by",
                FEATURES + ["class"],
                index=0,
                key="eda_sort",
            )
            show_df = df.sort_values(sort_col).reset_index(drop=True)
            # Colour-code class column
            st.dataframe(
                show_df,
                use_container_width=True,
                height=420,
                column_config={
                    "class": st.column_config.TextColumn("Class", width="medium"),
                    **{
                        f: st.column_config.NumberColumn(
                            FEATURE_LABELS[f], format="%.1f"
                        )
                        for f in FEATURES
                    },
                },
            )
            st.caption(f"{len(df):,} rows · {len(df.columns)} columns · "
                       f"{df.duplicated().sum()} duplicates preserved")

    except FileNotFoundError:
        st.info("Dataset artifacts not found — run `python main.py` first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MLFLOW TRACKER
# ══════════════════════════════════════════════════════════════════════════════

elif "MLflow Tracker" in page:

    st.markdown(f"""
    <h1 style="font-size:2rem;margin-bottom:4px;">MLflow Experiment Tracker</h1>
    <p style="color:{MUTED};font-size:0.95rem;margin-bottom:32px;">
        All experiment runs recorded in <code>mlflow.db</code>.
        Compare hyperparameters and metrics across training sessions.
    </p>
    """, unsafe_allow_html=True)

    try:
        runs = load_mlflow_runs()

        if runs.empty:
            st.info("No MLflow runs found in mlflow.db.")
        else:
            # ── Summary strip ─────────────────────────────────────────────────
            ml1, ml2, ml3, ml4 = st.columns(4)
            finished = runs[runs["status"] == "FINISHED"]
            with ml1: st.metric("Total Runs",    len(runs))
            with ml2: st.metric("Finished",      len(finished))

            # Best AUC across runs — handle mixed column names
            auc_col = "test_auc" if "test_auc" in runs.columns else "cv_auc"
            if auc_col in runs.columns:
                best_auc = runs[auc_col].max()
                with ml3: st.metric("Best Test AUC", f"{best_auc:.4f}" if pd.notna(best_auc) else "—")

            if "best_model" in runs.columns:
                top_model = runs["best_model"].dropna().mode()
                with ml4: st.metric("Most Common Winner",
                                    top_model.iloc[0].replace("_"," ").title()
                                    if len(top_model) else "—")

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

            # ── AUC over runs chart ───────────────────────────────────────────
            if auc_col in runs.columns and "start_time" in runs.columns:
                chart_df = runs[["run_short", "start_time", auc_col,
                                 "best_model"]].dropna(subset=[auc_col])
                chart_df = chart_df.sort_values("start_time")

                line_fig = go.Figure()
                line_fig.add_trace(go.Scatter(
                    x=chart_df["run_short"],
                    y=chart_df[auc_col],
                    mode="lines+markers+text",
                    line=dict(color=GOLD, width=2),
                    marker=dict(size=9, color=GOLD, line=dict(color=GOLD_LIGHT, width=1)),
                    text=chart_df["best_model"].str.replace("_"," ").str.title(),
                    textposition="top center",
                    textfont=dict(size=10, color=MUTED),
                    hovertemplate=(
                        "<b>Run %{x}</b><br>"
                        f"{auc_col}: %{{y:.4f}}<br>"
                        "<extra></extra>"
                    ),
                ))
                line_fig = apply_theme(line_fig, "Test AUC Across Experiment Runs")
                line_fig.update_layout(
                    height=320,
                    yaxis_range=[
                        max(0, chart_df[auc_col].min() - 0.05),
                        min(1.02, chart_df[auc_col].max() + 0.02),
                    ],
                    xaxis_title="Run ID (short)",
                    yaxis_title="AUC",
                )
                st.plotly_chart(line_fig, use_container_width=True,
                                config={"displayModeBar": False})

            # ── Runs table ────────────────────────────────────────────────────
            display_cols = ["run_short", "start_time", "status", "best_model"]
            metric_cols  = [c for c in ["cv_auc_mean", "test_auc",
                                        "test_accuracy", "cv_auc", "cv_auc_std"]
                            if c in runs.columns]
            param_cols   = [c for c in ["C", "kernel", "gamma",
                                        "n_estimators", "max_depth",
                                        "learning_rate", "num_features"]
                            if c in runs.columns]

            show_cols = [c for c in display_cols + metric_cols + param_cols
                         if c in runs.columns]

            col_config = {
                "run_short":   st.column_config.TextColumn("Run ID"),
                "start_time":  st.column_config.DatetimeColumn("Timestamp",
                                format="YYYY-MM-DD HH:mm"),
                "status":      st.column_config.TextColumn("Status"),
                "best_model":  st.column_config.TextColumn("Best Model"),
                "cv_auc_mean": st.column_config.NumberColumn("CV AUC Mean",
                                format="%.4f"),
                "test_auc":    st.column_config.NumberColumn("Test AUC",
                                format="%.4f"),
                "test_accuracy": st.column_config.NumberColumn("Test Acc",
                                  format="%.4f"),
                "cv_auc":      st.column_config.NumberColumn("CV AUC",
                                format="%.4f"),
                "cv_auc_std":  st.column_config.NumberColumn("CV Std",
                                format="%.6f"),
            }

            st.dataframe(
                runs[show_cols].sort_values("start_time", ascending=False)
                    .reset_index(drop=True),
                use_container_width=True,
                height=420,
                column_config=col_config,
            )

            # ── Model frequency bar ───────────────────────────────────────────
            if "best_model" in runs.columns:
                model_counts = (
                    runs["best_model"].dropna()
                    .value_counts()
                    .reset_index()
                )
                model_counts.columns = ["model", "count"]
                model_counts["model"] = (
                    model_counts["model"].str.replace("_", " ").str.title()
                )

                bar_fig = go.Figure(go.Bar(
                    x=model_counts["model"],
                    y=model_counts["count"],
                    marker_color=GOLD,
                    marker_line_color=GOLD_LIGHT,
                    marker_line_width=1,
                    text=model_counts["count"],
                    textposition="outside",
                    textfont=dict(color=WHITE),
                ))
                bar_fig = apply_theme(bar_fig, "Model Selection Frequency")
                bar_fig.update_layout(height=300, yaxis_title="Times Selected as Best")
                st.plotly_chart(bar_fig, use_container_width=True,
                                config={"displayModeBar": False})

    except Exception as ex:
        st.error(f"Could not load MLflow data: {ex}")
        st.info("Ensure `mlflow.db` exists in the project root "
                "and at least one training run has completed.")