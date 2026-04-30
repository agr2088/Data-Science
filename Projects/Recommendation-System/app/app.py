"""
CourseIQ — Online Course Recommendation Dashboard
Royal aesthetic: Indian Purple · Royal Purple · Light Bronze
Layer 1 → User ID → Top-N Hybrid Recommendations
Layer 2 → Course Selection → Semantic Content-Similar Courses
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Ensure project root is in the path to load the model correctly
_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR not in sys.path:
    sys.path.append(_DIR)

from src.models.hybrid_model import HybridModel

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="CourseIQ",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded",
)

RAW  = os.path.join(_DIR, "data", "raw", "online_course_recommendation.xlsx")
EVAL = os.path.join(_DIR, "reports", "evaluation_metrics.json")

CAT_MAP = {
    "Python for Beginners": "Programming",
    "Advanced Machine Learning": "Data Science",
    "AI for Business Leaders": "Data Science",
    "Data Visualization with Tableau": "Data Science",
    "Blockchain and Decentralized Applications": "Technology",
    "Cloud Computing Essentials": "Technology",
    "DevOps and Continuous Deployment": "Technology",
    "Networking and System Administration": "Technology",
    "Cybersecurity for Professionals": "Security",
    "Ethical Hacking Masterclass": "Security",
    "Game Development with Unity": "Programming",
    "Mobile App Development with Swift": "Programming",
    "Fundamentals of Digital Marketing": "Business",
    "Project Management Fundamentals": "Business",
    "Personal Finance and Wealth Building": "Finance",
    "Stock Market and Trading Strategies": "Finance",
    "Graphic Design with Canva": "Creative",
    "Photography and Video Editing": "Creative",
    "Public Speaking Mastery": "Communication",
    "Fitness and Nutrition Coaching": "Health",
}

CAT_COLORS = {
    "Programming": "#C85A29", "Data Science": "#079999",
    "Technology": "#7851A9",  "Security": "#C0392B",
    "Business": "#27AE60",    "Finance": "#D9A761",
    "Creative": "#A424CC",    "Communication": "#2471A3",
    "Health": "#1E8449",
    "Other Domain": "#5C6B7A"
}

ROYAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');
:root {
    --ip:#492666; --rp:#7851A9; --bv:#8309A2; --hp:#A424CC;
    --mp:#E5C5A3; --lb:#D9A761; --dt:#BA9D74; --dn:#079999;
    --bg:#100820; --card:rgba(73,38,102,0.50); --glow:rgba(217,167,97,0.32);
    --t1:#EEE0CC; --t2:#C8B089;
}
.stApp {
    background: radial-gradient(ellipse at 12% 0%, #2a0a44 0%, #100820 45%),
                radial-gradient(ellipse at 88% 100%, #061a2a 0%, #100820 50%) !important;
    background-blend-mode: screen !important;
    background-size: 200% 200% !important;
    animation: nebulaDrift 20s ease-in-out infinite !important;
}
[data-testid="stSidebar"] {
    background: rgba(73,38,102,0.72) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border-right: 1px solid rgba(217,167,97,0.30) !important;
    box-shadow: 4px 0 32px rgba(120,81,169,0.22) !important;
}
[data-testid="stSidebar"] * { color: var(--t1) !important; }
[data-testid="stSidebar"] label {
    font-family:'Inter',sans-serif !important; font-size:0.72rem !important;
    letter-spacing:0.10em !important; text-transform:uppercase !important;
    color:var(--lb) !important;
}
[data-testid="stSidebar"] .stSelectbox>div>div,
[data-testid="stSidebar"] .stNumberInput>div>div {
    background:rgba(120,81,169,0.25) !important;
    border:1px solid rgba(217,167,97,0.28) !important; border-radius:7px !important;
}
[data-testid="stSidebar"] input {
    background:rgba(120,81,169,0.20) !important; color:var(--t1) !important;
    font-family:'Space Mono',monospace !important; font-size:1rem !important;
}
[data-testid="stMetric"] {
    background: linear-gradient(140deg, rgba(73,38,102,0.65), rgba(120,81,169,0.32)) !important;
    border:1px solid rgba(217,167,97,0.38) !important; border-radius:12px !important;
    padding:1rem 1.1rem !important;
    box-shadow: 0 4px 22px var(--glow), inset 0 1px 0 rgba(217,167,97,0.12) !important;
    transition: transform 0.25s ease, box-shadow 0.25s ease !important;
}
[data-testid="stMetric"]:hover { transform:translateY(-3px) !important; box-shadow:0 10px 36px var(--glow) !important; }
[data-testid="stMetricLabel"] {
    font-family:'Inter',sans-serif !important; font-size:0.66rem !important;
    letter-spacing:0.14em !important; text-transform:uppercase !important; color:var(--lb) !important;
}
[data-testid="stMetricValue"] {
    font-family:'EB Garamond',serif !important; font-size:1.85rem !important;
    color:var(--mp) !important; font-weight:600 !important;
}
.stButton>button {
    font-family:'Inter',sans-serif !important;
    background:linear-gradient(135deg,#7851A9,#A424CC) !important; color:#F0E8D8 !important;
    border:1px solid rgba(217,167,97,0.45) !important; border-radius:8px !important;
    letter-spacing:0.07em !important; font-size:0.77rem !important; font-weight:500 !important;
    padding:0.5rem 1.4rem !important; box-shadow:0 4px 18px rgba(131,9,162,0.38) !important;
    transition:all 0.28s ease !important;
}
.stButton>button:hover {
    background:linear-gradient(135deg,#A424CC,#C030E8) !important;
    box-shadow:0 6px 28px rgba(164,36,204,0.55),0 0 18px rgba(217,167,97,0.28) !important;
    transform:translateY(-2px) !important;
}
.stSelectbox>div>div, .stMultiSelect>div>div {
    background:rgba(73,38,102,0.52) !important; border:1px solid rgba(217,167,97,0.33) !important;
    border-radius:8px !important; color:var(--t1) !important;
}
div[data-baseweb="select"] * { color:var(--t1) !important; }
.stSelectbox svg,.stMultiSelect svg { color:var(--lb) !important; }
.stNumberInput input, .stTextInput input {
    background:rgba(73,38,102,0.45) !important; border:1px solid rgba(217,167,97,0.33) !important;
    border-radius:8px !important; color:var(--mp) !important;
    font-family:'Space Mono',monospace !important; font-size:1.1rem !important;
}
.stSlider>div>div>div>div { background:var(--rp) !important; }
.stSlider>div>div>div>div>div { background:var(--lb) !important; }
.stTabs [data-baseweb="tab-list"] {
    background:rgba(73,38,102,0.45) !important; border-bottom:2px solid rgba(217,167,97,0.28) !important;
    border-radius:10px 10px 0 0 !important; padding:0 0.4rem !important; gap:0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family:'Inter',sans-serif !important; font-size:0.74rem !important;
    letter-spacing:0.09em !important; text-transform:uppercase !important;
    color:var(--dt) !important; border-radius:8px 8px 0 0 !important;
    padding:0.65rem 1.3rem !important; transition:all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(180deg,rgba(131,9,162,0.32),transparent) !important;
    border-bottom:3px solid var(--lb) !important; color:var(--mp) !important;
}
.streamlit-expanderHeader {
    font-family:'Inter',sans-serif !important; font-size:0.78rem !important;
    letter-spacing:0.08em !important; background:rgba(73,38,102,0.38) !important;
    border:1px solid rgba(217,167,97,0.22) !important; border-radius:8px !important;
    color:var(--mp) !important;
}
[data-testid="stDataFrame"] { border:1px solid rgba(217,167,97,0.25) !important; border-radius:10px !important; overflow:hidden !important; }
.stAlert { background:rgba(73,38,102,0.45) !important; border:1px solid rgba(217,167,97,0.3) !important; border-radius:8px !important; color:var(--mp) !important; }
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:rgba(73,38,102,0.25); }
::-webkit-scrollbar-thumb { background:rgba(217,167,97,0.45); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:rgba(217,167,97,0.7); }
#MainMenu, footer, header { visibility:hidden !important; }
.block-container { padding-top:1.2rem !important; padding-bottom:2rem !important; }
.rec-rank { font-family:'Inter',sans-serif; font-size:0.60rem; letter-spacing:0.18em; text-transform:uppercase; color:#BA9D74; margin-bottom:0.18rem; }
.rec-title { font-family:'EB Garamond',serif; font-size:1.05rem; font-weight:600; color:#E5C5A3; line-height:1.28; margin-bottom:0.28rem; }
.rec-meta { font-family:'Inter',sans-serif; font-size:0.76rem; color:#BA9D74; margin-bottom:0.4rem; }
.score-pill { display:inline-block; font-family:'Space Mono',monospace; font-size:0.70rem; font-weight:700; color:#100820; background:linear-gradient(90deg,#D9A761,#BA9D74); border-radius:20px; padding:0.15rem 0.7rem; margin-right:0.4rem; }
.badge-diff { display:inline-block; font-family:'Inter',sans-serif; font-size:0.60rem; letter-spacing:0.10em; text-transform:uppercase; padding:0.13rem 0.55rem; border-radius:4px; border:1px solid; margin-right:0.3rem; }
.b-beg { color:#27AE60; border-color:#27AE60; background:rgba(39,174,96,0.10); }
.b-int { color:#E67E22; border-color:#E67E22; background:rgba(230,126,34,0.10); }
.b-adv { color:#C0392B; border-color:#C0392B; background:rgba(192,57,43,0.10); }
.b-cat { color:#BA9D74; border-color:rgba(217,167,97,0.40); background:rgba(217,167,97,0.08); }
.b-cert{ color:#D9A761; border-color:#D9A761; background:rgba(217,167,97,0.12); }
.mbar-wrap { background:rgba(73,38,102,0.35); border-radius:4px; height:5px; margin-top:4px; overflow:hidden; }
.stat-tile {
    background:linear-gradient(140deg, rgba(73,38,102,0.60), rgba(7,153,153,0.18));
    border:1px solid rgba(217,167,97,0.30); border-radius:12px;
    padding:0.85rem 1rem; text-align:center; box-shadow:0 3px 16px rgba(73,38,102,0.35);
}
.stat-label { font-family:'Inter',sans-serif; font-size:0.60rem; letter-spacing:0.14em; text-transform:uppercase; color:#D9A761; margin-bottom:0.2rem; }
.stat-val { font-family:'EB Garamond',serif; font-size:1.8rem; font-weight:600; color:#E5C5A3; line-height:1.1; }
.stat-sub { font-family:'Inter',sans-serif; font-size:0.68rem; color:#BA9D74; margin-top:0.1rem; }
.progress-ring {
    display:inline-flex; align-items:center; justify-content:center;
    width:48px; height:48px; border-radius:50%;
    border:2px solid rgba(217,167,97,0.50);
    font-family:'Space Mono',monospace; font-size:0.75rem; font-weight:700; color:#D9A761;
    background:radial-gradient(circle, rgba(73,38,102,0.85) 55%, rgba(120,81,169,0.22) 100%);
    flex-shrink:0; box-shadow:0 0 12px rgba(217,167,97,0.32);
}

/* ── ANIMATED PIPELINE ── */
@keyframes particleFlow {
    0%   { left:-12%; opacity:0; }
    10%  { opacity:1; }
    90%  { opacity:1; }
    100% { left:108%; opacity:0; }
}
@keyframes nodeGlow {
    0%,100% { box-shadow:0 0 8px rgba(217,167,97,0.25), 0 0 0px rgba(164,36,204,0.0); }
    50%      { box-shadow:0 0 22px rgba(217,167,97,0.85), 0 0 40px rgba(164,36,204,0.35); }
}
@keyframes nodePulseActive {
    0%,100% { box-shadow:0 0 12px rgba(217,167,97,0.4), inset 0 0 10px rgba(217,167,97,0.05); }
    50%      { box-shadow:0 0 30px rgba(217,167,97,1.0), inset 0 0 20px rgba(217,167,97,0.15); }
}
@keyframes flowLine {
    0%   { background-position:0% 50%; }
    100% { background-position:200% 50%; }
}
@keyframes fadeInUp {
    0%   { opacity:0; transform:translateY(12px); }
    100% { opacity:1; transform:translateY(0); }
}
@keyframes weightBar {
    0%   { width:0%; }
    100% { width:var(--bar-w); }
}

.pipeline-wrap {
    background: linear-gradient(135deg, rgba(16,8,32,0.85), rgba(73,38,102,0.45));
    border: 1px solid rgba(217,167,97,0.30);
    border-radius: 16px;
    padding: 1.4rem 1.6rem 1.2rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.pipeline-wrap::before {
    content:'';
    position:absolute; inset:0;
    background: radial-gradient(ellipse at 30% 50%, rgba(120,81,169,0.12) 0%, transparent 70%),
                radial-gradient(ellipse at 80% 50%, rgba(7,153,153,0.08) 0%, transparent 60%);
    pointer-events:none;
}
.pipe-label-row {
    display:flex; justify-content:center; gap:4rem;
    font-family:'Inter',sans-serif; font-size:0.60rem;
    letter-spacing:0.18em; text-transform:uppercase;
    color:#BA9D74; margin-bottom:0.7rem;
}
.pipe-label-row span { opacity:0.75; }
.pipe-label-row .pipe-arrow { color:rgba(217,167,97,0.5); }

.pipeline-row {
    display:flex; align-items:stretch; gap:0; width:100%; min-height:140px;
}
.pipe-node-user {
    flex-shrink:0; width:110px; align-self:center;
    background: linear-gradient(135deg, rgba(73,38,102,0.70), rgba(16,8,32,0.60));
    border: 1.5px solid rgba(217,167,97,0.45);
    border-radius: 12px;
    padding: 0.8rem 0.6rem;
    text-align:center;
    animation: nodeGlow 2.5s ease-in-out infinite;
    position:relative;
}
.pipe-node-user .node-icon { font-size:1.5rem; margin-bottom:0.2rem; }
.pipe-node-user .node-title {
    font-family:'Inter',sans-serif; font-size:0.68rem; font-weight:600;
    color:#E5C5A3; letter-spacing:0.06em; text-transform:uppercase;
}
.pipe-node-user .node-sub {
    font-family:'Inter',sans-serif; font-size:0.58rem; color:#BA9D74;
    margin-top:0.15rem; line-height:1.35;
}

.pipe-connector {
    flex:0 0 32px; height:3px; position:relative; overflow:visible; min-width:18px; align-self:center;
}
.pipe-connector-line {
    height:3px; border-radius:2px;
    background: linear-gradient(90deg, #492666, #7851A9 40%, #D9A761 60%, #492666);
    background-size:200% 100%;
    animation: flowLine 1.8s linear infinite;
    position:relative;
}
.pipe-connector-line.dashed {
    height:2px;
    background: none;
    border-top: 2px dashed rgba(217,167,97,0.40);
    animation: none;
}
.pipe-connector-particle {
    position:absolute; top:-4px;
    width:10px; height:10px; border-radius:50%;
    background: radial-gradient(circle, #D9A761, #A424CC);
    animation: particleFlow 1.8s ease-in-out infinite;
    box-shadow: 0 0 6px rgba(217,167,97,0.8);
}
.pipe-connector-particle:nth-child(2) { animation-delay:0.6s; }
.pipe-connector-particle:nth-child(3) { animation-delay:1.2s; }

.pipe-node-hybrid {
    flex-shrink:0; width:130px; align-self:center;
    background: linear-gradient(135deg, rgba(131,9,162,0.35), rgba(73,38,102,0.65));
    border: 2px solid #D9A761;
    border-radius: 14px;
    padding: 0.9rem 0.7rem;
    text-align:center;
    animation: nodePulseActive 2s ease-in-out infinite;
    position:relative; z-index:2;
}
.pipe-node-hybrid .hyb-icon {
    font-size:1.6rem; margin-bottom:0.25rem;
    display:inline-block;
}
.pipe-node-hybrid .hyb-title {
    font-family:'EB Garamond',serif; font-size:1.0rem; font-weight:600;
    color:#D9A761; letter-spacing:0.05em;
}
.pipe-node-hybrid .hyb-sub {
    font-family:'Inter',sans-serif; font-size:0.58rem; color:#C8B089;
    letter-spacing:0.08em; text-transform:uppercase;
}

.pipe-model-group {
    display:flex; flex:1; gap:4px; align-items:stretch; flex-direction:column; justify-content:space-between;
}
.pipe-model-node {
    flex:1;
    border-radius:8px;
    padding:0.32rem 0.5rem;
    display:flex; align-items:center; gap:0.4rem;
    border:1px solid rgba(217,167,97,0.22);
    background:rgba(73,38,102,0.30);
    transition: all 0.4s ease;
    opacity:0.35;
    position:relative;
}
.pipe-model-node.active {
    opacity:1;
    animation: auraGlow 2.2s ease-in-out infinite, modelActivePulse 2.8s ease-in-out infinite;
}
.pipe-model-node .mn-icon { font-size:0.95rem; flex-shrink:0; }
.pipe-model-node .mn-title {
    font-family:'Inter',sans-serif; font-size:0.58rem; font-weight:600;
    letter-spacing:0.07em; text-transform:uppercase;
    margin-bottom:0; line-height:1.2;
}
.pipe-model-node .mn-sub {
    font-family:'Inter',sans-serif; font-size:0.50rem; color:#BA9D74;
    line-height:1.2;
}
.mn-semantic  { border-color:rgba(7,153,153,0.55)!important;  }
.mn-semantic.active  { background:rgba(7,153,153,0.15)!important; }
.mn-semantic .mn-title  { color:#079999; }
.mn-collab    { border-color:rgba(120,81,169,0.55)!important; }
.mn-collab.active    { background:rgba(120,81,169,0.15)!important; }
.mn-collab .mn-title    { color:#7851A9; }
.mn-pop       { border-color:rgba(164,36,204,0.55)!important; }
.mn-pop.active       { background:rgba(164,36,204,0.15)!important; }
.mn-pop .mn-title       { color:#A424CC; }
.mn-interest  { border-color:rgba(217,167,97,0.55)!important; }
.mn-interest.active  { background:rgba(217,167,97,0.10)!important; }
.mn-interest .mn-title  { color:#D9A761; }
.mn-knn       { border-color:rgba(200,90,41,0.55)!important;  }
.mn-knn.active       { background:rgba(200,90,41,0.12)!important; }
.mn-knn .mn-title       { color:#C85A29; }

.pipe-node-output {
    flex-shrink:0; width:110px; align-self:center;
    background: linear-gradient(135deg, rgba(7,153,153,0.20), rgba(16,8,32,0.65));
    border: 1.5px solid rgba(7,153,153,0.50);
    border-radius: 12px;
    padding: 0.8rem 0.6rem;
    text-align:center;
    animation: nodeGlow 2.8s ease-in-out infinite 0.5s;
}
.pipe-node-output .out-icon { font-size:1.5rem; margin-bottom:0.2rem; }
.pipe-node-output .out-title {
    font-family:'Inter',sans-serif; font-size:0.68rem; font-weight:600;
    color:#079999; letter-spacing:0.06em; text-transform:uppercase;
}
.pipe-node-output .out-sub {
    font-family:'Inter',sans-serif; font-size:0.58rem; color:#BA9D74;
    margin-top:0.15rem; line-height:1.35;
}

/* ── MODE BADGE ── */
.mode-badge {
    display:inline-flex; align-items:center; gap:0.4rem;
    border-radius:20px; padding:0.3rem 1rem;
    font-family:'Space Mono',monospace; font-size:0.72rem; font-weight:700;
    margin:0.6rem auto 0; border:1px solid;
}
.mode-cold   { color:#079999; border-color:#079999; background:rgba(7,153,153,0.10); }
.mode-warm   { color:#E67E22; border-color:#E67E22; background:rgba(230,126,34,0.10); }
.mode-full   { color:#27AE60; border-color:#27AE60; background:rgba(39,174,96,0.10); }

/* ── HYBRID ARCHITECTURE PANEL ── */
.hybrid-arch {
    background: linear-gradient(135deg, rgba(73,38,102,0.45), rgba(16,8,32,0.55));
    border: 1px solid rgba(217,167,97,0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 0.8rem;
    animation: fadeInUp 0.5s ease both;
}
.arch-title {
    font-family:'Inter',sans-serif; font-size:0.68rem; font-weight:600;
    letter-spacing:0.14em; text-transform:uppercase; color:#D9A761;
    margin-bottom:0.3rem;
}
.arch-sub {
    font-family:'Inter',sans-serif; font-size:0.70rem; color:#BA9D74;
    margin-bottom:0.8rem;
}
.arch-models-row {
    display:flex; gap:8px; flex-wrap:wrap;
}
.arch-model-card {
    flex:1; min-width:100px;
    background: rgba(16,8,32,0.45);
    border:1px dashed rgba(217,167,97,0.25);
    border-radius:10px; padding:0.7rem 0.6rem;
    animation: fadeInUp 0.5s ease both;
}
.arch-model-card .am-title {
    font-family:'Inter',sans-serif; font-size:0.60rem; font-weight:600;
    letter-spacing:0.10em; text-transform:uppercase; margin-bottom:0.15rem;
}
.arch-model-card .am-sub {
    font-family:'Inter',sans-serif; font-size:0.57rem; color:#BA9D74;
    line-height:1.4;
}
.am-semantic { color:#079999; border-color:rgba(7,153,153,0.35)!important; }
.am-collab   { color:#7851A9; border-color:rgba(120,81,169,0.35)!important; }
.am-pop      { color:#A424CC; border-color:rgba(164,36,204,0.35)!important; }
.am-interest { color:#D9A761; border-color:rgba(217,167,97,0.35)!important; }
.am-knn      { color:#C85A29; border-color:rgba(200,90,41,0.35)!important; }

/* ── WEIGHT BARS ── */
.wt-row { display:flex; gap:6px; margin-top:0.8rem; flex-wrap:wrap; }
.wt-card {
    flex:1; min-width:80px;
    background:rgba(16,8,32,0.40);
    border:1px solid rgba(217,167,97,0.18);
    border-radius:8px; padding:0.55rem 0.7rem;
    animation: fadeInUp 0.6s ease both;
}
.wt-card .wc-label {
    font-family:'Inter',sans-serif; font-size:0.58rem;
    letter-spacing:0.12em; text-transform:uppercase;
    margin-bottom:0.3rem;
}
.wt-card .wc-pct {
    font-family:'Space Mono',monospace; font-size:0.85rem; font-weight:700;
    color:#E5C5A3; float:right;
}
.wt-card .wc-bar-bg {
    clear:both;
    height:5px; border-radius:3px;
    background:rgba(73,38,102,0.50);
    margin-top:0.25rem; overflow:hidden;
}
.wt-card .wc-bar-fill {
    height:100%; border-radius:3px;
    animation: weightBar 1s ease both 0.3s;
}

/* ── HYBRID CORE PULSE ── */
@keyframes corePulse {
    0%,100% { transform:translate(-50%,-50%) scale(0.85); box-shadow:0 0 8px 2px rgba(217,167,97,0.55),0 0 20px 4px rgba(164,36,204,0.20); opacity:0.7; }
    50%      { transform:translate(-50%,-50%) scale(1.28); box-shadow:0 0 22px 6px rgba(217,167,97,0.90),0 0 45px 10px rgba(164,36,204,0.55); opacity:1; }
}
@keyframes coreRing {
    0%   { transform:translate(-50%,-50%) scale(1);   opacity:0.60; }
    100% { transform:translate(-50%,-50%) scale(2.6); opacity:0; }
}
.hybrid-core {
    position:absolute; top:50%; left:50%;
    width:14px; height:14px; border-radius:50%;
    background:radial-gradient(circle, #D9A761 0%, #A424CC 65%, transparent 100%);
    animation: corePulse 1.8s ease-in-out infinite;
    z-index:3; pointer-events:none;
}
.hybrid-core-ring {
    position:absolute; top:50%; left:50%;
    width:14px; height:14px; border-radius:50%;
    border:1.5px solid rgba(217,167,97,0.65);
    transform:translate(-50%,-50%);
    animation: coreRing 1.8s ease-out infinite;
    pointer-events:none; z-index:2;
}

/* ── RESPONSIVE SVG CONNECTIONS ── */
.hybrid-links {
    flex: 0 0 72px;
    position: relative;
    align-self: stretch;
}

.connection-line {
    stroke-width: 2.5px;
    fill: none;
    stroke-dasharray: 8 6;
    animation: flowLineSvg 1.2s linear infinite;
    filter: drop-shadow(0 0 4px currentColor);
}

@keyframes flowLineSvg {
    to { stroke-dashoffset: -28; }
}

/* ── ACTIVE MODEL AURA ── */
@keyframes auraGlow {
    0%,100% { box-shadow:0 0 10px 2px rgba(217,167,97,0.18),inset 0 0 8px rgba(217,167,97,0.04); }
    50%      { box-shadow:0 0 26px 5px rgba(217,167,97,0.50),inset 0 0 16px rgba(217,167,97,0.10); }
}
@keyframes modelActivePulse {
    0%,100% { transform:scale(1.00); }
    50%      { transform:scale(1.025); }
}
.mn-semantic.active  { box-shadow:0 0 20px rgba(7,153,153,0.45),  inset 0 0 10px rgba(7,153,153,0.07)  !important; }
.mn-collab.active    { box-shadow:0 0 20px rgba(120,81,169,0.45), inset 0 0 10px rgba(120,81,169,0.07) !important; }
.mn-pop.active       { box-shadow:0 0 20px rgba(164,36,204,0.45), inset 0 0 10px rgba(164,36,204,0.07) !important; }
.mn-interest.active  { box-shadow:0 0 20px rgba(217,167,97,0.50), inset 0 0 10px rgba(217,167,97,0.09) !important; }
.mn-knn.active       { box-shadow:0 0 20px rgba(200,90,41,0.45),  inset 0 0 10px rgba(200,90,41,0.07)  !important; }

/* ── UPGRADE 1: Nebula background drift ── */
@keyframes nebulaDrift {
    0%,100% { background-position: 12% 0%, 88% 100%; }
    33%      { background-position: 18% 6%, 82% 94%; }
    66%      { background-position: 9%  3%, 91% 97%; }
}

/* ── UPGRADE 2: Title shimmer ── */
@keyframes shimmerText {
    0%   { background-position: -200% center; }
    100% { background-position: 300% center; }
}
.title-shimmer {
    background: linear-gradient(90deg, #D9A761 25%, #FFF3D4 45%, #E5C5A3 55%, #D9A761 75%);
    background-size: 250% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmerText 5s linear infinite;
}

/* ── UPGRADE 3: Royal divider sweep ── */
@keyframes dividerSweep {
    0%   { background-position: 100% 0; }
    100% { background-position: -100% 0; }
}
.royal-divider {
    border: none; height: 1px; margin: 1.4rem 0;
    background: linear-gradient(90deg,
        transparent, #492666 5%, #7851A9 25%, #D9A761 50%, #7851A9 75%, #492666 95%, transparent);
    background-size: 200% 100%;
    animation: dividerSweep 4s linear infinite;
}

/* ── UPGRADE 6: mbar-fill animates from 0 ── */
.mbar-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #7851A9, #D9A761);
    width: 0;
    animation: weightBar 1.2s ease-out both;
}

/* ── UPGRADE 7: rec-card and sim-card staggered fadeInUp ── */
.rec-card {
    background: linear-gradient(135deg, rgba(73,38,102,0.60), rgba(120,81,169,0.28));
    border: 1px solid rgba(217,167,97,0.30); border-left: 3px solid #D9A761;
    border-radius: 10px; padding: 1rem 1.2rem; margin: 0.55rem 0;
    transition: transform 0.22s ease, box-shadow 0.22s ease;
    box-shadow: 0 3px 14px rgba(73,38,102,0.40);
    animation: fadeInUp 0.45s ease both;
}
.sim-card {
    background: linear-gradient(135deg, rgba(7,153,153,0.15), rgba(73,38,102,0.50));
    border: 1px solid rgba(7,153,153,0.35); border-left: 3px solid #079999;
    border-radius: 10px; padding: 0.9rem 1.1rem; margin: 0.5rem 0;
    transition: transform 0.22s ease, box-shadow 0.22s ease;
    animation: fadeInUp 0.45s ease both;
}

/* ── UPGRADE 8: stat-row-anim for dataset stats rows ── */
.stat-row-anim {
    animation: fadeInUp 0.4s ease both;
}

/* ── UPGRADE 10: mode badge pulse per color ── */
@keyframes modePulseCold {
    0%,100% { box-shadow: 0 0 6px #079999; }
    50%     { box-shadow: 0 0 18px #079999, 0 0 32px rgba(7,153,153,0.40); }
}
@keyframes modePulseWarm {
    0%,100% { box-shadow: 0 0 6px #E67E22; }
    50%     { box-shadow: 0 0 18px #E67E22, 0 0 32px rgba(230,126,34,0.40); }
}
@keyframes modePulseFull {
    0%,100% { box-shadow: 0 0 6px #27AE60; }
    50%     { box-shadow: 0 0 18px #27AE60, 0 0 32px rgba(39,174,96,0.40); }
}
.mode-cold { animation: modePulseCold 2s ease-in-out infinite; }
.mode-warm { animation: modePulseWarm 2s ease-in-out infinite; }
.mode-full { animation: modePulseFull 2s ease-in-out infinite; }

/* ── UPGRADE 11: sec-header underline draw ── */
@keyframes headerUnderline {
    from { width: 0; }
    to   { width: 100%; }
}
.sec-header {
    font-family: 'EB Garamond', serif; font-size: 1.12rem; font-weight: 600;
    color: #D9A761; letter-spacing: 0.08em;
    padding-bottom: 0.35rem; margin-bottom: 0.9rem;
    position: relative; border-bottom: none;
}
.sec-header::after {
    content: '';
    position: absolute; bottom: 0; left: 0;
    height: 1px; width: 0;
    background: linear-gradient(90deg, #D9A761, rgba(217,167,97,0.25));
    animation: headerUnderline 0.7s ease-out both 0.15s;
}

/* ── UPGRADE 5: SVG progress ring ── */
@keyframes ringFill {
    from { stroke-dashoffset: 125.66; }
    to   { stroke-dashoffset: 0; }
}
.ring-arc {
    animation: none;
}
.progress-ring-svg { flex-shrink: 0; display: block; }
</style>
"""

PT = dict(
    paper_bgcolor="rgba(16,8,32,0)", plot_bgcolor="rgba(73,38,102,0.12)",
    font=dict(family="Inter, sans-serif", color="#C8B089", size=11),
    margin=dict(l=12, r=12, t=44, b=12),
)

def _hex_rgba(h, a=0.18):
    h = h.lstrip("#")
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

def diff_badge(d):
    cls = {"Beginner":"b-beg","Intermediate":"b-int","Advanced":"b-adv"}.get(d,"b-beg")
    return f'<span class="badge-diff {cls}">{d}</span>' if d else ""

def cat_badge(c):
    return f'<span class="badge-diff b-cat">{c}</span>' if c else ""

def cert_badge(c):
    return '<span class="badge-diff b-cert">Certificate</span>' if c else ""

def _safe(row, col, default=0):
    v = row.get(col, default)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return v


@st.cache_resource(show_spinner=False)
def load_hybrid_model():
    try:
        return HybridModel.load()
    except Exception as e:
        st.error(f"Error loading trained Hybrid Model: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_raw():
    df = pd.read_excel(RAW)
    df["category"]   = df["course_name"].map(CAT_MAP).fillna("Other Domain")
    df["cert_bin"]   = (df["certification_offered"]    == "Yes").astype(int)
    df["study_bin"]  = (df["study_material_available"] == "Yes").astype(int)
    df["diff_num"]   = df["difficulty_level"].map({"Beginner":0,"Intermediate":1,"Advanced":2})
    df["completion"] = (df["time_spent_hours"] / df["course_duration_hours"].replace(0, np.nan)).clip(0, 1)
    df["eng_score"]  = 0.4*df["rating"]/5 + 0.35*df["feedback_score"] + 0.25*df["completion"]
    return df

@st.cache_data(show_spinner=False)
def build_courses(df):
    agg = df.groupby("course_name").agg(
        avg_rating     =("rating",               "mean"),
        avg_price      =("course_price",         "mean"),
        avg_enrollment =("enrollment_numbers",   "mean"),
        avg_duration   =("course_duration_hours","mean"),
        avg_feedback   =("feedback_score",       "mean"),
        avg_completion =("completion",           "mean"),
        avg_eng        =("eng_score",            "mean"),
        user_count     =("user_id",              "nunique"),
        cert_offered   =("cert_bin",             "max"),
        study_mat      =("study_bin",            "max"),
        difficulty     =("difficulty_level",     "first"),
        diff_num       =("diff_num",             "mean"),
    ).reset_index()
    agg["category"] = agg["course_name"].map(CAT_MAP).fillna("Other Domain")
    scaler = MinMaxScaler()
    agg["pop_score"] = scaler.fit_transform(
        (agg["avg_enrollment"] * agg["user_count"]).values.reshape(-1,1)
    ).flatten()
    return agg.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def compute_similarity(courses):
    import pickle
    content_model_path = os.path.join(_DIR, "data", "processed", "content_model.pkl")
    MAT_ATTRS    = ["similarity_matrix", "sim_matrix", "_similarity_matrix", "_sim_matrix"]
    COURSE_ATTRS = ["course_ids", "course_names", "_course_ids", "_course_names", "courses", "items", "_items"]
    try:
        with open(content_model_path, "rb") as f:
            content_model = pickle.load(f)
        sim_matrix    = None
        model_courses = None
        for mat_attr in MAT_ATTRS:
            if not hasattr(content_model, mat_attr): continue
            mat = getattr(content_model, mat_attr)
            if mat is None: continue
            for crs_attr in COURSE_ATTRS:
                if not hasattr(content_model, crs_attr): continue
                crs = getattr(content_model, crs_attr)
                if crs is None or len(crs) == 0: continue
                if hasattr(mat, "shape") and mat.shape == (len(crs), len(crs)):
                    sim_matrix    = mat
                    model_courses = list(crs)
                    break
            if sim_matrix is not None: break
        if sim_matrix is None:
            for attr, val in vars(content_model).items():
                if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[0] == val.shape[1]:
                    n = val.shape[0]
                    for crs_attr in COURSE_ATTRS:
                        crs = getattr(content_model, crs_attr, None)
                        if crs is not None and len(crs) == n:
                            sim_matrix    = val
                            model_courses = list(crs)
                            break
                if sim_matrix is not None: break
        if sim_matrix is not None and model_courses is not None:
            sim_df = pd.DataFrame(sim_matrix, index=model_courses, columns=model_courses)
            return sim_df, "semantic"
    except FileNotFoundError: pass
    except Exception: pass
    feat = ["avg_rating", "avg_price", "avg_enrollment", "avg_feedback", "avg_duration", "diff_num", "avg_eng", "pop_score"]
    X   = MinMaxScaler().fit_transform(courses[feat].fillna(0).values)
    sim = cosine_similarity(X)
    return (pd.DataFrame(sim, index=courses["course_name"], columns=courses["course_name"]), "numeric")

@st.cache_data(show_spinner=False)
def get_user_history(df, uid):
    sub = df[df["user_id"]==uid][["course_name","rating","time_spent_hours","category","difficulty_level","course_price"]]
    return sub.sort_values("rating", ascending=False).drop_duplicates("course_name").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_eval():
    with open(EVAL) as f:
        return json.load(f)

def hybrid_recommend(df, courses, hybrid_model, uid, n=8, diff_f="All", cat_f="All"):
    hist = get_user_history(df, uid)
    if hybrid_model is None:
        return pd.DataFrame(), hist
    raw_recs = hybrid_model.recommend(user_id=uid, n=max(n * 5, 50))
    if raw_recs.empty:
        return pd.DataFrame(), hist
    raw_recs["category"] = raw_recs["course_name"].map(CAT_MAP).fillna("Other Domain")
    diff_map = {0: "Beginner", 1: "Intermediate", 2: "Advanced"}
    if "difficulty_level" in raw_recs.columns:
        raw_recs["difficulty_str"] = raw_recs["difficulty_level"].map(diff_map)
    else:
        raw_recs["difficulty_str"] = "Beginner"
    ui_cols = courses[["course_name", "avg_enrollment", "avg_duration", "avg_feedback", "avg_eng"]].drop_duplicates("course_name")
    recs = raw_recs.merge(ui_cols, on="course_name", how="left")
    recs["match_pct"] = (recs["hybrid_score"] * 100).clip(52, 98).round().astype(int)
    recs["difficulty"] = recs["difficulty_str"]
    recs["cert_offered"] = recs["certification_offered"]
    recs["avg_price"] = recs["course_price"]
    recs["avg_rating"] = recs["rating"]
    if diff_f and diff_f != "All":
        recs = recs[recs["difficulty"] == diff_f]
    if cat_f and cat_f != "All":
        recs = recs[recs["category"] == cat_f]
    return recs.head(n).reset_index(drop=True), hist

def semantic_recommend(sim_df, courses, source, n=6):
    catalogue_names = set(courses["course_name"].tolist())
    idx = None
    if source in sim_df.index:
        idx = source
    else:
        src_lower = source.lower()
        for entry in sim_df.index:
            entry_lower = entry.lower()
            if src_lower in entry_lower or entry_lower in src_lower:
                idx = entry
                break
    if idx is None:
        feat = ["avg_rating", "avg_price", "avg_enrollment", "avg_feedback", "avg_duration", "diff_num", "avg_eng", "pop_score"]
        X   = MinMaxScaler().fit_transform(courses[feat].fillna(0).values)
        sim = cosine_similarity(X)
        fb_df = pd.DataFrame(sim, index=courses["course_name"], columns=courses["course_name"])
        if source not in fb_df.index:
            return pd.DataFrame()
        scores = fb_df.loc[source].drop(index=source, errors="ignore")
        top    = scores.nlargest(n).reset_index()
        top.columns = ["course_name", "similarity_score"]
        return top.merge(courses, on="course_name", how="left")
    scores = sim_df.loc[idx].copy()
    scores = scores.drop(index=idx, errors="ignore")
    scores = scores[scores.index.isin(catalogue_names)]
    scores = scores[scores.index != source]
    if scores.empty:
        return pd.DataFrame()
    top = scores.nlargest(n).reset_index()
    top.columns = ["course_name", "similarity_score"]
    return top.merge(courses, on="course_name", how="left")

def render_rec_card(rank, row, delay=0):
    score  = int(_safe(row,"match_pct",80))
    diff   = _safe(row,"difficulty","")
    cat    = _safe(row,"category","")
    cert   = bool(_safe(row,"cert_offered",0))
    price  = float(_safe(row,"avg_price",0))
    rating = float(_safe(row,"avg_rating",0))
    dur    = float(_safe(row,"avg_duration",0))
    enroll = int(_safe(row,"avg_enrollment",0))
    circ   = 125.66
    offset = round(circ * (1 - score / 100), 2)
    uid    = f"ring-rec-{rank}"
    return f"""
<div class="rec-card" style="animation-delay:{delay}ms">
  <div style="display:flex;align-items:flex-start;gap:0.9rem;">
    <svg class="progress-ring-svg" viewBox="0 0 48 48" width="48" height="48">
      <circle cx="24" cy="24" r="20" fill="none" stroke="rgba(73,38,102,0.45)" stroke-width="3"/>
      <circle id="{uid}" cx="24" cy="24" r="20" fill="none" stroke="#D9A761" stroke-width="3"
        stroke-dasharray="125.66" stroke-dashoffset="125.66"
        stroke-linecap="round"
        style="transform:rotate(-90deg);transform-origin:center;"/>
      <text x="24" y="28" text-anchor="middle"
        font-family="'Space Mono',monospace" font-size="9"
        font-weight="700" fill="#D9A761">{score}%</text>
    </svg>
    <div style="flex:1;min-width:0;">
      <div class="rec-rank">Recommendation #{rank}</div>
      <div class="rec-title">{row['course_name']}</div>
      <div class="rec-meta">\u23f1 {dur:.0f}h &nbsp;\u00b7&nbsp; \U0001f4b0 ${price:.0f} &nbsp;\u00b7&nbsp; \u2b50 {rating:.2f} &nbsp;\u00b7&nbsp; \U0001f465 {enroll:,}</div>
      <div style="margin-bottom:0.35rem;">{diff_badge(diff)}{cat_badge(cat)}{cert_badge(cert)}</div>
      <div class="mbar-wrap"><div class="mbar-fill" style="--bar-w:{score}%;animation-delay:{delay}ms;"></div></div>
    </div>
  </div>
</div>
<script>(function(){{
  setTimeout(function(){{
    var el=document.getElementById('{uid}');
    if(!el||el.dataset.animated)return;
    el.dataset.animated='1';
    el.animate([{{strokeDashoffset:125.66}},{{strokeDashoffset:{offset}}}],
      {{duration:900,delay:{delay},easing:'cubic-bezier(0.22,1,0.36,1)',fill:'forwards'}});
  }},{delay+150});
}})();</script>"""

def render_sim_card(rank, row):
    sim    = float(_safe(row,"similarity_score",0))*100
    diff   = _safe(row,"difficulty","")
    cat    = _safe(row,"category","")
    price  = float(_safe(row,"avg_price",0))
    rating = float(_safe(row,"avg_rating",0))
    dur    = float(_safe(row,"avg_duration",0))
    circ   = 125.66
    offset = round(circ * (1 - sim / 100), 2)
    uid    = f"ring-sim-{rank}"
    return f"""
<div class="sim-card">
  <div style="display:flex;align-items:flex-start;gap:0.9rem;">
    <svg class="progress-ring-svg" viewBox="0 0 48 48" width="48" height="48">
      <circle cx="24" cy="24" r="20" fill="none" stroke="rgba(7,153,153,0.20)" stroke-width="3"/>
      <circle id="{uid}" cx="24" cy="24" r="20" fill="none" stroke="#079999" stroke-width="3"
        stroke-dasharray="125.66" stroke-dashoffset="125.66"
        stroke-linecap="round"
        style="transform:rotate(-90deg);transform-origin:center;"/>
      <text x="24" y="28" text-anchor="middle"
        font-family="'Space Mono',monospace" font-size="9"
        font-weight="700" fill="#079999">{sim:.0f}%</text>
    </svg>
    <div style="flex:1;min-width:0;">
      <div class="rec-rank" style="color:#079999;">Similar Course #{rank}</div>
      <div class="rec-title">{row['course_name']}</div>
      <div class="rec-meta">\u23f1 {dur:.0f}h &nbsp;\u00b7&nbsp; \U0001f4b0 ${price:.0f} &nbsp;\u00b7&nbsp; \u2b50 {rating:.2f}</div>
      <div>{diff_badge(diff)}{cat_badge(cat)}</div>
    </div>
  </div>
</div>
<script>(function(){{
  setTimeout(function(){{
    var el=document.getElementById('{uid}');
    if(!el||el.dataset.animated)return;
    el.dataset.animated='1';
    el.animate([{{strokeDashoffset:125.66}},{{strokeDashoffset:{offset}}}],
      {{duration:900,easing:'cubic-bezier(0.22,1,0.36,1)',fill:'forwards'}});
  }},150);
}})();</script>"""

def chart_radar(recs):
    dims   = ["avg_rating","avg_enrollment","avg_feedback","avg_eng","avg_duration"]
    labels = ["Rating","Enrolment","Feedback","Engagement","Duration"]
    mat    = MinMaxScaler().fit_transform(recs[dims].fillna(0).values)
    colors = ["#D9A761","#7851A9","#079999","#A424CC","#C85A29"]
    fig    = go.Figure()
    for i, (_, row) in enumerate(recs.head(5).iterrows()):
        v = mat[i].tolist() + [mat[i][0]]
        col = colors[i % len(colors)]
        nm  = row["course_name"][:26] + ("…" if len(row["course_name"])>26 else "")
        fig.add_trace(go.Scatterpolar(
            r=v, theta=labels+[labels[0]], fill="toself", name=nm,
            line=dict(color=col, width=2), fillcolor=_hex_rgba(col,0.10),
            hovertemplate="<b>%{theta}</b>: %{r:.2f}<extra>"+row["course_name"]+"</extra>",
        ))
    fig.update_layout(
        **PT,
        polar=dict(
            radialaxis=dict(visible=True,range=[0,1],tickfont=dict(size=8),showticklabels=False,gridcolor="rgba(217,167,97,0.18)"),
            angularaxis=dict(tickfont=dict(size=9,color="#C8B089"),gridcolor="rgba(217,167,97,0.14)"),
            bgcolor="rgba(73,38,102,0.15)",
        ),
        title=dict(text="Multi-Dimensional Profile",font=dict(size=13,color="#D9A761"),x=0.5),
        legend=dict(font=dict(size=9),bgcolor="rgba(73,38,102,0.65)",bordercolor="#D9A761",borderwidth=1),
        height=360,
    )
    return fig

def chart_sunburst(user_hist, courses):
    all_cats   = sorted(set(CAT_MAP.values()))
    known_cats = user_hist["category"].dropna().unique().tolist() if not user_hist.empty else []
    blind_cats = [c for c in all_cats if c not in known_cats]
    cnt_map    = courses.groupby("category")["course_name"].nunique().to_dict()
    labels  = ["Universe"] + known_cats + blind_cats
    parents = [""]         + ["Universe"]*len(known_cats) + ["Universe"]*len(blind_cats)
    values  = [0] + [cnt_map.get(c,1)*3 for c in known_cats] + [cnt_map.get(c,1) for c in blind_cats]
    colors  = (["rgba(217,167,97,0.55)"]
               + [CAT_COLORS.get(c,"#7851A9") for c in known_cats]
               + [_hex_rgba(CAT_COLORS.get(c,"#7851A9"),0.35) for c in blind_cats])
    fig = go.Figure(go.Sunburst(
        labels=labels, parents=parents, values=values,
        hovertemplate="<b>%{label}</b><br>Courses: %{value}<extra></extra>",
        branchvalues="remainder",
        textfont=dict(family="Inter",size=10),
        marker=dict(colors=colors, line=dict(color="#492666",width=1.2)),
        maxdepth=2, leaf=dict(opacity=0.88),
    ))
    fig.update_layout(
        **PT,
        title=dict(text="Your Domain Landscape & Blind Spots",font=dict(size=13,color="#D9A761"),x=0.5),
        height=380,
    )
    return fig

def chart_sankey(df):
    dfc = df.dropna(subset=["category","difficulty_level"])
    grp = dfc.groupby(["difficulty_level","category"]).size().reset_index(name="cnt")
    nodes = sorted(grp["difficulty_level"].unique()) + sorted(grp["category"].unique())
    nmap  = {n:i for i,n in enumerate(nodes)}
    dcol  = {"Beginner":"#27AE60","Intermediate":"#E67E22","Advanced":"#C0392B"}
    ncols = [dcol.get(n, CAT_COLORS.get(n,"#7851A9")) for n in nodes]
    fig   = go.Figure(go.Sankey(
        node=dict(pad=14,thickness=16,line=dict(color="#D9A761",width=0.8),
                  label=nodes, color=ncols, hovertemplate="%{label}<br>Flow: %{value:,}<extra></extra>"),
        link=dict(
            source=[nmap[r["difficulty_level"]] for _,r in grp.iterrows()],
            target=[nmap[r["category"]]         for _,r in grp.iterrows()],
            value =[r["cnt"]                    for _,r in grp.iterrows()],
            color =["rgba(217,167,97,0.22)"]*len(grp),
            hovertemplate="%{source.label} \u2192 %{target.label}<br>%{value:,} enrolments<extra></extra>",
        ),
        textfont=dict(family="Inter",size=10,color="#C8B089"),
    ))
    fig.update_layout(
        **PT,
        title=dict(text="Learner Journey \u2014 Difficulty \u2192 Domain Flow",font=dict(size=13,color="#D9A761"),x=0.5),
        height=400,
    )
    return fig

def chart_scatter(courses):
    fig = px.scatter(
        courses, x="avg_price", y="avg_rating",
        size="avg_enrollment", color="category",
        hover_name="course_name", size_max=42,
        color_discrete_map=CAT_COLORS,
        custom_data=["avg_feedback","user_count","difficulty","avg_duration"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>Price: $%{x:.0f}<br>Rating: %{y:.2f}\u2605<br>"
            "Feedback: %{customdata[0]:.2f}<br>Learners: %{customdata[1]:,}<br>"
            "Level: %{customdata[2]}<br>Duration: %{customdata[3]:.0f}h<extra></extra>"
        ),
        marker=dict(line=dict(color="#D9A761",width=0.8), opacity=0.82),
    )
    fig.update_layout(
        **PT,
        title=dict(text="Price vs Rating \u2014 Enrolment Universe",font=dict(size=13,color="#D9A761"),x=0.5),
        xaxis=dict(title="Avg Price ($)",tickfont=dict(size=9),gridcolor="rgba(217,167,97,0.12)"),
        yaxis=dict(title="Avg Rating \u2605",tickfont=dict(size=9),gridcolor="rgba(217,167,97,0.12)"),
        legend=dict(font=dict(size=9),bgcolor="rgba(73,38,102,0.65)",bordercolor="#D9A761",borderwidth=1,title_text=""),
        height=380,
    )
    return fig

def chart_treemap(courses):
    tmp = courses.copy()
    tmp["score_pct"] = (tmp["avg_eng"]*100).round(1).clip(lower=0.5)
    fig = px.treemap(
        tmp, path=["category","course_name"], values="score_pct", color="avg_rating",
        color_continuous_scale=[[0,"#492666"],[0.4,"#7851A9"],[0.7,"#D9A761"],[1,"#E5C5A3"]],
        custom_data=["avg_rating","avg_price","difficulty","user_count"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{label}</b><br>Rating: %{customdata[0]:.2f}\u2605<br>"
            "Price: $%{customdata[1]:.0f}<br>Level: %{customdata[2]}<br>"
            "Learners: %{customdata[3]:,}<extra></extra>"
        ),
        textfont=dict(family="Inter",size=11,color="#F0E8D8"),
        marker=dict(line=dict(color="#492666",width=1.5), pad=dict(t=22,l=3,r=3,b=3)),
        root_color="rgba(73,38,102,0.1)",
    )
    fig.update_layout(
        **PT,
        coloraxis_colorbar=dict(title=dict(text="Rating",font=dict(size=10)),
                                tickfont=dict(size=9),len=0.6,thickness=10,
                                outlinecolor="#D9A761",outlinewidth=1),
        title=dict(text="Domain Influence Treemap",font=dict(size=13,color="#D9A761"),x=0.5),
        height=420,
    )
    return fig

def chart_sim_matrix(sim_df, courses):
    all_names = courses["course_name"].tolist()
    names = [n for n in all_names if n in sim_df.index]
    if not names:
        names = list(sim_df.index)
    short = [n[:18]+"…" if len(n)>18 else n for n in names]
    z     = sim_df.loc[names, names].values
    fig   = go.Figure(go.Heatmap(
        z=z, x=short, y=short,
        colorscale=[[0,"#100820"],[0.4,"#492666"],[0.75,"#7851A9"],[1,"#D9A761"]],
        hovertemplate="<b>%{y}</b> \u2194 <b>%{x}</b><br>Similarity: %{z:.3f}<extra></extra>",
        colorbar=dict(tickfont=dict(size=8,color="#C8B089"),outlinecolor="#D9A761",outlinewidth=1),
        showscale=True,
    ))
    fig.update_layout(
        **PT,
        title=dict(text="Course Similarity Matrix",font=dict(size=13,color="#D9A761"),x=0.5),
        xaxis=dict(tickfont=dict(size=7,color="#C8B089"),tickangle=45),
        yaxis=dict(tickfont=dict(size=7,color="#C8B089")),
        height=450,
    )
    return fig

def chart_eval_heatmap(metrics):
    models  = ["hybrid","content","collaborative","popularity","knn"]
    mets    = ["ndcg_at_k","map_at_k","hit_rate_at_k","precision_at_k","recall_at_k"]
    labels_y= ["NDCG@10","MAP@10","HitRate@10","Precision@10","Recall@10"]
    labels_x= ["Hybrid","Content","Collab","Popularity","KNN"]
    tier    = "course_family"
    z, txt  = [], []
    for m in mets:
        row,trow = [],[]
        for mdl in models:
            v = metrics["models"][mdl][tier].get(m,0)
            row.append(v); trow.append(f"{v:.3f}")
        z.append(row); txt.append(trow)
    fig = go.Figure(go.Heatmap(
        z=z, x=labels_x, y=labels_y, text=txt, texttemplate="%{text}",
        colorscale=[[0,"#1A0A2E"],[0.3,"#492666"],[0.65,"#7851A9"],[1,"#D9A761"]],
        showscale=True,
        hovertemplate="<b>%{y}</b> \u2014 %{x}<br>Value: %{z:.4f}<extra></extra>",
        colorbar=dict(tickfont=dict(size=9,color="#C8B089"),outlinecolor="#D9A761",outlinewidth=1),
    ))
    fig.update_layout(
        **PT,
        title=dict(text="Model Evaluation Heatmap \u2014 Course Family Tier",font=dict(size=13,color="#D9A761"),x=0.5),
        xaxis=dict(tickfont=dict(size=10,color="#C8B089")),
        yaxis=dict(tickfont=dict(size=10,color="#C8B089")),
        height=320,
    )
    return fig

def chart_model_bars(metrics):
    models  = ["hybrid","content","collaborative","popularity","knn"]
    labels  = ["Hybrid","Content","Collab","Popularity","KNN"]
    tier    = "course_family"
    ndcg    = [metrics["models"][m][tier]["ndcg_at_k"]    for m in models]
    hr      = [metrics["models"][m][tier]["hit_rate_at_k"] for m in models]
    pal     = ["#D9A761","#7851A9","#079999","#A424CC","#C85A29"]
    fig     = go.Figure()
    for i,(lbl,nd,h) in enumerate(zip(labels,ndcg,hr)):
        fig.add_trace(go.Bar(
            x=[lbl], y=[nd], name=lbl,
            marker_color=pal[i], marker_line_color="#D9A761", marker_line_width=0.7,
            hovertemplate=f"<b>{lbl}</b><br>NDCG@10: {nd:.4f}<br>HitRate@10: {h:.4f}<extra></extra>",
            showlegend=False,
        ))
    fig.update_layout(
        **PT,
        title=dict(text="NDCG@10 \u2014 Model Comparison",font=dict(size=13,color="#D9A761"),x=0.5),
        xaxis=dict(tickfont=dict(size=11,color="#C8B089")),
        yaxis=dict(title="NDCG@10",tickfont=dict(size=9),gridcolor="rgba(217,167,97,0.12)"),
        height=300, bargap=0.35,
    )
    return fig

def chart_cat_bar(courses):
    agg  = courses.groupby("category").agg(Rating=("avg_rating","mean"),Engagement=("avg_eng","mean")).reset_index()
    cols = [CAT_COLORS.get(c,"#7851A9") for c in agg["category"]]
    fig  = go.Figure()
    fig.add_trace(go.Bar(
        x=agg["category"], y=agg["Rating"], name="Avg Rating",
        marker_color=cols, marker_line_color="#D9A761", marker_line_width=0.7,
        hovertemplate="<b>%{x}</b><br>Rating: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=agg["category"], y=agg["Engagement"]*5, name="Engagement\u00d75",
        marker_color=[_hex_rgba(c,0.55) for c in cols], marker_line_color="#D9A761", marker_line_width=0.5,
        hovertemplate="<b>%{x}</b><br>Engagement: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **PT,
        title=dict(text="Category Intelligence Matrix",font=dict(size=13,color="#D9A761"),x=0.5),
        barmode="group",
        xaxis=dict(tickfont=dict(size=9,color="#C8B089"),tickangle=30,gridcolor="rgba(217,167,97,0.10)"),
        yaxis=dict(title="Score",tickfont=dict(size=9),gridcolor="rgba(217,167,97,0.10)"),
        legend=dict(font=dict(size=9),bgcolor="rgba(73,38,102,0.65)",bordercolor="#D9A761",borderwidth=1),
        height=340,
    )
    return fig

def main():
    st.markdown(ROYAL_CSS, unsafe_allow_html=True)

    with st.spinner("Starting recommendation engine…"):
        df_raw  = load_raw()
        courses = build_courses(df_raw)
        sim_df, sim_source = compute_similarity(courses)
        metrics = load_eval()
        hybrid_model = load_hybrid_model()

    all_users   = sorted(df_raw["user_id"].unique())
    all_courses = sorted(courses["course_name"].tolist())

    # ── SIDEBAR ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:0.8rem 0 1.2rem;">
            <div style="font-family:'EB Garamond',serif;font-size:1.55rem;font-weight:600;
                color:#D9A761;letter-spacing:0.06em;">👑 CourseIQ</div>
            <div style="font-family:'Inter',sans-serif;font-size:0.70rem;color:#BA9D74;
                letter-spacing:0.12em;text-transform:uppercase;margin-top:0.2rem;">
                Recommendation Oracle</div>
        </div>
        <div class="royal-divider"></div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-header" style="font-size:0.78rem;">🔍 Layer 1 — User Lookup</div>', unsafe_allow_html=True)
        uid_input = st.number_input(
            "Scholar ID (sync with main bar)",
            min_value=int(all_users[0]), max_value=int(all_users[-1]),
            value=int(all_users[0]), step=1,
            key="sidebar_uid",
        )
        n_recs = 10

        st.markdown('<div class="royal-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-header" style="font-size:0.78rem;">🎯 Filters</div>', unsafe_allow_html=True)
        diff_filter = st.selectbox("Difficulty", ["All","Beginner","Intermediate","Advanced"])
        cat_filter  = st.selectbox("Domain", ["All"]+sorted(set(CAT_MAP.values())))

        st.markdown('<div class="royal-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-header" style="font-size:0.78rem;">⚙️ Recommendation Engine</div>', unsafe_allow_html=True)
        model_choice = st.selectbox(
            "Select Model",
            ["Hybrid", "Content-Based", "Collaborative", "Popularity"],
            index=0,
            key="model_choice",
        )

        st.markdown('<div class="royal-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Inter',sans-serif;font-size:0.65rem;color:#BA9D74;
            text-align:center;line-height:1.8;">
            Hybrid · ALS · KNN · MiniLM<br>
            <span style="color:#7851A9;">{len(all_users):,} Scholars · {len(courses["course_name"].unique()):,} Courses</span>
        </div>
        """, unsafe_allow_html=True)

    # ── HEADER ────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:0.6rem 0 0.3rem;">
        <div style="font-family:'EB Garamond',serif;font-size:2.4rem;font-weight:600;
            letter-spacing:0.05em;">
            <span class="title-shimmer">Online Course Recommendation System</span>
        </div>
        <div style="font-family:'Inter',sans-serif;font-size:0.78rem;color:#BA9D74;
            letter-spacing:0.14em;text-transform:uppercase;margin-top:0.3rem;">
            Hybrid · Content · Collaborative · Popularity · KNN &nbsp;|&nbsp;
            Interest + ALS + MiniLM Edition v4.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── MAIN USER ID SEARCH BAR ───────────────────────────
    search_col1, search_col2, search_col3 = st.columns([1, 2, 1])
    with search_col2:
        st.markdown("""
        <div style="text-align:center;margin-bottom:0.4rem;">
            <span style="font-family:'Inter',sans-serif;font-size:0.72rem;letter-spacing:0.14em;
                text-transform:uppercase;color:#D9A761;">🔍 Scholar Lookup</span>
        </div>""", unsafe_allow_html=True)
        uid_main = st.number_input(
            "Enter Scholar ID to personalise dashboard",
            min_value=int(all_users[0]),
            max_value=int(all_users[-1]),
            value=int(st.session_state.get("sidebar_uid", all_users[0])),
            step=1,
            key="main_uid_search",
            label_visibility="collapsed",
        )
        st.markdown(f"""
        <div style="text-align:center;margin-top:0.25rem;">
            <span style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#7851A9;">
                Scholar ID range: {int(all_users[0])} — {int(all_users[-1])}
            </span>
        </div>""", unsafe_allow_html=True)

    uid_input = uid_main

    # ── COMPUTE — Model-Routed ────────────────────────────
    # Get user history and compute initial rec bounds
    _, user_hist = hybrid_recommend(
        df_raw, courses, hybrid_model, uid_input,
        n=1, diff_f="All", cat_f="All",
    )

    if model_choice == "Hybrid":
        recs, user_hist = hybrid_recommend(
            df_raw, courses, hybrid_model, uid_input,
            n=n_recs, diff_f=diff_filter, cat_f=cat_filter,
        )

    elif model_choice == "Content-Based":
        if not user_hist.empty:
            seed_course = user_hist.iloc[0]["course_name"]
        else:
            seed_course = courses.sort_values("pop_score", ascending=False).iloc[0]["course_name"]
            
        if not seed_course:
            st.warning("Select a course for Content-Based recommendations.")
            st.stop()
            
        recs = semantic_recommend(sim_df, courses, seed_course, n=n_recs)
        if not recs.empty:
            recs["match_pct"] = (recs["similarity_score"] * 100).clip(40, 99).round().astype(int)
            recs["avg_price"]  = recs.get("avg_price", 0)
            recs["avg_rating"] = recs.get("avg_rating", 0)
            recs["cert_offered"] = recs.get("cert_offered", 0)
            
            if "difficulty" not in recs.columns:
                recs["difficulty"] = "Beginner"
                
            if diff_filter and diff_filter != "All":
                recs = recs[recs["difficulty"] == diff_filter]
            if cat_filter and cat_filter != "All" and "category" in recs.columns:
                recs = recs[recs["category"] == cat_filter]

    elif model_choice == "Collaborative":
        if not uid_input:
            st.warning("Enter User ID for Collaborative model.")
            st.stop()
            
        try:
            raw = hybrid_model.collaborative_model.recommend(user_id=uid_input, n=n_recs * 3)
            if raw is not None and not raw.empty:
                raw["category"] = raw["course_name"].map(CAT_MAP).fillna("Other Domain")
                ui_cols = courses[["course_name","avg_enrollment","avg_duration","avg_feedback","avg_eng","avg_price","avg_rating","difficulty","cert_offered"]].drop_duplicates("course_name")
                recs = raw.merge(ui_cols, on="course_name", how="left")
                score_col = next((c for c in ["score","collaborative_score","cf_score","rating"] if c in recs.columns), None)
                if score_col:
                    recs["match_pct"] = (recs[score_col] / recs[score_col].max() * 100).clip(40, 98).round().astype(int)
                else:
                    recs["match_pct"] = 75
                recs["avg_price"]    = recs.get("avg_price", 0)
                recs["avg_rating"]   = recs.get("avg_rating", 0)
                recs["difficulty"]   = recs.get("difficulty", "Beginner")
                recs["cert_offered"] = recs.get("cert_offered", 0)
                if diff_filter != "All":
                    recs = recs[recs["difficulty"] == diff_filter]
                if cat_filter != "All":
                    recs = recs[recs["category"] == cat_filter]
                recs = recs.head(n_recs).reset_index(drop=True)
            else:
                recs = pd.DataFrame()
        except Exception:
            recs = pd.DataFrame()

    elif model_choice == "Popularity":
        try:
            pop_fn = getattr(hybrid_model, "popularity_model", None) or getattr(hybrid_model, "_popularity_model", None)
            if pop_fn is not None:
                fn = getattr(pop_fn, "get_top_courses", None) or getattr(pop_fn, "recommend", None) or getattr(pop_fn, "top_courses", None)
                if fn:
                    raw = fn(n=n_recs * 3)
                else:
                    raw = None
            else:
                raw = None
            if raw is not None and not raw.empty:
                raw["category"] = raw["course_name"].map(CAT_MAP).fillna("Other Domain")
                ui_cols = courses[["course_name","avg_enrollment","avg_duration","avg_feedback","avg_eng","avg_price","avg_rating","difficulty","cert_offered","pop_score"]].drop_duplicates("course_name")
                recs = raw.merge(ui_cols, on="course_name", how="left")
            else:
                recs = courses.sort_values("pop_score", ascending=False).head(n_recs * 3).copy()
                recs["category"] = recs["course_name"].map(CAT_MAP).fillna("Other Domain")
            recs["match_pct"]    = (recs["pop_score"] * 100).clip(40, 99).round().astype(int)
            recs["avg_price"]    = recs.get("avg_price", 0)
            recs["avg_rating"]   = recs.get("avg_rating", 0)
            recs["difficulty"]   = recs.get("difficulty", "Beginner")
            recs["cert_offered"] = recs.get("cert_offered", 0)
            
            if diff_filter and diff_filter != "All":
                recs = recs[recs["difficulty"] == diff_filter]
            if cat_filter and cat_filter != "All":
                recs = recs[recs["category"] == cat_filter]
                
            recs = recs.head(n_recs).reset_index(drop=True)
        except Exception:
            recs = courses.sort_values("pop_score", ascending=False).head(n_recs * 3).copy()
            recs["category"]     = recs["course_name"].map(CAT_MAP).fillna("Other Domain")
            recs["match_pct"]    = (recs["pop_score"] * 100).clip(40, 99).round().astype(int)
            recs["avg_price"]    = recs.get("avg_price", 0)
            recs["avg_rating"]   = recs.get("avg_rating", 0)
            recs["difficulty"]   = recs.get("difficulty", "Beginner")
            recs["cert_offered"] = recs.get("cert_offered", 0)
            
            if diff_filter and diff_filter != "All":
                recs = recs[recs["difficulty"] == diff_filter]
            if cat_filter and cat_filter != "All":
                recs = recs[recs["category"] == cat_filter]
                
            recs = recs.head(n_recs).reset_index(drop=True)

    else:
        recs = pd.DataFrame()

    # ── REC MODE & DYNAMIC WEIGHTS ────────────────────────
    n_hist = len(user_hist)
    if n_hist >= 4:
        rec_mode = "Full"
        _wts = {"Content":0.70, "Popularity":0.12, "Collab":0.08, "Interest":0.07, "KNN":0.03}
        _mode_color = "#27AE60"
    elif n_hist >= 2:
        rec_mode = "Warm"
        _wts = {"Content":0.65, "Popularity":0.18, "Interest":0.08, "Collab":0.06, "KNN":0.03}
        _mode_color = "#E67E22"
    else:
        rec_mode = "Cold"
        _wts = {"Popularity":0.55, "Content":0.45, "Interest":0.0, "Collab":0.0, "KNN":0.0}
        _mode_color = "#079999"

    st.markdown('<div class="royal-divider"></div>', unsafe_allow_html=True)

    # ── KPI STRIP ─────────────────────────────────────────
    udf  = df_raw[df_raw["user_id"]==uid_input]
    total_scholars = df_raw["user_id"].nunique()
    total_courses  = courses["course_name"].nunique()
    k1,k2,k3,k4,k5,k6 = st.columns(6)

    # Build KPI data for countup component
    def _kpi_num(val):
        """Extract numeric value and suffix from a KPI value."""
        s = str(val)
        suffix = ""
        if "★" in s: suffix = "★"
        elif s.endswith("h"): suffix = "h"
        try:
            num = float(s.replace(",","").replace("★","").replace("h","").replace("$","").strip())
            decimals = 2 if suffix == "★" else 0
            return num, suffix, decimals
        except Exception:
            return None, None, None

    kpi_data = [
        ("Courses Taken",    n_hist,                                                   "by this scholar"),
        ("Avg Rating Given", f"{udf['rating'].mean():.2f}★" if not udf.empty else "—", "out of 5.0"),
        ("Hours Invested",   f"{udf['time_spent_hours'].sum():.0f}h" if not udf.empty else "—", "total learning"),
        ("Domains Explored", user_hist['category'].nunique() if not user_hist.empty else 0, f"of {len(set(CAT_MAP.values()))} total"),
        ("Total Scholars",   f"{total_scholars:,}",                                    "in dataset"),
        ("Total Courses",    f"{total_courses:,}",                                     "in catalogue"),
    ]

    # Render static tiles (always show real value — no JS needed for display)
    for col, (lbl, val, sub) in zip([k1,k2,k3,k4,k5,k6], kpi_data):
        with col:
            st.markdown(f"""
            <div class="stat-tile">
                <div class="stat-label">{lbl}</div>
                <div class="stat-val" id="kpi-{lbl.lower().replace(' ','-')}">{val}</div>
                <div class="stat-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # Countup animation via components.v1.html (scripts actually execute here)
    import streamlit.components.v1 as components
    kpi_js_data = []
    for lbl, val, sub in kpi_data:
        num, suffix, decimals = _kpi_num(val)
        if num is not None:
            kpi_js_data.append(f'{{"id":"kpi-{lbl.lower().replace(" ","-")}","target":{num},"suffix":"{suffix}","dec":{decimals}}}')
    kpi_js_array = "[" + ",".join(kpi_js_data) + "]"

    components.html(f"""
<script>
(function(){{
  var items = {kpi_js_array};
  function easeOut(t){{ return 1 - Math.pow(1-t, 3); }}
  function countup(el, target, suffix, dec){{
    var dur = 1200, start = performance.now();
    function tick(now){{
      var p = Math.min((now - start) / dur, 1);
      var v = easeOut(p) * target;
      el.textContent = (dec > 0 ? v.toFixed(dec) : Math.round(v).toLocaleString()) + suffix;
      if(p < 1) requestAnimationFrame(tick);
    }}
    requestAnimationFrame(tick);
  }}
  function run(){{
    items.forEach(function(item){{
      // Walk up into parent frames to find the element
      var el = null;
      try {{ el = window.parent.document.getElementById(item.id); }} catch(e){{}}
      if(!el) try {{ el = window.top.document.getElementById(item.id); }} catch(e){{}}
      if(el) countup(el, item.target, item.suffix, item.dec);
    }});
  }}
  setTimeout(run, 300);
}})();
</script>
""", height=0)

    st.markdown('<div class="royal-divider"></div>', unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────
    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        "  👑 Recommendations  ",
        "  🔗 Course Similarity  ",
        "  🗺️ Domain Intelligence  ",
        "  📊 Model Evaluation  ",
        "  📚 Data Explorer  ",
    ])

    # ════ TAB 1 — USER RECOMMENDATIONS ════
    with tab1:
        st.markdown('<div class="sec-header">⚙️ &nbsp;Processing Pipeline</div>', unsafe_allow_html=True)

        # ── Build per-model active flags ──────────────────────────────────────
        _model_defs = [
            ("semantic",  "mn-semantic",  "📄", "SEMANTIC",     "(Content)",     "Course content similarity using embeddings"),
            ("collab",    "mn-collab",    "👥", "COLLABORATIVE","(ALS)",         "User–item interactions via ALS matrix factorization"),
            ("pop",       "mn-pop",       "🔥", "POPULARITY",   "",              "Global popularity based recommendations"),
            ("interest",  "mn-interest",  "🧑", "USER-INTEREST","(Interest Model)","User profile &amp; interest alignment"),
            ("knn",       "mn-knn",       "🔗", "KNN",          "(Nearest Neighbors)","Similar users via KNN neighborhood"),
        ]
        _active_all = (model_choice == "Hybrid")
        _active_map = {
            "semantic":  _active_all or model_choice == "Content-Based",
            "collab":    _active_all or model_choice == "Collaborative",
            "pop":       _active_all or model_choice == "Popularity",
            "interest":  _active_all,
            "knn":       _active_all,
        }

        # ── Weight colour map ─────────────────────────────────────────────────
        _wt_colors = {
            "Content":    ("#079999", "#07999966"),
            "Popularity": ("#A424CC", "#A424CC66"),
            "Collaborative": ("#7851A9", "#7851A966"),
            "Collab":     ("#7851A9", "#7851A966"),
            "Interest":   ("#D9A761", "#D9A76166"),
            "User Interest": ("#D9A761", "#D9A76166"),
            "KNN":        ("#C85A29", "#C85A2966"),
        }

        # ── Build model nodes HTML ────────────────────────────────────────────
        model_nodes_html = ""
        for key, css, icon, title, sub, desc in _model_defs:
            act_cls = "active" if _active_map[key] else ""
            delay   = {"semantic":0,"collab":1,"pop":2,"interest":3,"knn":4}[key] * 0.15
            model_nodes_html += f"""
            <div class="pipe-model-node {css} {act_cls}" id="model-{key}" style="animation-delay:{delay}s;">
                <div class="mn-icon">{icon}</div>
                <div class="mn-title">{title}</div>
                <div class="mn-sub">{sub}</div>
            </div>"""

        # ── Particle connector helper ─────────────────────────────────────────
        def particle_connector(dashed=False):
            dash_cls = "dashed" if dashed else ""
            return f"""
            <div class="pipe-connector">
                <div class="pipe-connector-line {dash_cls}">
                    {"" if dashed else
                     '<div class="pipe-connector-particle"></div><div class="pipe-connector-particle"></div><div class="pipe-connector-particle"></div>'}
                </div>
            </div>"""

        mode_badge_cls = {"Cold":"mode-cold","Warm":"mode-warm","Full":"mode-full"}.get(rec_mode,"mode-full")
        mode_dot_color = {"Cold":"#079999","Warm":"#E67E22","Full":"#27AE60"}.get(rec_mode,"#27AE60")

        # Pre-compute connector HTML — must NOT be called inside f-string slots
        _conn_solid = particle_connector(dashed=False)

        pipeline_html = (
            '<div class="pipeline-wrap" style="position: relative;">'
            '<div class="pipe-label-row">'
            '<span>USER INPUT</span>'
            '<span class="pipe-arrow">\u2192</span>'
            '<span>MODEL LAYER</span>'
            '<span class="pipe-arrow">\u2192</span>'
            '<span>OUTPUT</span>'
            '</div>'
            '<div class="pipeline-row">'
            '<div class="pipe-node-user">'
            '<div class="node-icon">\U0001f9d1</div>'
            '<div class="node-title">USER INPUT</div>'
            '<div class="node-sub">User ID / History<br>Preferences</div>'
            '</div>'
            + _conn_solid +
            '<div class="pipe-node-hybrid" id="hybrid-node" style="position:relative;overflow:visible;">'
            '<div class="hybrid-core-ring"></div>'
            '<div class="hybrid-core-ring" style="animation-delay:0.6s;"></div>'
            '<div class="hybrid-core-ring" style="animation-delay:1.2s;"></div>'
            '<div class="hybrid-core"></div>'
            '<div class="hyb-icon">\U0001f9e0</div>'
            '<div class="hyb-title">HYBRID</div>'
            '<div class="hyb-sub">Ensemble Model</div>'
            '</div>'
            '<div class="hybrid-links">'
            '<svg style="position:absolute; top:0; left:0; width:100%; height:100%; overflow:visible;" viewBox="0 0 100 100" preserveAspectRatio="none">'
            '<path d="M 0,50 C 40,50 50,10 100,10" class="connection-line" stroke="#079999" />'
            '<path d="M 0,50 C 40,50 50,30 100,30" class="connection-line" stroke="#7851A9" />'
            '<path d="M 0,50 C 40,50 50,50 100,50" class="connection-line" stroke="#A424CC" />'
            '<path d="M 0,50 C 40,50 50,70 100,70" class="connection-line" stroke="#D9A761" />'
            '<path d="M 0,50 C 40,50 50,90 100,90" class="connection-line" stroke="#C85A29" />'
            '</svg>'
            '</div>'
            '<div class="pipe-model-group">'
            + model_nodes_html +
            '</div>'
            + _conn_solid +
            '<div class="pipe-node-output">'
            '<div class="out-icon">\U0001f3af</div>'
            '<div class="out-title">OUTPUT</div>'
            '<div class="out-sub">Top-N Personalized<br>Recommendations</div>'
            '</div>'
            '</div>'
            '<div style="text-align:center;margin-top:0.8rem;">'
            f'<span class="mode-badge {mode_badge_cls}">'
            f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{mode_dot_color};box-shadow:0 0 6px {mode_dot_color};"></span>'
            f' Mode: {rec_mode}-Start'
            '</span>'
            '</div>'
            '</div>'
        )
        st.markdown(pipeline_html, unsafe_allow_html=True)

        # ── HYBRID ARCHITECTURE + WEIGHT BARS (only when Hybrid selected) ─────
        if model_choice == "Hybrid":
            arch_cards_html = ""
            for key, css, icon, title, sub, desc in _model_defs:
                am_cls = {"semantic":"am-semantic","collab":"am-collab","pop":"am-pop","interest":"am-interest","knn":"am-knn"}[key]
                delay  = {"semantic":0,"collab":1,"pop":2,"interest":3,"knn":4}[key]*0.08
                arch_cards_html += f"""
                <div class="arch-model-card {am_cls}" style="animation-delay:{delay}s;">
                    <div class="am-title">{icon} {title} {sub}</div>
                    <div class="am-sub">{desc}</div>
                </div>"""

            # Weight bars
            active_wts = {k: v for k, v in _wts.items() if v > 0}
            wt_bars_html = ""
            for i, (k, v) in enumerate(active_wts.items()):
                pct       = int(v * 100)
                col_fg, _ = _wt_colors.get(k, ("#7851A9","#7851A966"))
                delay_s   = i * 0.1
                wt_bars_html += f"""
                <div class="wt-card" style="animation-delay:{delay_s}s;">
                    <span class="wc-label" style="color:{col_fg};">{k.upper()}</span>
                    <span class="wc-pct">{pct}%</span>
                    <div class="wc-bar-bg">
                        <div class="wc-bar-fill" style="--bar-w:{pct}%;width:{pct}%;background:linear-gradient(90deg,{col_fg},{col_fg}99);"></div>
                    </div>
                </div>"""

            st.markdown(f"""
            <div class="hybrid-arch">
                <div class="arch-title">🔀 &nbsp;HYBRID INTERNAL ARCHITECTURE</div>
                <div class="arch-sub">Hybrid combines multiple models using adaptive weighting based on user history (Cold / Warm / Full Start).</div>
                <div class="arch-models-row">{arch_cards_html}</div>
                <div style="font-family:'Inter',sans-serif;font-size:0.68rem;font-weight:600;
                    letter-spacing:0.12em;text-transform:uppercase;color:#D9A761;
                    margin-top:1rem;margin-bottom:0.4rem;">
                    📊 MODEL CONTRIBUTIONS <span style="color:#BA9D74;font-weight:400;">(Current Weights)</span>
                </div>
                <div class="wt-row">{wt_bars_html}</div>
                <div style="margin-top:0.8rem;font-family:'Inter',sans-serif;font-size:0.66rem;
                    color:#BA9D74;padding:0.55rem 0.8rem;background:rgba(73,38,102,0.25);
                    border-radius:7px;border-left:2px solid rgba(217,167,97,0.30);">
                    ℹ️ The hybrid model dynamically adjusts the contribution of each component based on data availability and user interaction patterns.
                    &nbsp;&nbsp;&nbsp;
                    <span style="color:#079999;">● Cold Start</span>
                    &nbsp;<span style="color:#E67E22;">● Warm Start</span>
                    &nbsp;<span style="color:#27AE60;">● Full Start</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")

        col_l, col_r = st.columns([1.05, 0.95], gap="large")
        with col_l:
            st.markdown(f'<div class="sec-header">Top {n_recs} Recommendations — via {model_choice}</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Inter',sans-serif;font-size:0.74rem;color:#BA9D74;
                margin-bottom:0.8rem;padding:0.5rem 0.9rem;
                background:rgba(73,38,102,0.28);border-radius:7px;
                border-left:2px solid rgba(217,167,97,0.40);">
                🎲 Mode: <b style="color:#D9A761;">{rec_mode}-Start</b>
                &middot; History: <b style="color:#D9A761;">{n_hist} courses</b>
            </div>""", unsafe_allow_html=True)
            
            if hybrid_model is None and model_choice == "Hybrid":
                st.error("Hybrid model failed to load. Ensure 'data/processed' contains trained model files.")
            elif recs is None or len(recs) == 0:
                st.warning("No recommendations found. Try another user, course, or model.")
            else:
                html = "".join(render_rec_card(i+1, row, i*70) for i,(_,row) in enumerate(recs.iterrows()))
                st.markdown(html, unsafe_allow_html=True)

        with col_r:
            st.markdown('<div class="sec-header">Multi-Dimensional Profile</div>', unsafe_allow_html=True)
            if not recs.empty:
                st.plotly_chart(chart_radar(recs), use_container_width=True)
            if not user_hist.empty:
                st.markdown('<div class="sec-header" style="margin-top:0.8rem;">Scholar\'s History</div>', unsafe_allow_html=True)
                with st.expander("View Courses Taken", expanded=False):
                    disp = user_hist.rename(columns={"course_name":"Course","category":"Domain","difficulty_level":"Level","rating":"Rating"})
                    disp["Rating"] = disp["Rating"].apply(lambda x: f"{x:.1f} ★")
                    st.dataframe(disp[["Course","Domain","Level","Rating"]], hide_index=True, use_container_width=True)

    # ════ TAB 2 — COURSE SIMILARITY ════
    with tab2:
        st.markdown('<div class="sec-header">🔗 Layer 2 — Semantic Course Similarity</div>', unsafe_allow_html=True)
        st.caption(
            "ℹ️ Semantic Similarity uses **MiniLM sentence embeddings** to measure meaning-level closeness between course titles. "
            "This is distinct from the Content-Based model inside the Hybrid system, which uses engineered numerical features "
            "(rating, price, difficulty, etc.) for scoring."
        )

        ctrl_col, n_col = st.columns([3, 1], gap="medium")
        with ctrl_col:
            selected_course = st.selectbox(
                "Select a course to find similar ones",
                all_courses,
                index=0,
                key="tab2_course_select",
            )
        with n_col:
            n_similar = st.slider("Top-N Similar", 3, 10, 5, key="tab2_n_similar")

        similar_courses = semantic_recommend(sim_df, courses, selected_course, n=n_similar)

        src_row = courses[courses["course_name"]==selected_course].iloc[0]
        src_cat = CAT_MAP.get(selected_course,"Other Domain")
        src_col_hex = CAT_COLORS.get(src_cat,"#7851A9")
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{_hex_rgba(src_col_hex,0.35)},rgba(73,38,102,0.55));
            border:1px solid {_hex_rgba(src_col_hex,0.55)};border-radius:12px;
            padding:1.1rem 1.4rem;margin-bottom:1.2rem;">
            <div style="font-family:'Inter',sans-serif;font-size:0.65rem;letter-spacing:0.14em;
                text-transform:uppercase;color:#BA9D74;margin-bottom:0.3rem;">🎯 Selected Source Course</div>
            <div style="font-family:'EB Garamond',serif;font-size:1.35rem;font-weight:600;
                color:#E5C5A3;margin-bottom:0.3rem;">{selected_course}</div>
            <div style="font-family:'Inter',sans-serif;font-size:0.78rem;color:#C8B089;">
                {diff_badge(src_row["difficulty"])} {cat_badge(src_cat)}
                &middot; ⭐ {src_row["avg_rating"]:.2f}
                &middot; ⏱ {src_row["avg_duration"]:.0f}h
                &middot; 💰 ${src_row["avg_price"]:.0f}
                &middot; 👥 {int(src_row["avg_enrollment"]):,} avg enrolments
            </div>
        </div>""", unsafe_allow_html=True)

        col_l2, col_r2 = st.columns([1, 1], gap="large")
        with col_l2:
            st.markdown(f'<div class="sec-header">Top {n_similar} Semantically Similar Courses</div>', unsafe_allow_html=True)
            if sim_source == "semantic":
                _sim_label = "MiniLM-L6-v2 semantic embeddings (all-MiniLM-L6-v2 · 400-course catalogue)"
            else:
                _sim_label = "Numeric cosine similarity on feature vectors: rating · price · enrolment · duration · difficulty · engagement"
            st.markdown(f"""
            <div style="font-family:'Inter',sans-serif;font-size:0.73rem;color:#BA9D74;
                margin-bottom:0.8rem;font-style:italic;">
                {_sim_label}
            </div>""", unsafe_allow_html=True)
            if similar_courses.empty:
                st.warning("No similar courses found.")
            else:
                sim_html = "".join(render_sim_card(i+1, row) for i,(_,row) in enumerate(similar_courses.iterrows()))
                st.markdown(sim_html, unsafe_allow_html=True)

        with col_r2:
            st.markdown('<div class="sec-header">Full Similarity Matrix</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_sim_matrix(sim_df, courses), use_container_width=True)

        st.markdown('<div class="royal-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-header">Feature Comparison — Source vs Similar</div>', unsafe_allow_html=True)
        if not similar_courses.empty:
            cmp_rows  = [src_row] + [similar_courses.iloc[i] for i in range(min(4, len(similar_courses)))]
            cmp_names = [selected_course] + similar_courses["course_name"].head(4).tolist()
            feat_cols = ["avg_rating","avg_price","avg_duration","avg_feedback","avg_eng"]
            feat_lbl  = ["Rating","Price","Duration","Feedback","Engagement"]
            pal2 = ["#D9A761","#079999","#7851A9","#A424CC","#C85A29"]
            fig_cmp = go.Figure()
            for fi,(fc,fl) in enumerate(zip(feat_cols,feat_lbl)):
                raw_v   = np.array([[r[fc] for r in cmp_rows]]).T
                scaled  = MinMaxScaler().fit_transform(raw_v).flatten()
                fig_cmp.add_trace(go.Bar(
                    name=fl,
                    x=[n[:22]+"…" if len(n)>22 else n for n in cmp_names],
                    y=scaled, marker_color=pal2[fi%len(pal2)],
                    marker_line_color="#D9A761", marker_line_width=0.5,
                    hovertemplate=f"<b>%{{x}}</b><br>{fl}: %{{y:.3f}}<extra></extra>",
                ))
            fig_cmp.update_layout(
                **PT,
                title=dict(text="Normalised Feature Comparison",font=dict(size=13,color="#D9A761"),x=0.5),
                barmode="group", bargap=0.2,
                xaxis=dict(tickfont=dict(size=8,color="#C8B089"),tickangle=20),
                yaxis=dict(title="Normalised Score (0–1)",tickfont=dict(size=9),gridcolor="rgba(217,167,97,0.10)"),
                legend=dict(font=dict(size=9),bgcolor="rgba(73,38,102,0.65)",bordercolor="#D9A761",borderwidth=1),
                height=340,
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

    # ════ TAB 3 — DOMAIN INTELLIGENCE ════
    with tab3:
        st.markdown(f'<div class="sec-header">🗺️ Scholar #{uid_input} — Domain Intelligence</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 1], gap="large")
        with col_a:
            st.markdown('<div class="sec-header">Your Domain Landscape & Blind Spots</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_sunburst(user_hist, courses), use_container_width=True)

        with col_b:
            st.markdown('<div class="sec-header">Your Completed Courses — Engagement Breakdown</div>', unsafe_allow_html=True)
            if user_hist.empty:
                st.info("No history yet for this scholar. Recommendations are popularity-based.")
            else:
                udf_full = df_raw[df_raw["user_id"] == uid_input].drop_duplicates("course_name")
                fig_usr = go.Figure()
                fig_usr.add_trace(go.Bar(
                    name="Your Rating",
                    x=udf_full["course_name"].apply(lambda x: x[:20]+"..." if len(x)>20 else x),
                    y=udf_full["rating"],
                    marker_color="#D9A761",
                    marker_line_color="#BA9D74", marker_line_width=0.6,
                    hovertemplate="<b>%{x}</b><br>Rating: %{y:.1f}★<extra></extra>",
                ))
                fig_usr.add_trace(go.Bar(
                    name="Hours Spent",
                    x=udf_full["course_name"].apply(lambda x: x[:20]+"..." if len(x)>20 else x),
                    y=udf_full["time_spent_hours"],
                    marker_color="#7851A9",
                    marker_line_color="#BA9D74", marker_line_width=0.6,
                    hovertemplate="<b>%{x}</b><br>Hours: %{y:.1f}h<extra></extra>",
                ))
                fig_usr.update_layout(
                    **PT,
                    title=dict(text=f"Scholar #{uid_input} Course History",font=dict(size=13,color="#D9A761"),x=0.5),
                    barmode="group", bargap=0.25,
                    xaxis=dict(tickfont=dict(size=8,color="#C8B089"),tickangle=30,gridcolor="rgba(217,167,97,0.10)"),
                    yaxis=dict(tickfont=dict(size=9),gridcolor="rgba(217,167,97,0.10)"),
                    legend=dict(font=dict(size=9),bgcolor="rgba(73,38,102,0.65)",bordercolor="#D9A761",borderwidth=1),
                    height=380,
                )
                st.plotly_chart(fig_usr, use_container_width=True)

        st.markdown('<div class="royal-divider"></div>', unsafe_allow_html=True)

        col_c, col_d = st.columns([1, 1], gap="large")
        with col_c:
            st.markdown('<div class="sec-header">Your Interest Profile by Domain</div>', unsafe_allow_html=True)
            if user_hist.empty:
                st.info("Complete at least one course to see your interest profile.")
            else:
                cat_counts = user_hist["category"].value_counts().reset_index()
                cat_counts.columns = ["category", "courses_taken"]
                cat_cols = [CAT_COLORS.get(c,"#7851A9") for c in cat_counts["category"]]
                fig_cat = go.Figure(go.Bar(
                    x=cat_counts["category"], y=cat_counts["courses_taken"],
                    marker_color=cat_cols, marker_line_color="#D9A761", marker_line_width=0.7,
                    hovertemplate="<b>%{x}</b><br>Courses Taken: %{y}<extra></extra>",
                ))
                fig_cat.update_layout(
                    **PT,
                    title=dict(text=f"Scholar #{uid_input} Domain Interests",font=dict(size=13,color="#D9A761"),x=0.5),
                    xaxis=dict(tickfont=dict(size=9,color="#C8B089"),tickangle=25,gridcolor="rgba(217,167,97,0.10)"),
                    yaxis=dict(title="Courses Taken",tickfont=dict(size=9),gridcolor="rgba(217,167,97,0.10)"),
                    height=340,
                )
                st.plotly_chart(fig_cat, use_container_width=True)

        with col_d:
            st.markdown('<div class="sec-header">Price vs Rating — Enrolment Universe</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_scatter(courses), use_container_width=True)

        st.markdown('<div class="royal-divider"></div>', unsafe_allow_html=True)

        col_e, col_f = st.columns([1, 1], gap="large")
        with col_e:
            st.markdown('<div class="sec-header">Category Intelligence Matrix</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_cat_bar(courses), use_container_width=True)
        with col_f:
            st.markdown('<div class="sec-header">Domain Influence Treemap</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_treemap(courses), use_container_width=True)

    # ════ TAB 4 — MODEL EVALUATION ════
    with tab4:
        st.markdown('<div class="sec-header">Offline Evaluation — Course Family Tier (k=10, 500 users)</div>', unsafe_allow_html=True)
        model_labels = {"hybrid":"Hybrid","content":"Content","collaborative":"Collab","popularity":"Popularity","knn":"KNN"}
        model_colors = {"hybrid":"#D9A761","content":"#7851A9","collaborative":"#079999","popularity":"#A424CC","knn":"#C85A29"}
        tier = "course_family"
        m_cols = st.columns(5)
        for mc,(mdl,lbl) in zip(m_cols, model_labels.items()):
            ndcg = metrics["models"][mdl][tier]["ndcg_at_k"]
            hr   = metrics["models"][mdl][tier]["hit_rate_at_k"]
            mpc  = metrics["models"][mdl][tier]["map_at_k"]
            cov  = metrics["catalogue_coverage"].get(mdl,0)
            col  = model_colors[mdl]
            with mc:
                st.markdown(f"""
                <div class="stat-tile" style="border-color:{_hex_rgba(col,0.55)}">
                    <div class="stat-label" style="color:{col};">{lbl}</div>
                    <div style="font-family:'Space Mono',monospace;font-size:1.4rem;color:#E5C5A3;font-weight:700;">{ndcg:.4f}</div>
                    <div class="stat-sub">NDCG@10</div>
                    <div class="stat-sub">HR@10: {hr:.3f} &middot; MAP: {mpc:.3f}</div>
                    <div class="stat-sub">Coverage: {cov:.1%}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="royal-divider"></div>', unsafe_allow_html=True)
        col_e,col_f = st.columns([1,1], gap="large")
        with col_e:
            st.plotly_chart(chart_eval_heatmap(metrics), use_container_width=True)
        with col_f:
            st.plotly_chart(chart_model_bars(metrics), use_container_width=True)

        st.markdown('<div class="royal-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-header">Full Metrics Table — All Tiers</div>', unsafe_allow_html=True)
        tiers_all = [t for t in ["exact_course","course_family","soft_relevance"]
                     if any(t in metrics["models"][m] for m in model_labels)]
        rows = []
        for mdl,lbl in model_labels.items():
            for t in tiers_all:
                if t not in metrics["models"][mdl]:
                    continue
                m = metrics["models"][mdl][t]
                rows.append({"Model":lbl,"Tier":t.replace("_"," ").title(),
                    "Precision@10":round(m.get("precision_at_k",0),4),"Recall@10":round(m.get("recall_at_k",0),4),
                    "MAP@10":round(m.get("map_at_k",0),4),"NDCG@10":round(m.get("ndcg_at_k",0),4),
                    "HitRate@10":round(m.get("hit_rate_at_k",0),4),})
        st.dataframe(pd.DataFrame(rows).sort_values(["Tier","NDCG@10"],ascending=[True,False]),
                     hide_index=True, use_container_width=True, height=350)

        with st.expander("Evaluation Protocol Details"):
            p = metrics["protocol"]
            st.markdown(f"""
            <div style="font-family:'Inter',sans-serif;font-size:0.80rem;color:#C8B089;line-height:1.9;">
                <b style="color:#D9A761;">Protocol:</b> {p['type']}<br>
                <b style="color:#D9A761;">Relevance Tiers:</b> {', '.join(p['relevance_tiers'])}<br>
                <b style="color:#D9A761;">k:</b> {p['k']} &nbsp;&nbsp; <b style="color:#D9A761;">Evaluated Users:</b> {p['evaluated_users']}<br>
                <b style="color:#D9A761;">Note:</b> {p['note']}<br>
                <b style="color:#D9A761;">Interest Profile Coverage:</b> {metrics['interest_profile_coverage']:.1%}
            </div>""", unsafe_allow_html=True)

    # ════ TAB 5 — DATA EXPLORER ════
    with tab5:
        st.markdown('<div class="sec-header">Course Catalogue — Complete Codex</div>', unsafe_allow_html=True)
        col_f1,col_f2 = st.columns(2)
        with col_f1: filt_cat2 = st.selectbox("Domain", ["All"]+sorted(set(CAT_MAP.values())), key="t5_cat")
        with col_f2: filt_dif2 = st.selectbox("Difficulty", ["All","Beginner","Intermediate","Advanced"], key="t5_diff")
        codex = courses.copy()
        if filt_cat2!="All": codex = codex[codex["category"]==filt_cat2]
        if filt_dif2!="All": codex = codex[codex["difficulty"]==filt_dif2]
        disp_c = codex[["course_name","category","difficulty","avg_rating","avg_price","avg_enrollment","avg_duration","avg_feedback","avg_eng","cert_offered","study_mat","user_count"]].copy()
        disp_c.columns = ["Course","Domain","Level","Rating","Price($)","Enrolment","Duration(h)","Feedback","Engagement","Cert","Study Mat","Learners"]
        disp_c["Rating"]     = disp_c["Rating"].round(3)
        disp_c["Price($)"]   = disp_c["Price($)"].round(0).astype(int)
        disp_c["Enrolment"]  = disp_c["Enrolment"].round(0).astype(int)
        disp_c["Duration(h)"]= disp_c["Duration(h)"].round(1)
        disp_c["Feedback"]   = disp_c["Feedback"].round(3)
        disp_c["Engagement"] = disp_c["Engagement"].round(3)
        disp_c["Cert"]       = disp_c["Cert"].apply(lambda x: "✅" if x else "❌")
        disp_c["Study Mat"]  = disp_c["Study Mat"].apply(lambda x: "✅" if x else "❌")
        disp_c["Learners"]   = disp_c["Learners"].astype(int)
        st.dataframe(disp_c.sort_values("Rating",ascending=False), hide_index=True, use_container_width=True, height=440)

        st.markdown('<div class="royal-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-header">User History Browser</div>', unsafe_allow_html=True)
        uid_e = st.number_input("Browse User ID", min_value=int(all_users[0]), max_value=int(all_users[-1]), value=uid_input, step=1, key="t5_uid")
        hist_e = get_user_history(df_raw, uid_e)
        if hist_e.empty:
            st.info("No interaction history for this user ID.")
        else:
            sh = hist_e.rename(columns={"course_name":"Course","category":"Domain","difficulty_level":"Level","rating":"Rating","time_spent_hours":"Time(h)","course_price":"Price($)"})
            sh["Rating"] = sh["Rating"].apply(lambda x: f"{x:.2f} ★")
            sh["Time(h)"]= sh["Time(h)"].apply(lambda x: f"{x:.1f}h")
            sh["Price($)"]= sh["Price($)"].apply(lambda x: f"${x:.0f}")
            st.dataframe(sh[["Course","Domain","Level","Rating","Time(h)","Price($)"]], hide_index=True, use_container_width=True)

        st.markdown('<div class="royal-divider"></div>', unsafe_allow_html=True)
        col_s1,col_s2 = st.columns([1,1], gap="large")
        with col_s1:
            st.markdown('<div class="sec-header">Dataset Statistics</div>', unsafe_allow_html=True)
            stats = {
                "Total Records": f"{len(df_raw):,}", "Total Users": f"{df_raw['user_id'].nunique():,}",
                "Total Courses": f"{total_courses:,}", "Avg Rating": f"{df_raw['rating'].mean():.3f} ★",
                "Avg Price": f"${df_raw['course_price'].mean():.0f}",
                "Avg Duration": f"{df_raw['course_duration_hours'].mean():.1f}h",
                "Avg Enrolment": f"{df_raw['enrollment_numbers'].mean():.0f}",
                "Avg Completion": f"{df_raw['completion'].mean():.1%}",
            }
            for i, (k, v) in enumerate(stats.items()):
                st.markdown(f"""
                <div class="stat-row-anim" style="animation-delay:{i*60}ms;display:flex;justify-content:space-between;align-items:center;
                    padding:0.35rem 0.7rem;margin:0.2rem 0;
                    background:rgba(73,38,102,0.28);border-radius:6px;
                    border-left:2px solid rgba(217,167,97,0.30);">
                    <span style="font-family:'Inter',sans-serif;font-size:0.76rem;color:#BA9D74;">{k}</span>
                    <span style="font-family:'Space Mono',monospace;font-size:0.82rem;color:#E5C5A3;font-weight:700;">{v}</span>
                </div>""", unsafe_allow_html=True)

        with col_s2:
            st.markdown('<div class="sec-header">Hybrid Ensemble Weights</div>', unsafe_allow_html=True)
            weight_modes = {
                "Full-Start (≥4 interactions)": {"Content":0.70,"Popularity":0.12,"Collaborative":0.08,"User Interest":0.07,"KNN":0.03},
                "Warm-Start (2–3 interactions)": {"Content":0.65,"Popularity":0.18,"User Interest":0.08,"Collaborative":0.06,"KNN":0.03},
                "Cold-Start (<2 interactions)":      {"Popularity":0.55,"Content":0.45},
            }
            for mode,wts in weight_modes.items():
                st.markdown(f"""
                <div style="font-family:'Inter',sans-serif;font-size:0.70rem;letter-spacing:0.10em;
                    text-transform:uppercase;color:#D9A761;margin:0.7rem 0 0.25rem;">{mode}</div>""", unsafe_allow_html=True)
                for i, (arm, w) in enumerate(wts.items()):
                    bar_pct = int(w*100)
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:0.6rem;margin:0.15rem 0;">
                        <span style="font-family:'Inter',sans-serif;font-size:0.73rem;color:#C8B089;width:130px;">{arm}</span>
                        <div style="flex:1;background:rgba(73,38,102,0.35);border-radius:3px;height:7px;overflow:hidden;">
                            <div style="--bar-w:{bar_pct}%;width:0;height:100%;border-radius:3px;background:linear-gradient(90deg,#7851A9,#D9A761);animation:weightBar 1s ease both {i*80}ms;"></div>
                        </div>
                        <span style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#D9A761;width:36px;text-align:right;">{w:.0%}</span>
                    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="royal-divider"></div>
    <div style="text-align:center;font-family:'Inter',sans-serif;font-size:0.68rem;
        color:#7851A9;letter-spacing:0.10em;padding-bottom:0.8rem;">
        <span id="footer-ticker">COURSEIQ &nbsp;&middot;&nbsp; ONLINE COURSE RECOMMENDATION SYSTEM &nbsp;&middot;&nbsp; INTEREST · ALS · KNN · MINILM &nbsp;&middot;&nbsp; v4.0</span>
    </div>
    <script>
    (function(){
        var el = document.getElementById('footer-ticker');
        if (!el || el.dataset.typed === '1') return;
        var full = el.innerHTML;
        el.innerHTML = '';
        el.dataset.typed = '1';
        var chars = Array.from(full);
        var i = 0;
        var t = setInterval(function(){
            if(i >= chars.length){ clearInterval(t); return; }
            el.innerHTML += chars[i++];
        }, 28);
    })();
    </script>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()