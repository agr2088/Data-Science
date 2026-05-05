"""
utils/theme.py
Royal Black + Gold — CSS injection, section headers, KPI cards, dividers.
No changes needed from original; reproduced clean.
"""
import streamlit as st


SECTION_ICONS = {
    "HERO":           "✦",
    "INTELLIGENCE":   "◆",
    "WORLD MAP":      "🌍",
    "CLUSTER ATLAS":  "◈",
    "COUNTRY LENS":   "◉",
    "MODEL ARENA":    "★",
    "FEATURE VAULT":  "◇",
}

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


def inject_global_css():
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

    <style>
    /* ── CSS Variables ──────────────────────────────────────────── */
    :root {
      --bg-void:       #050507;
      --bg-base:       #0A0A0F;
      --bg-surface:    #111118;
      --bg-elevated:   #18181F;
      --bg-hover:      #1E1E28;
      --gold-100:      #FFF3C4;
      --gold-300:      #FFD966;
      --gold-500:      #D4A017;
      --gold-700:      #A67C00;
      --gold-900:      #5C4300;
      --text-primary:  #F5ECD7;
      --text-secondary:#B8A87A;
      --text-muted:    #5A5040;
      --cluster-0:     #D4A017;
      --cluster-1:     #7EC8A4;
      --cluster-2:     #6EB3D4;
      --cluster-3:     #C97B5E;
      --border-line:   1px solid #2A2415;
      --border-gold:   1px solid #A67C00;
      --border-glow:   0 0 12px rgba(212,160,23,0.25);
      --radius-sm:     6px;
      --radius-md:     12px;
      --radius-lg:     20px;
      --font-display:  'Cinzel', serif;
      --font-body:     'Inter', sans-serif;
      --font-mono:     'JetBrains Mono', monospace;
    }

    /* ── Reset & Base ───────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {
      background-color: var(--bg-base) !important;
      color: var(--text-primary) !important;
      font-family: var(--font-body) !important;
    }
    [data-testid="stAppViewContainer"] > .main {
      background-color: var(--bg-base) !important;
    }
    .block-container {
      padding: 0 !important;
      max-width: 100% !important;
    }

    /* ── Hide Streamlit chrome ──────────────────────────────────── */
    #MainMenu, footer, header { display: none !important; }
    [data-testid="stToolbar"]      { display: none !important; }
    [data-testid="stDecoration"]   { display: none !important; }
    [data-testid="stStatusWidget"] { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }

    /* ── Scrollbar ──────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: var(--bg-void); }
    ::-webkit-scrollbar-thumb { background: var(--gold-900); border-radius: 2px; }

    /* ── HERO ───────────────────────────────────────────────────── */
    .hero-wrap {
      text-align: center;
      padding: 72px 24px 48px;
      background: radial-gradient(ellipse 80% 60% at 50% 0%,
                  rgba(212,160,23,0.08) 0%, transparent 70%);
      border-bottom: var(--border-gold);
    }
    .hero-crown {
      font-size: 28px;
      color: var(--gold-500);
      margin-bottom: 12px;
      display: block;
    }
    .hero-title {
      font-family: var(--font-display);
      font-size: clamp(24px, 4vw, 52px);
      letter-spacing: 0.12em;
      color: var(--gold-300);
      text-shadow: 0 0 40px rgba(212,160,23,0.4);
      margin: 0;
      line-height: 1.2;
    }
    .hero-tagline {
      color: var(--text-secondary);
      font-size: 12px;
      letter-spacing: 0.25em;
      text-transform: uppercase;
      margin: 12px 0 28px;
      font-family: var(--font-mono);
    }
    .hero-pills {
      display: flex;
      justify-content: center;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }
    .pill {
      display: inline-block;
      padding: 4px 14px;
      border: var(--border-gold);
      border-radius: 100px;
      font-size: 11px;
      color: var(--gold-300);
      letter-spacing: 0.1em;
      font-family: var(--font-mono);
      background: rgba(164,124,0,0.08);
    }
    .hero-badges {
      display: flex;
      justify-content: center;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }
    .badge {
      display: inline-block;
      background: var(--bg-elevated);
      padding: 3px 10px;
      border-radius: var(--radius-sm);
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-secondary);
      border: var(--border-line);
    }

    /* ── COMMAND DECK ───────────────────────────────────────────── */
    .command-deck {
      background: var(--bg-surface);
      border-bottom: var(--border-gold);
      padding: 14px 32px;
    }
    .cmd-label {
      font-family: var(--font-mono);
      font-size: 9px;
      letter-spacing: 0.25em;
      color: var(--gold-700);
      text-transform: uppercase;
      margin: 0 0 4px 0;
    }

    /* ── Streamlit widget overrides ─────────────────────────────── */
    .stRadio > div { gap: 4px; }
    .stRadio label {
      color: var(--text-secondary) !important;
      font-family: var(--font-mono) !important;
      font-size: 12px !important;
    }
    .stSelectbox > div > div {
      background: var(--bg-elevated) !important;
      border-color: #2A2415 !important;
      color: var(--text-primary) !important;
      font-family: var(--font-body) !important;
    }
    .stMultiSelect > div > div {
      background: var(--bg-elevated) !important;
      border-color: #2A2415 !important;
    }
    .stTextInput > div > div > input {
      background: var(--bg-elevated) !important;
      border-color: #2A2415 !important;
      color: var(--text-primary) !important;
      font-family: var(--font-body) !important;
    }
    .stTextInput > div > div > input:focus {
      border-color: var(--gold-700) !important;
      box-shadow: 0 0 0 1px var(--gold-900) !important;
    }
    .stTabs [data-baseweb="tab-list"] {
      background: var(--bg-surface) !important;
      border-bottom: var(--border-gold) !important;
      gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
      background: transparent !important;
      color: var(--text-secondary) !important;
      font-family: var(--font-mono) !important;
      font-size: 11px !important;
      letter-spacing: 0.1em !important;
      padding: 8px 20px !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
      color: var(--gold-300) !important;
      border-bottom: 2px solid var(--gold-500) !important;
    }
    .stExpander {
      background: var(--bg-surface) !important;
      border: var(--border-line) !important;
      border-radius: var(--radius-md) !important;
    }
    .stExpander summary {
      color: var(--gold-300) !important;
      font-family: var(--font-mono) !important;
      font-size: 12px !important;
      letter-spacing: 0.1em !important;
    }

    /* ── KPI CARDS ──────────────────────────────────────────────── */
    .kpi-card {
      background: var(--bg-elevated);
      border: var(--border-line);
      border-top: 2px solid var(--gold-700);
      border-radius: var(--radius-md);
      padding: 18px 16px 14px;
      text-align: center;
      transition: box-shadow 0.2s;
      min-height: 115px;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }
    .kpi-card:hover { box-shadow: var(--border-glow); }
    .kpi-icon { font-size: 18px; color: var(--gold-500); margin-bottom: 4px; }
    .kpi-label {
      font-family: var(--font-mono);
      font-size: 9px;
      letter-spacing: 0.2em;
      color: var(--gold-700);
      text-transform: uppercase;
    }
    .kpi-value {
      font-family: var(--font-display);
      font-size: 20px;
      color: var(--gold-300);
      margin: 4px 0 2px;
      line-height: 1.2;
      word-break: break-word;
    }
    .kpi-sub { font-size: 10px; color: var(--text-muted); font-family: var(--font-mono); }
    .intel-chart-wrap { margin-top: 16px; }

    /* ── SECTION HEADERS ────────────────────────────────────────── */
    .section-header { padding: 32px 32px 12px; }
    .section-title {
      font-family: var(--font-display);
      font-size: 18px;
      letter-spacing: 0.2em;
      color: var(--gold-300);
      margin: 0;
      text-transform: uppercase;
    }
    .section-sub {
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--gold-700);
      letter-spacing: 0.15em;
      margin: 4px 0 0;
      text-transform: uppercase;
    }

    /* ── GOLD DIVIDER ───────────────────────────────────────────── */
    .gold-divider {
      height: 1px;
      background: linear-gradient(90deg,
        transparent 0%, #A67C00 30%, #FFD966 50%, #A67C00 70%, transparent 100%);
      margin: 24px 32px;
    }

    /* ── CLUSTER CARDS ──────────────────────────────────────────── */
    .cluster-card {
      background: var(--bg-elevated);
      border: var(--border-line);
      border-radius: var(--radius-md);
      padding: 20px 16px;
      height: 100%;
      transition: box-shadow 0.2s;
    }
    .cluster-card:hover { box-shadow: var(--border-glow); }
    .cc-id {
      font-family: var(--font-mono);
      font-size: 10px;
      letter-spacing: 0.2em;
      text-transform: uppercase;
      margin-bottom: 4px;
    }
    .cc-name {
      font-family: var(--font-display);
      font-size: 13px;
      color: var(--text-primary);
      margin-bottom: 14px;
      line-height: 1.3;
    }
    .cc-stat {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 5px 0;
      border-bottom: var(--border-line);
    }
    .cc-stat:last-child { border-bottom: none; }
    .cc-stat-label {
      font-family: var(--font-mono);
      font-size: 9px;
      color: var(--text-muted);
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }
    .cc-stat-value {
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--text-primary);
    }

    /* ── COUNTRY BANNER ─────────────────────────────────────────── */
    .country-banner {
      background: var(--bg-elevated);
      border: var(--border-line);
      border-left: 4px solid;
      border-radius: var(--radius-md);
      padding: 20px 24px;
      margin: 0 32px 20px;
      display: flex;
      align-items: center;
      gap: 24px;
      flex-wrap: wrap;
    }
    .cb-country {
      font-family: var(--font-display);
      font-size: 22px;
      color: var(--text-primary);
    }
    .cb-cluster {
      font-family: var(--font-mono);
      font-size: 12px;
      letter-spacing: 0.1em;
    }
    .cb-agree {
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-muted);
      margin-left: auto;
    }
    .no-result {
      background: var(--bg-elevated);
      border: var(--border-line);
      border-radius: var(--radius-md);
      padding: 20px 24px;
      margin: 0 32px 20px;
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--text-muted);
      text-align: center;
    }

    /* ── MODEL MEMBERSHIP TABLE ─────────────────────────────────── */
    .membership-wrap {
      background: var(--bg-elevated);
      border: var(--border-line);
      border-radius: var(--radius-md);
      padding: 16px;
    }
    .section-micro {
      font-family: var(--font-mono);
      font-size: 9px;
      letter-spacing: 0.25em;
      color: var(--gold-700);
      text-transform: uppercase;
      margin: 0 0 10px 0;
    }
    .membership-table {
      width: 100%;
      border-collapse: collapse;
    }
    .membership-table td {
      padding: 6px 4px;
      font-family: var(--font-mono);
      font-size: 11px;
      border-bottom: var(--border-line);
    }
    .membership-table tr:last-child td { border-bottom: none; }
    .mm-model { color: var(--text-secondary); width: 40%; }
    .mm-badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: var(--radius-sm);
      font-size: 10px;
    }
    .majority-label {
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--text-muted);
      margin-top: 10px;
      padding-top: 8px;
      border-top: var(--border-line);
    }
    .majority-label strong { color: var(--gold-300); }

    /* ── MODEL ARENA ROWS ───────────────────────────────────────── */
    .model-row {
      display: flex;
      align-items: center;
      gap: 0;
      padding: 10px 0;
      border-bottom: var(--border-line);
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--text-secondary);
    }
    .model-row--winner { color: var(--gold-300); }
    .model-row--invalid { opacity: 0.45; }
    .mr-name   { flex: 2; color: inherit; font-size: 13px; }
    .mr-metric { flex: 1.2; text-align: right; font-size: 12px; }
    .badge-valid {
      display: inline-block;
      padding: 2px 8px;
      border-radius: var(--radius-sm);
      font-size: 10px;
      background: rgba(126,200,164,0.15);
      border: 1px solid #7EC8A4;
      color: #7EC8A4;
      flex: 1;
      text-align: center;
    }
    .badge-invalid {
      display: inline-block;
      padding: 2px 8px;
      border-radius: var(--radius-sm);
      font-size: 10px;
      background: rgba(201,123,94,0.15);
      border: 1px solid #C97B5E;
      color: #C97B5E;
      flex: 1;
      text-align: center;
    }
    .badge-winner {
      display: inline-block;
      padding: 2px 8px;
      border-radius: var(--radius-sm);
      font-size: 10px;
      background: rgba(212,160,23,0.15);
      border: 1px solid var(--gold-700);
      color: var(--gold-300);
      flex: 1;
      text-align: center;
    }
    .arena-header {
      display: flex;
      align-items: center;
      gap: 0;
      padding: 8px 0;
      font-family: var(--font-mono);
      font-size: 9px;
      letter-spacing: 0.15em;
      color: var(--gold-700);
      text-transform: uppercase;
      border-bottom: var(--border-gold);
      margin-bottom: 4px;
    }
    .ah-col { flex: 1.2; text-align: right; }
    .ah-col:first-child { flex: 2; text-align: left; }

    /* ── PADDING WRAPPERS ───────────────────────────────────────── */
    .padded { padding: 0 32px; }
    </style>
    """, unsafe_allow_html=True)


def render_section_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div class="section-header">
      <h2 class="section-title">{title}</h2>
      {'<p class="section-sub">' + subtitle + '</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def render_gold_divider():
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)


def render_kpi_card(icon: str, label: str, value: str, sub: str):
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)