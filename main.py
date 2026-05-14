"""
main.py — Industrial AI Platform Entry Point
Run: streamlit run main.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np

from scripts.csv_loader    import load_csv, get_csv_summary
from scripts.retriever     import FAISSRetriever
from scripts.analytics     import compute_full_analytics
from scripts.utils         import detect_column_roles

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Industrial AI Platform",
    page_icon   = "🏭",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

:root {
    --bg:       #0a0e17;
    --surface:  #111827;
    --border:   #1e293b;
    --accent:   #00d4ff;
    --accent2:  #ff6b2b;
    --green:    #00ff88;
    --red:      #ff3b5c;
    --text:     #e2e8f0;
    --muted:    #64748b;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

.stApp { background: var(--bg) !important; }

.stSidebar { background: var(--surface) !important; border-right: 1px solid var(--border); }
.stSidebar .stMarkdown { color: var(--text) !important; }

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    border-left: 3px solid var(--accent);
}
.metric-card h3 { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 2px; margin: 0; }
.metric-card p  { font-size: 2rem; font-weight: 800; color: var(--accent); margin: 0.3rem 0 0; font-family: 'JetBrains Mono', monospace; }

.upload-zone {
    border: 2px dashed var(--accent);
    border-radius: 16px;
    padding: 3rem;
    text-align: center;
    background: rgba(0,212,255,0.03);
    margin: 2rem 0;
}

.status-ok    { color: var(--green); font-weight: 700; }
.status-warn  { color: #ffb800;      font-weight: 700; }
.status-err   { color: var(--red);   font-weight: 700; }

h1 { font-size: 2.5rem !important; font-weight: 800 !important; }
h2 { font-size: 1.6rem !important; font-weight: 700 !important; color: var(--accent); }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; }

.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.5rem 1.5rem !important;
    font-family: 'Syne', sans-serif !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s;
}
.stButton > button:hover { background: #00b8d9 !important; transform: translateY(-1px); }

.stProgress > div > div > div > div { background: var(--accent) !important; }

div[data-testid="stFileUploader"] {
    background: var(--surface);
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid var(--border);
}

.tag {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    color: var(--accent);
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ─── Session state init ───────────────────────────────────────────────────────
def init_session():
    defaults = {
        "df":           None,
        "fhash":        None,
        "col_roles":    None,
        "analytics":    None,
        "retriever":    None,
        "csv_summary":  None,
        "index_built":  False,
        "chat_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 Industrial AI")
    st.markdown("---")

    st.markdown("### 📂 Upload CSV Data")
    uploaded = st.file_uploader(
        "Upload maintenance CSV (up to 40k rows)",
        type=["csv"],
        help="Supports machine logs, maintenance records, error logs, cost data"
    )

    if uploaded and (st.session_state.fhash is None or
                     uploaded.name != st.session_state.get("uploaded_name")):

        st.session_state.uploaded_name = uploaded.name

        with st.spinner("📊 Loading and cleaning data..."):
            try:
                df, fhash = load_csv(uploaded)
                col_roles = detect_column_roles(df)
                analytics = compute_full_analytics(df)
                summary   = get_csv_summary(df, col_roles)

                st.session_state.df          = df
                st.session_state.fhash       = fhash
                st.session_state.col_roles   = col_roles
                st.session_state.analytics   = analytics
                st.session_state.csv_summary = summary
                st.session_state.index_built = False
                st.session_state.chat_history = []

                # Build FAISS index
                retriever = FAISSRetriever(fhash)
                if not retriever.load_if_cached():
                    st.info("🔨 Building semantic index... (first time only)")
                    progress_bar = st.progress(0)
                    retriever.build(df, col_roles, progress_cb=lambda p: progress_bar.progress(p))
                    progress_bar.empty()

                st.session_state.retriever   = retriever
                st.session_state.index_built = True
                st.success(f"✅ Indexed {len(df):,} rows")

            except Exception as e:
                st.error(f"❌ Error: {e}")

    st.markdown("---")

    # Status indicators
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown(f"""
        **Dataset Status**
        - 📋 Rows: `{len(df):,}`
        - 📊 Cols: `{len(df.columns)}`
        - 🔍 Index: {'✅ Ready' if st.session_state.index_built else '⏳ Building'}
        """)

        if st.session_state.col_roles:
            cr = st.session_state.col_roles
            if cr["machine_id"]:
                st.markdown(f"- 🤖 Machine col: `{cr['machine_id'][0]}`")
            if cr["datetime"]:
                st.markdown(f"- 📅 Date col: `{cr['datetime'][0]}`")
            if cr["numeric"]:
                st.markdown(f"- 🔢 Numeric: `{len(cr['numeric'])} cols`")
    else:
        st.info("👆 Upload a CSV to begin")

    st.markdown("---")
    st.markdown("""
    **Navigation**
    - 🏠 Home (this page)
    - 📊 [Dashboard](Dashboard)
    - 🤖 [AI Chat](AI_Chat)
    - 🔬 [Data Explorer](Data_Explorer)
    """)

    st.markdown("---")
    st.markdown("**Model:** `gemma:2b` via Ollama")
    st.markdown("**Embeddings:** `AllMiniLM-L6-v2`")


# ─── Main content ─────────────────────────────────────────────────────────────
st.markdown("# 🏭 Industrial Maintenance AI")
st.markdown("**Production-grade predictive maintenance intelligence platform**")

if st.session_state.df is None:
    # Landing / upload prompt
    st.markdown("""
    <div class="upload-zone">
        <h2 style="color:#00d4ff; margin-bottom:0.5rem;">Upload Your Maintenance Data</h2>
        <p style="color:#64748b; font-size:1.1rem;">
            Supports CSV files up to 40,000 rows — machine logs, error codes, maintenance records, cost data
        </p>
        <br/>
        <span class="tag">🔍 Semantic Search</span>
        <span class="tag">📈 Predictive Analytics</span>
        <span class="tag">💰 Cost Analysis</span>
        <span class="tag">🤖 AI Root Cause</span>
        <span class="tag">👨‍🔧 Technician Insights</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Semantic Retrieval</h3>
            <p style="font-size:1rem; color:#e2e8f0;">AllMiniLM-L6-v2 embeds every row — ask anything in plain English</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>AI Generation</h3>
            <p style="font-size:1rem; color:#e2e8f0;">Gemma:2b answers from your data only — zero hallucination</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Industrial KPIs</h3>
            <p style="font-size:1rem; color:#e2e8f0;">MTBF, MTTR, failure rates, cost breakdown, health scores</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Dataset loaded — show overview
    df        = st.session_state.df
    analytics = st.session_state.analytics

    st.markdown("## 📋 Dataset Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    def metric_html(label, value, color="#00d4ff"):
        return f"""<div class="metric-card">
            <h3>{label}</h3>
            <p style="color:{color};">{value}</p>
        </div>"""

    machines = len(df[st.session_state.col_roles["machine_id"][0]].unique()) \
        if st.session_state.col_roles["machine_id"] else "N/A"
    faults   = analytics.get("failure_analysis", {}).get("status_distribution", {})
    total_faults = sum(faults.values()) if faults else "N/A"

    cost_total = analytics.get("cost_analysis", {}).get("total_cost", None)
    cost_str   = f"${cost_total:,.0f}" if cost_total else "N/A"

    c1.markdown(metric_html("Total Records", f"{len(df):,}"), unsafe_allow_html=True)
    c2.markdown(metric_html("Machines", str(machines), "#ff6b2b"), unsafe_allow_html=True)
    c3.markdown(metric_html("Fault Events", str(total_faults), "#ff3b5c"), unsafe_allow_html=True)
    c4.markdown(metric_html("Total Cost", cost_str, "#00ff88"), unsafe_allow_html=True)
    c5.markdown(metric_html("Columns", str(len(df.columns)), "#a78bfa"), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🗂️ Detected Column Roles")

    cr = st.session_state.col_roles
    cols_display = st.columns(4)
    role_icons = {"machine_id":"🤖","datetime":"📅","numeric":"🔢",
                  "status":"⚠️","technician":"👨‍🔧","description":"📝","other":"📌"}
    role_colors = {"machine_id":"#00d4ff","datetime":"#a78bfa","numeric":"#00ff88",
                   "status":"#ff3b5c","technician":"#ff6b2b","description":"#64748b","other":"#334155"}

    for i, (role, cols_) in enumerate(cr.items()):
        if cols_:
            with cols_display[i % 4]:
                icon  = role_icons.get(role, "📌")
                color = role_colors.get(role, "#64748b")
                tags  = "".join(f'<span class="tag" style="border-color:{color};color:{color};">{c}</span>' for c in cols_)
                st.markdown(f"**{icon} {role.replace('_',' ').title()}**<br>{tags}",
                            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 👀 Sample Data (first 50 rows)")
    st.dataframe(df.head(50), use_container_width=True, height=300)

    st.markdown("---")
    st.info("📊 Go to **Dashboard** for full analytics → **AI Chat** to query your data with AI")