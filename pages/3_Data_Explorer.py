"""
pages/3_Data_Explorer.py — Interactive data explorer with filters, 
machine-level drill-down, and AI report generation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scripts.generator import generate_machine_report
st.set_page_config(page_title="Data Explorer | Industrial AI", page_icon="🔬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
:root { --bg:#0a0e17; --surface:#111827; --border:#1e293b; --accent:#00d4ff; --text:#e2e8f0; --muted:#64748b; }
html,body,[class*="css"] { background-color:var(--bg) !important; color:var(--text) !important; font-family:'Syne',sans-serif; }
.stApp { background:var(--bg) !important; }
h1 { font-size:2rem !important; font-weight:800 !important; }
h2 { font-size:1.4rem !important; color:var(--accent) !important; }
.filter-panel { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:1rem; margin-bottom:1rem; }
.stButton > button { background:var(--accent) !important; color:#000 !important; font-weight:700 !important; border-radius:8px !important; border:none !important; font-family:'Syne',sans-serif !important; }
.report-box { background:#0f172a; border:1px solid #1e293b; border-radius:12px; padding:1.5rem; font-size:0.9rem; line-height:1.7; white-space:pre-wrap; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = {
    "paper_bgcolor": "#0a0e17",
    "plot_bgcolor":  "#111827",
    "font":          {"color": "#e2e8f0", "family": "Syne"},
    "xaxis":         {"gridcolor": "#1e293b"},
    "yaxis":         {"gridcolor": "#1e293b"},
}

# ─── Guard ────────────────────────────────────────────────────────────────────
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ Upload a CSV on the Home page first.")
    st.stop()

df        = st.session_state.df
col_roles = st.session_state.col_roles
analytics = st.session_state.analytics

st.markdown("# 🔬 Data Explorer")

# ─── Sidebar filters ──────────────────────────────────────────────────────────
st.markdown("## 🎛️ Filters")

filter_cols = st.columns(4)
filtered_df = df.copy()

# Machine filter
mid_col = col_roles["machine_id"][0] if col_roles["machine_id"] else None
if mid_col:
    machines = ["All"] + sorted(df[mid_col].dropna().astype(str).unique().tolist())
    sel_machine = filter_cols[0].selectbox("Machine ID", machines)
    if sel_machine != "All":
        filtered_df = filtered_df[filtered_df[mid_col].astype(str) == sel_machine]

# Status filter
status_col = col_roles["status"][0] if col_roles["status"] else None
if status_col:
    statuses = ["All"] + sorted(df[status_col].dropna().astype(str).unique().tolist())
    sel_status = filter_cols[1].selectbox("Status / Fault Type", statuses)
    if sel_status != "All":
        filtered_df = filtered_df[filtered_df[status_col].astype(str) == sel_status]

# Date filter
dt_col = col_roles["datetime"][0] if col_roles["datetime"] else None
if dt_col:
    try:
        dt_min = df[dt_col].dropna().min().date()
        dt_max = df[dt_col].dropna().max().date()
        sel_dates = filter_cols[2].date_input("Date Range", value=(dt_min, dt_max))
        if len(sel_dates) == 2:
            filtered_df = filtered_df[
                (filtered_df[dt_col].dt.date >= sel_dates[0]) &
                (filtered_df[dt_col].dt.date <= sel_dates[1])
            ]
    except:
        pass

# Tech filter
tech_col = col_roles["technician"][0] if col_roles["technician"] else None
if tech_col:
    techs = ["All"] + sorted(df[tech_col].dropna().astype(str).unique().tolist())
    sel_tech = filter_cols[3].selectbox("Technician", techs)
    if sel_tech != "All":
        filtered_df = filtered_df[filtered_df[tech_col].astype(str) == sel_tech]

# Free-text search
st.markdown("")
search_q = st.text_input("🔍 Free-text filter (searches all columns)", placeholder="Type to filter rows...")
if search_q.strip():
    mask = df.astype(str).apply(lambda col: col.str.contains(search_q, case=False, na=False)).any(axis=1)
    filtered_df = filtered_df[mask]

st.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} rows**")
st.markdown("---")

# ─── Data table ───────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Data Table", "📈 Column Visualizer", "🤖 Machine AI Report"])

with tab1:
    page_size = st.selectbox("Rows per page", [50, 100, 250, 500, 1000], index=1)
    page      = st.number_input("Page", min_value=1, max_value=max(1, len(filtered_df)//page_size + 1), value=1)
    start     = (page - 1) * page_size
    end       = start + page_size
    st.dataframe(filtered_df.iloc[start:end], use_container_width=True, height=500)

    col_dl, _ = st.columns([1, 4])
    csv_bytes  = filtered_df.to_csv(index=False).encode()
    col_dl.download_button("⬇️ Download Filtered CSV", csv_bytes, "filtered_data.csv", "text/csv")

with tab2:
    st.markdown("### 📈 Visualize Any Column")
    chart_cols = st.columns(3)
    x_col = chart_cols[0].selectbox("X-axis", df.columns.tolist())
    y_col = chart_cols[1].selectbox("Y-axis", df.columns.tolist(),
                                     index=min(1, len(df.columns)-1))
    chart_type = chart_cols[2].selectbox("Chart Type", ["Bar","Scatter","Line","Box","Histogram","Violin"])

    color_col = None
    if mid_col:
        color_col = mid_col
    elif status_col:
        color_col = status_col

    try:
        plot_df = filtered_df[[c for c in [x_col, y_col, color_col] if c]].dropna()
        if chart_type == "Bar":
            fig = px.bar(plot_df,    x=x_col, y=y_col, color=color_col)
        elif chart_type == "Scatter":
            fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col)
        elif chart_type == "Line":
            fig = px.line(plot_df,   x=x_col, y=y_col, color=color_col)
        elif chart_type == "Box":
            fig = px.box(plot_df,    x=x_col, y=y_col, color=color_col)
        elif chart_type == "Histogram":
            fig = px.histogram(plot_df, x=x_col, nbins=40)
        elif chart_type == "Violin":
            fig = px.violin(plot_df, x=x_col, y=y_col, color=color_col)
        fig.update_layout(**PLOTLY_THEME, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {e}")

    # Correlation heatmap
    num_cols_ = col_roles["numeric"]
    if len(num_cols_) >= 2:
        st.markdown("### 🔥 Numeric Correlation Heatmap")
        corr = filtered_df[num_cols_].corr()
        fig  = px.imshow(corr, color_continuous_scale="RdBu_r", aspect="auto",
                         text_auto=".2f")
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### 🤖 AI-Powered Machine Analysis Report")
    st.markdown("Select a machine to generate a detailed AI maintenance report (uses Gemma2:2b)")

    if mid_col:
        machines_list = sorted(df[mid_col].dropna().astype(str).unique().tolist())
        sel_m         = st.selectbox("Select Machine", machines_list)
        mdf           = df[df[mid_col].astype(str) == sel_m]

        mc1, mc2, mc3 = st.columns(3)
        ms_data = analytics.get("machine_stats", {}).get(sel_m, {})
        mc1.metric("Total Events",    ms_data.get("total_events", len(mdf)))
        mc2.metric("Failure Count",   ms_data.get("failure_count", "N/A"))
        mc3.metric("Health Score",    f"{ms_data.get('health_score', 'N/A')}/100")

        mc4, mc5, mc6 = st.columns(3)
        mc4.metric("Failure Rate",    f"{ms_data.get('failure_rate', 0):.1f}%")
        mc5.metric("MTBF (hrs)",      ms_data.get("mtbf_hours", "N/A"))
        mc6.metric("Total Cost",      f"${ms_data.get('total_cost', 0):,.0f}" if ms_data.get("total_cost") else "N/A")

        if st.button(f"🤖 Generate AI Report for {sel_m}"):
            with st.spinner("Generating AI report... (Ollama Gemma2:2b)"):
                try:
                    report = generate_machine_report(sel_m, mdf, analytics)
                    st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)

                    dl_col, _ = st.columns([1,4])
                    dl_col.download_button(
                        "⬇️ Download Report",
                        report.encode(),
                        f"report_{sel_m}.txt",
                        "text/plain"
                    )
                except Exception as e:
                    st.error(f"Report generation failed: {e}\n\nMake sure Ollama is running: `ollama serve`")
        
        st.markdown("---")
        st.markdown(f"**Raw data for machine: {sel_m}** ({len(mdf)} rows)")
        st.dataframe(mdf, use_container_width=True, height=400)
    else:
        st.info("No machine ID column detected in your CSV. The AI report requires a machine identifier column.")