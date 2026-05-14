"""
pages/1_Dashboard.py — Full analytics dashboard with KPIs, charts, machine health
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Dashboard | Industrial AI", page_icon="📊", layout="wide")

# ─── CSS (dark theme) ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
:root { --bg:#0a0e17; --surface:#111827; --border:#1e293b; --accent:#00d4ff; --accent2:#ff6b2b; --green:#00ff88; --red:#ff3b5c; --text:#e2e8f0; --muted:#64748b; }
html,body,[class*="css"] { background-color:var(--bg) !important; color:var(--text) !important; font-family:'Syne',sans-serif; }
.stApp { background:var(--bg) !important; }
.kpi { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:1.2rem 1.5rem; border-left:3px solid var(--accent); margin:4px; }
.kpi label { font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:2px; display:block; }
.kpi span { font-size:1.8rem; font-weight:800; color:var(--accent); font-family:'JetBrains Mono',monospace; }
h1 { font-size:2rem !important; font-weight:800 !important; }
h2 { font-size:1.4rem !important; color:var(--accent) !important; }
.stDataFrame { background:var(--surface) !important; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = {
    "paper_bgcolor": "#0a0e17",
    "plot_bgcolor":  "#111827",
    "font":          {"color": "#e2e8f0", "family": "Syne"},
    "xaxis":         {"gridcolor": "#1e293b", "linecolor": "#1e293b"},
    "yaxis":         {"gridcolor": "#1e293b", "linecolor": "#1e293b"},
}

def styled_chart(fig):
    fig.update_layout(**PLOTLY_THEME, margin=dict(t=40, b=20, l=20, r=20))
    return fig

def kpi(label, value, color="#00d4ff"):
    return f'<div class="kpi"><label>{label}</label><span style="color:{color};">{value}</span></div>'


# ─── Guard ────────────────────────────────────────────────────────────────────
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ No data loaded. Go to Home and upload a CSV first.")
    st.stop()

df        = st.session_state.df
analytics = st.session_state.analytics
col_roles = st.session_state.col_roles

st.markdown("# 📊 Industrial Intelligence Dashboard")

# ─── Top KPIs ─────────────────────────────────────────────────────────────────
st.markdown("## Key Performance Indicators")

cost   = analytics.get("cost_analysis", {})
fa     = analytics.get("failure_analysis", {})
ms     = analytics.get("machine_stats", {})
mh     = analytics.get("machine_health", {})

total_cost   = cost.get("total_cost",   None)
avg_health   = np.mean(list(mh.values())) if mh else None
fault_count  = sum(analytics.get("failure_analysis", {}).get("status_distribution", {}).values())
machine_cnt  = len(ms)
num_summary  = analytics.get("numeric_summary", {})

# Find downtime col
downtime_total = None
for col, stats in num_summary.items():
    if any(k in col.lower() for k in ["downtime","duration","hours"]):
        downtime_total = stats["sum"]
        break

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.markdown(kpi("Total Records", f"{len(df):,}"),          unsafe_allow_html=True)
c2.markdown(kpi("Machines",     str(machine_cnt), "#ff6b2b"), unsafe_allow_html=True)
c3.markdown(kpi("Fault Events", f"{fault_count:,}", "#ff3b5c"), unsafe_allow_html=True)
c4.markdown(kpi("Total Cost",   f"${total_cost:,.0f}" if total_cost else "N/A", "#00ff88"), unsafe_allow_html=True)
c5.markdown(kpi("Avg Health",   f"{avg_health:.1f}/100" if avg_health else "N/A", "#a78bfa"), unsafe_allow_html=True)
c6.markdown(kpi("Total Downtime", f"{downtime_total:,.1f}h" if downtime_total else "N/A", "#ffb800"), unsafe_allow_html=True)

st.markdown("---")

# ─── Row 1: Failure distribution + Machine health ─────────────────────────────
r1c1, r1c2 = st.columns(2)

with r1c1:
    st.markdown("### ⚠️ Fault / Status Distribution")
    status_dist = fa.get("status_distribution", {})
    if status_dist:
        fd = pd.DataFrame(list(status_dist.items()), columns=["Status","Count"]).sort_values("Count", ascending=False).head(15)
        fig = px.bar(fd, x="Count", y="Status", orientation="h",
                     color="Count", color_continuous_scale=["#1e293b","#ff3b5c"])
        fig = styled_chart(fig)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No status column detected")

with r1c2:
    st.markdown("### 💚 Machine Health Scores")
    if mh:
        health_df = pd.DataFrame(list(mh.items()), columns=["Machine","Health"]).sort_values("Health")
        colors = ["#ff3b5c" if h < 40 else "#ffb800" if h < 70 else "#00ff88" for h in health_df["Health"]]
        fig = go.Figure(go.Bar(
            x=health_df["Health"], y=health_df["Machine"],
            orientation="h", marker_color=colors,
            text=health_df["Health"].apply(lambda x: f"{x:.0f}"),
            textposition="outside"
        ))
        fig = styled_chart(fig)
        fig.update_xaxes(range=[0,100])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No machine ID column detected")

st.markdown("---")

# ─── Row 2: Cost by machine + Trend ──────────────────────────────────────────
r2c1, r2c2 = st.columns(2)

with r2c1:
    st.markdown("### 💰 Cost by Machine")
    cbm = cost.get("cost_by_machine", {})
    if cbm:
        cost_df = pd.DataFrame(list(cbm.items()), columns=["Machine","Cost"]).sort_values("Cost", ascending=False)
        fig = px.bar(cost_df, x="Machine", y="Cost",
                     color="Cost", color_continuous_scale=["#1e293b","#00d4ff"])
        fig = styled_chart(fig)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No cost column detected")

with r2c2:
    st.markdown("### 📈 Monthly Event Trend")
    trend = analytics.get("trend_data", {})
    me    = trend.get("monthly_events", [])
    if me:
        tdf = pd.DataFrame(me)
        fig = px.area(tdf, x="__month__", y="count",
                      color_discrete_sequence=["#00d4ff"],
                      markers=True)
        fig = styled_chart(fig)
        fig.update_traces(fill="tozeroy", fillcolor="rgba(0,212,255,0.1)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No datetime column for trend analysis")

st.markdown("---")

# ─── Row 3: MTBF / Failure rates table ────────────────────────────────────────
st.markdown("### 🔬 Machine-Level KPI Table")
if ms:
    rows = []
    for mid, stats in ms.items():
        rows.append({
            "Machine":       mid,
            "Events":        stats.get("total_events", 0),
            "Failures":      stats.get("failure_count", 0),
            "Failure Rate%": stats.get("failure_rate", 0),
            "Health Score":  stats.get("health_score", "N/A"),
            "MTBF (hrs)":    stats.get("mtbf_hours", "N/A"),
            "Total Downtime":stats.get("total_downtime", "N/A"),
            "Total Cost":    stats.get("total_cost", "N/A"),
        })
    tbl = pd.DataFrame(rows).sort_values("Failure Rate%", ascending=False)
    def color_health(val):
        if isinstance(val, (int,float)):
            if val < 40: return "color: #ff3b5c"
            if val < 70: return "color: #ffb800"
            return "color: #00ff88"
        return ""
    def color_failrate(val):
        if isinstance(val, (int,float)):
            if val > 50: return "color: #ff3b5c"
            if val > 25: return "color: #ffb800"
            return "color: #00ff88"
        return ""
    styled = (tbl.style
              .applymap(color_health,     subset=["Health Score"])
              .applymap(color_failrate,   subset=["Failure Rate%"])
              .format({"Failure Rate%": "{:.1f}%",
                       "Health Score": lambda x: f"{x:.1f}" if isinstance(x,(int,float)) else x,
                       "MTBF (hrs)": lambda x: f"{x:.1f}" if isinstance(x,(int,float)) else x,
                       "Total Cost": lambda x: f"${x:,.0f}" if isinstance(x,(int,float)) else x}))
    st.dataframe(styled, use_container_width=True, height=400)

# ─── Row 4: Technician stats ──────────────────────────────────────────────────
tech = analytics.get("technician_stats", {})
if tech.get("assignments"):
    st.markdown("---")
    st.markdown("### 👨‍🔧 Technician Workload Analysis")
    tc1, tc2 = st.columns(2)
    with tc1:
        assign_df = pd.DataFrame(list(tech["assignments"].items()), columns=["Technician","Jobs"])
        fig = px.pie(assign_df, names="Technician", values="Jobs",
                     color_discrete_sequence=px.colors.qualitative.Bold, hole=0.4)
        fig = styled_chart(fig)
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    with tc2:
        if tech.get("cost_by_tech"):
            ctdf = pd.DataFrame(list(tech["cost_by_tech"].items()), columns=["Technician","Cost"])
            fig  = px.bar(ctdf.sort_values("Cost", ascending=True), x="Cost", y="Technician",
                          orientation="h", color="Cost", color_continuous_scale=["#1e293b","#ff6b2b"])
            fig  = styled_chart(fig)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cost data for technicians")

# ─── Row 5: Numeric column distributions ──────────────────────────────────────
num_cols = col_roles["numeric"][:4]
if num_cols:
    st.markdown("---")
    st.markdown("### 🔢 Numeric Column Distributions")
    dist_cols = st.columns(len(num_cols))
    for i, col in enumerate(num_cols):
        with dist_cols[i]:
            fig = px.histogram(df, x=col, nbins=40,
                               color_discrete_sequence=["#00d4ff"])
            fig = styled_chart(fig)
            fig.update_layout(title=col, height=250, margin=dict(t=30,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)