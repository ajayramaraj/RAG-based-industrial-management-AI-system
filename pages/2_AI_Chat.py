"""
pages/2_AI_Chat.py — Industrial AI Copilot chat interface.
Strict RAG: retrieves only from uploaded CSV, generates with Gemma2:2b.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
from scripts.generator import generate_answer, classify_query
from scripts.retriever import FAISSRetriever

st.set_page_config(page_title="AI Chat | Industrial AI", page_icon="🤖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
:root { --bg:#0a0e17; --surface:#111827; --border:#1e293b; --accent:#00d4ff; --accent2:#ff6b2b; --green:#00ff88; --red:#ff3b5c; --text:#e2e8f0; --muted:#64748b; }
html,body,[class*="css"] { background-color:var(--bg) !important; color:var(--text) !important; font-family:'Syne',sans-serif; }
.stApp { background:var(--bg) !important; }
h1 { font-size:2rem !important; font-weight:800 !important; }

.user-bubble {
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 16px 16px 4px 16px;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0 0.5rem 20%;
    color: #e2e8f0;
    font-size: 0.95rem;
}
.ai-bubble {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 16px 16px 16px 4px;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 20% 0.5rem 0;
    color: #e2e8f0;
    font-size: 0.95rem;
    border-left: 3px solid #00d4ff;
}
.ai-bubble .label { font-size:0.7rem; color:#00d4ff; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.4rem; }
.user-bubble .label { font-size:0.7rem; color:#a78bfa; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.4rem; text-align:right; }

.query-type-tag {
    display:inline-block;
    background:rgba(255,107,43,0.1);
    border:1px solid rgba(255,107,43,0.3);
    border-radius:20px;
    padding:1px 10px;
    font-size:0.7rem;
    color:#ff6b2b;
    margin:1px;
}
.suggestion-btn {
    background: rgba(0,212,255,0.05) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 8px !important;
    padding: 0.4rem 0.8rem !important;
    color: #00d4ff !important;
    font-size: 0.85rem !important;
    cursor: pointer;
    transition: all 0.2s;
    width: 100%;
    text-align: left;
    margin: 3px 0;
}
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
    font-family: 'Syne', sans-serif !important;
}
.context-panel {
    background:#0f172a;
    border:1px solid #1e293b;
    border-radius:8px;
    padding:0.8rem;
    font-family:'JetBrains Mono',monospace;
    font-size:0.75rem;
    color:#64748b;
    max-height:200px;
    overflow-y:auto;
}
</style>
""", unsafe_allow_html=True)

# ─── Guard ────────────────────────────────────────────────────────────────────
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ No data loaded. Please go to Home and upload a CSV first.")
    st.stop()
if not st.session_state.get("index_built"):
    st.warning("⏳ Semantic index is still building. Please wait.")
    st.stop()

df        = st.session_state.df
analytics = st.session_state.analytics
retriever = st.session_state.retriever

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("# 🤖 Industrial AI Copilot")
st.markdown(f"Analyzing **{len(df):,} records** · Model: `gemma2:2b` · Retrieval: `AllMiniLM-L6-v2`")
st.markdown("---")

# ─── Layout: chat | context ───────────────────────────────────────────────────
chat_col, ctx_col = st.columns([3, 1])

with ctx_col:
    st.markdown("### 💡 Suggested Questions")

    machine_col = st.session_state.col_roles.get("machine_id", [None])[0]
    machines    = df[machine_col].dropna().unique()[:5].tolist() if machine_col else []

    suggestions = [
        "Which machine has the highest failure rate?",
        "What are the most common fault types?",
        "Summarize overall maintenance cost",
        "Predict which machines need maintenance next",
        "Which technician handled the most critical failures?",
        "What is the trend in downtime over time?",
        "Show root cause of recurring failures",
        "Reduce maintenance cost recommendations",
    ]
    if len(machines) > 0:
        suggestions.insert(2, f"Analyze machine {machines[0]} in detail")
        suggestions.insert(4, f"Why did {machines[0]} fail frequently?")

    # Store selected suggestion
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = ""

    for s in suggestions[:8]:
        if st.button(s, key=f"sug_{s[:20]}", use_container_width=True):
            st.session_state.pending_query = s
            st.rerun()

    st.markdown("---")
    st.markdown("### 🔍 Retrieval Settings")
    top_k = st.slider("Retrieved rows (top-k)", 5, 30, 15, help="More rows = more context, slower response")

    st.markdown("---")
    st.markdown("### ℹ️ How it Works")
    st.markdown("""
    1. Your question is embedded with **AllMiniLM**
    2. Top matching rows fetched from **FAISS index**
    3. Context sent to **Gemma:2b** via Ollama
    4. Answer generated ONLY from your CSV data
    """)

with chat_col:
    # Chat history display
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-bubble">
                <div class="label">👤 You</div>
                {msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            q_types = msg.get("query_types", [])
            tags    = "".join(f'<span class="query-type-tag">{t}</span>' for t in q_types)
            st.markdown(f"""
            <div class="ai-bubble">
                <div class="label">🤖 Industrial AI {tags}</div>
                {msg['content'].replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

            # Show retrieved context if available
            if msg.get("context_rows") and st.checkbox(f"Show retrieved context", key=f"ctx_{msg.get('id',0)}"):
                st.markdown('<div class="context-panel">' +
                            "<br>".join(f"[{i+1}] {r}" for i, r in enumerate(msg["context_rows"][:8])) +
                            "</div>", unsafe_allow_html=True)

    # ─── Input ────────────────────────────────────────────────────────────────
    st.markdown("")
    with st.container():
        inp_col, btn_col = st.columns([5,1])
        with inp_col:
            # Pre-fill from suggestion click
            default_val = st.session_state.get("pending_query", "")
            user_input = st.text_input(
                "Ask anything about your data...",
                value=default_val,
                placeholder="e.g. Which machine has the most downtime? Predict failures next month...",
                label_visibility="collapsed",
                key="chat_input"
            )
        with btn_col:
            send = st.button("Send ➤", use_container_width=True)

    # Clear pending query after reading
    if st.session_state.get("pending_query"):
        st.session_state.pending_query = ""

    if (send or default_val) and user_input.strip():
        query = user_input.strip()
        q_types = classify_query(query)

        # Add user message
        st.session_state.chat_history.append({
            "role":    "user",
            "content": query,
        })

        # Retrieve
        hits = retriever.retrieve(query, k=top_k)

        # Stream answer
        with st.spinner("🔍 Retrieving relevant data & generating answer..."):
            answer_placeholder = st.empty()
            full_answer = ""

            with answer_placeholder.container():
                st.markdown('<div class="ai-bubble"><div class="label">🤖 Industrial AI</div>', unsafe_allow_html=True)
                stream_container = st.empty()
                st.markdown("</div>", unsafe_allow_html=True)

            for chunk in generate_answer(query, hits, df, analytics, stream=True):
                full_answer += chunk
                stream_container.markdown(full_answer)

        # Save to history
        import time
        st.session_state.chat_history.append({
            "role":         "assistant",
            "content":      full_answer,
            "query_types":  q_types,
            "context_rows": [h["text"] for h in hits],
            "id":           int(time.time()),
        })

        st.rerun()

    # Clear chat
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()