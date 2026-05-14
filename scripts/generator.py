"""
generator.py — Ollama Gemma:2b RAG generation engine.
STRICT context-only answering — zero hallucination design.
Compatible with ollama Python library >=0.1.x and >=0.2.x (object + dict both handled).
"""
import re
import json
import ollama
import pandas as pd
import numpy as np

# ─── Model config ──────────────────────────────────────────────────────────────
OLLAMA_MODEL = "gemma:2b"

# ─── System prompt (strict RAG) ───────────────────────────────────────────────
SYSTEM_PROMPT = """You are an Industrial AI Analyst assistant built exclusively to analyze machine maintenance data.

STRICT RULES — YOU MUST FOLLOW:
1. Answer ONLY using the DATA CONTEXT provided below. Never use outside knowledge.
2. If the answer is not in the context, say: "This information is not available in the uploaded dataset."
3. For predictions, base them STRICTLY on patterns and trends visible in the provided data.
4. For cost analysis, use ONLY the actual cost values present in the data.
5. Always cite specific values, machine IDs, dates, or error codes from the context when answering.
6. Be concise but thorough. Use numbers whenever available.
7. Never fabricate machine names, error codes, costs, or dates not in the data.
8. Structure answers clearly: observation → analysis → recommendation (when applicable).

You specialize in:
- Machine failure analysis
- Downtime cost estimation
- Predictive maintenance recommendations
- Technician workload analysis
- Root cause identification
- Maintenance scheduling optimization
"""

# ─── Ollama connectivity check ────────────────────────────────────────────────
def check_ollama() -> tuple[bool, str]:
    """
    Returns (is_available, message).
    Handles both old (dict) and new (object) ollama library versions.
    """
    try:
        result = ollama.list()
        # New ollama library: result is an object with .models attribute
        if hasattr(result, "models"):
            models = [m.model for m in result.models]
        # Old ollama library: result is a dict with 'models' key
        elif isinstance(result, dict) and "models" in result:
            models = [m.get("name", m.get("model", "")) for m in result["models"]]
        else:
            models = []

        if not any(OLLAMA_MODEL in m for m in models):
            return False, f"Model '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}"
        return True, f"✅ Ollama online · Model: {OLLAMA_MODEL}"
    except Exception as e:
        return False, f"❌ Ollama not reachable. Run: ollama serve · Details: {e}"


# ─── Safe chunk content extractor ────────────────────────────────────────────
def _get_chunk_content(chunk) -> str:
    """
    Extract text content from a streaming chunk.
    Handles both old (dict) and new (Pydantic object) ollama library versions.
    """
    try:
        # New library: chunk is a ChatResponse object
        if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
            return chunk.message.content or ""
        # Old library: chunk is a dict
        if isinstance(chunk, dict):
            msg = chunk.get("message", {})
            if isinstance(msg, dict):
                return msg.get("content", "")
    except Exception:
        pass
    return ""


def _get_response_content(response) -> str:
    """
    Extract text content from a non-streaming response.
    Handles both old (dict) and new (Pydantic object) ollama library versions.
    """
    try:
        if hasattr(response, "message") and hasattr(response.message, "content"):
            return response.message.content or ""
        if isinstance(response, dict):
            msg = response.get("message", {})
            if isinstance(msg, dict):
                return msg.get("content", "")
    except Exception:
        pass
    return "⚠️ Could not parse Ollama response."


# ─── Query classifier ─────────────────────────────────────────────────────────
QUERY_TYPES = {
    "prediction":  ["predict", "forecast", "future", "next", "will", "expect", "likely", "risk", "probability"],
    "cost":        ["cost", "expense", "price", "budget", "spend", "money", "financial", "saving", "reduce cost"],
    "technician":  ["technician", "operator", "engineer", "who", "assigned", "personnel", "worker"],
    "root_cause":  ["why", "cause", "reason", "root", "origin", "due to", "because"],
    "summary":     ["summary", "overview", "total", "overall", "report", "all machines"],
    "anomaly":     ["anomaly", "unusual", "outlier", "abnormal", "spike", "strange"],
    "machine":     ["machine", "asset", "equipment", "unit", "device"],
    "failure":     ["fail", "fault", "error", "alarm", "breakdown", "downtime", "outage"],
}

def classify_query(query: str) -> list:
    ql = query.lower()
    types = [t for t, kws in QUERY_TYPES.items() if any(kw in ql for kw in kws)]
    return types or ["general"]


# ─── Context builder ──────────────────────────────────────────────────────────
def build_context(hits: list, df: pd.DataFrame,
                  analytics_summary: dict, query: str) -> str:
    lines = []

    lines.append("=== DATASET OVERVIEW ===")
    lines.append(f"Total records: {analytics_summary.get('total_rows', len(df))}")
    lines.append(f"Total columns: {analytics_summary.get('total_cols', len(df.columns))}")
    if analytics_summary.get("date_range"):
        lines.append(f"Date range: {analytics_summary['date_range']}")

    num_stats = analytics_summary.get("numeric_summary", {})
    if num_stats:
        lines.append("\n=== KEY METRICS FROM DATA ===")
        for col, stats in num_stats.items():
            lines.append(f"  {col}: min={stats['min']:.2f}, max={stats['max']:.2f}, "
                         f"mean={stats['mean']:.2f}, total={stats['sum']:.2f}")

    mh = analytics_summary.get("machine_health", {})
    if mh:
        lines.append("\n=== MACHINE HEALTH SCORES ===")
        for mid, score in list(mh.items())[:10]:
            lines.append(f"  {mid}: health_score={score:.1f}/100")

    tf = analytics_summary.get("top_failures", [])
    if tf:
        lines.append("\n=== TOP FAILURE MODES ===")
        for item in tf[:5]:
            lines.append(f"  {item}")

    lines.append(f"\n=== RELEVANT DATA ROWS (semantic match for: '{query}') ===")
    for i, hit in enumerate(hits[:15], 1):
        lines.append(f"  [{i}] score={hit['score']:.3f} | {hit['text']}")

    return "\n".join(lines)


def build_prompt(query: str, context: str, query_types: list) -> str:
    instructions = []
    if "prediction" in query_types:
        instructions.append(
            "Based on the failure trends in the data, provide a data-driven prediction. "
            "Include: probability estimate, time horizon, and preventive actions."
        )
    if "cost" in query_types:
        instructions.append(
            "Calculate costs using actual values from the data. Show: current cost, projected savings."
        )
    if "technician" in query_types:
        instructions.append(
            "Analyze technician workload from the data. Show: assignments, response times, efficiency."
        )
    if "root_cause" in query_types:
        instructions.append(
            "Perform root cause analysis using the retrieved records. Show: primary cause, evidence."
        )
    if "summary" in query_types:
        instructions.append(
            "Provide a structured executive summary with key KPIs, trends, and recommendations."
        )

    instruction_text = "\n".join(instructions) if instructions else \
        "Answer the question precisely using only the provided data context."

    return f"""DATA CONTEXT:
{context}

SPECIAL INSTRUCTIONS:
{instruction_text}

USER QUESTION: {query}

ANSWER (use ONLY data from context above, never invent values):"""


# ─── Main Generator ───────────────────────────────────────────────────────────
def generate_answer(query: str, hits: list, df: pd.DataFrame,
                    analytics_summary: dict, stream: bool = True):
    query_types = classify_query(query)
    context     = build_context(hits, df, analytics_summary, query)
    prompt      = build_prompt(query, context, query_types)

    if stream:
        return _stream_generate(prompt)
    else:
        return _generate(prompt)


def _stream_generate(prompt: str):
    """Streaming generator for Streamlit. Handles old + new ollama library."""
    try:
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            stream=True,
            options={
                "temperature":    0.1,
                "top_p":          0.9,
                "num_predict":    1024,
                "repeat_penalty": 1.1,
            }
        )
        for chunk in stream:
            delta = _get_chunk_content(chunk)
            if delta:
                yield delta
    except Exception as e:
        yield (f"\n⚠️ **Ollama Error:** {e}\n\n"
               f"Make sure Ollama is running:\n```\nollama serve\n```\n"
               f"And model is pulled:\n```\nollama pull {OLLAMA_MODEL}\n```")


def _generate(prompt: str) -> str:
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            options={"temperature": 0.1, "top_p": 0.9, "num_predict": 1024}
        )
        return _get_response_content(response)
    except Exception as e:
        return f"⚠️ Ollama Error: {e}"


def generate_machine_report(machine_id: str, machine_df: pd.DataFrame,
                             analytics_summary: dict) -> str:
    rows_text = machine_df.head(20).to_string(index=False)
    prompt = f"""You are analyzing machine: {machine_id}

MACHINE DATA:
{rows_text}

DATASET STATS:
{json.dumps({k: v for k, v in analytics_summary.items()
             if k not in ('col_roles', 'trend_data')}, indent=2, default=str)}

Generate a comprehensive maintenance report for this machine including:
1. Current health status (based on data only)
2. Failure pattern analysis
3. Cost impact (if cost data available)
4. Recommended maintenance actions
5. Risk prediction for next 30 days (based on historical pattern — label clearly as prediction)

Use ONLY information from the data above."""

    return _generate(prompt)

# Alias for compatibility
check_ollama_available = check_ollama