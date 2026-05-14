# 🏭 Industrial Maintenance Intelligence Platform

A production-grade AI system for predictive maintenance analytics.
Built with: Streamlit · AllMiniLM · FAISS · Ollama (gemma2:2b)

---

## 📁 Project Structure

```
industrial_ai/
├── main.py                  ← Entry point (run this)
├── requirements.txt
├── data/                    ← Uploaded CSV + metadata saved here
├── index_store/             ← FAISS index + chunk store (auto-generated)
├── models/                  ← (reserved for future ML models)
├── notebooks/               ← Jupyter notebooks (optional analysis)
├── pages/
│   ├── 1_Dashboard.py       ← Upload, KPIs, charts, risk table
│   ├── 2_AI_Chat.py         ← AI copilot with streaming RAG
│   └── 3_Data_Explorer.py   ← Filters, stats, distributions, correlations
└── scripts/
    ├── csv_loader.py        ← CSV ingestion, cleaning, column detection
    ├── analytics.py         ← MTBF, MTTR, health scores, cost analysis
    ├── retriever.py         ← AllMiniLM + FAISS semantic search
    ├── generator.py         ← Ollama gemma2:2b grounded generation
    └── utils.py             ← Session state, formatting, helpers
```

---

## ⚙️ Setup

### 1. Install Python dependencies
```bash
cd industrial_ai
pip install -r requirements.txt
```

### 2. Install & start Ollama
Download from https://ollama.ai then:
```bash
# Terminal 1: start server
ollama serve

# Terminal 2: pull model
ollama pull gemma2:2b
```

### 3. Run the app
```bash
streamlit run main.py
```

---

## 🚀 Features

### Dashboard (Page 1)
- Upload CSV up to 40,000 rows
- Auto-detects: machine IDs, timestamps, error codes, status, downtime, cost, technician
- Builds AllMiniLM vector index with progress bar (cached per file hash)
- KPIs: total records, machines, errors, downtime, cost, projected annual cost
- Charts: failure frequency, trend over time, machine health scores, cost by machine
- MTBF & MTTR table per machine
- 30-day risk prediction with recommended actions
- Technician workload pie chart

### AI Chat (Page 2)
- Streaming responses from gemma2:2b via Ollama
- Retrieves top-K relevant records from FAISS (CSV-only, no hallucination)
- Auto-detects machine mentions in query → filters retrieval
- Suggested quick questions
- One-click Predictive Maintenance Report generation
- Chat history with clear button
- Shows source records used per answer

### Data Explorer (Page 3)
- Dynamic multi-filter: machine, error code, status, date range, text search
- Export filtered data as CSV
- Per-column statistics (min/max/mean/std for numeric, value counts for categorical)
- Distribution charts: histogram, violin, box
- Scatter plot: any two numeric columns
- Pearson correlation heatmap

---

## 📊 Supported CSV Columns (auto-detected)

| Role         | Example column names                                    |
|-------------|--------------------------------------------------------|
| Machine ID   | machine, asset, equipment, unit, device, id             |
| Timestamp    | time, date, timestamp, datetime, created, logged        |
| Error Code   | error, fault, alarm, code, failure, issue, alert        |
| Status       | status, state, condition, mode, operational             |
| Description  | description, notes, message, comment, detail, reason    |
| Numeric      | temp, pressure, speed, rpm, voltage, vibration, flow    |
| Downtime     | downtime, duration, hours, minutes, elapsed             |
| Cost         | cost, expense, charge, amount, price, repair            |
| Technician   | technician, tech, operator, assigned, engineer, worker  |

---

## 💡 Example Questions for AI Chat

- "Which machine has failed the most times?"
- "What are the top 3 error codes and their total cost?"
- "Predict which machines are at risk in the next 30 days"
- "Why does Machine M-23 keep failing?"
- "How can we reduce maintenance costs?"
- "What is the average downtime for vibration errors?"
- "Which technician has the highest workload?"
- "Give me a full health report for all machines"
- "What patterns do you see in the critical failures?"

---

## 🔒 AI Grounding (No Hallucination)

The AI ONLY answers from:
1. Retrieved records from your FAISS index (built from your CSV)
2. Pre-computed analytics context (MTBF, costs, health scores)

It never uses external knowledge or training data for factual claims.
Predictions are clearly labelled as `📊 PREDICTION:`.

---

## 🗑️ Files to REMOVE from original structure

These files from the original project are replaced:
- `scripts/analytics.py` → replaced with full version
- `scripts/csv_loader.py` → replaced with full version  
- `scripts/generator.py` → replaced with full version
- `scripts/retriever.py` → replaced with full version
- `scripts/utils.py` → replaced with full version
- `pages/1_Dashboard.py` → replaced with full version
- `pages/2_AI_Chat.py` → replaced with full version
- `pages/3_Data_Explorer.py` → replaced with full version
- `main.py` → replaced with full version

Keep:
- `.venv/` → your virtual environment
- `data/` → auto-populated
- `index_store/` → auto-populated
- `models/` → reserved
- `notebooks/` → reserved for Jupyter

---

## 🎓 Placement Project Highlights

**Technologies demonstrated:**
- NLP & semantic embeddings (sentence-transformers)
- Vector database & similarity search (FAISS)
- RAG (Retrieval Augmented Generation) architecture
- LLM integration (Ollama local inference)
- Industrial analytics (MTBF, MTTR, availability)
- Predictive risk modeling
- Full-stack web app (Streamlit)
- Large data handling (40K rows)

**System design concepts shown:**
- Modular architecture (5 independent scripts)
- Session state management
- Index caching (hash-based)
- Streaming LLM output
- Grounded generation (no hallucination)
- Auto column detection & data cleaning
