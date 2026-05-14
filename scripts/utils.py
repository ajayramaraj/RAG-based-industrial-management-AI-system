"""
utils.py — Shared utility functions for Industrial AI Platform
"""
import os
import re
import json
import hashlib
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
INDEX_DIR  = BASE_DIR / "index_store"
MODEL_DIR  = BASE_DIR / "models"

for d in [DATA_DIR, INDEX_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─── CSV hash (cache invalidation) ────────────────────────────────────────────
def file_hash(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()[:12]


# ─── Column type detection ─────────────────────────────────────────────────────
DATETIME_HINTS  = ["date", "time", "timestamp", "created", "updated", "recorded"]
NUMERIC_HINTS   = ["cost", "price", "downtime", "duration", "count", "amount",
                   "hours", "minutes", "quantity", "rate", "temp", "pressure",
                   "vibration", "rpm", "voltage", "current", "load", "speed"]
MACHINE_HINTS   = ["machine", "asset", "equipment", "unit", "device", "id"]
STATUS_HINTS    = ["status", "state", "condition", "fault", "error", "alarm",
                   "severity", "priority", "type", "code", "category"]
TECH_HINTS      = ["technician", "operator", "engineer", "assigned", "personnel", "worker"]

def detect_column_roles(df: pd.DataFrame) -> dict:
    roles = {
        "datetime": [],
        "numeric": [],
        "machine_id": [],
        "status": [],
        "technician": [],
        "description": [],
        "other": []
    }
    for col in df.columns:
        cl = col.lower().replace(" ", "_")
        if any(h in cl for h in DATETIME_HINTS):
            roles["datetime"].append(col)
        elif any(h in cl for h in MACHINE_HINTS):
            roles["machine_id"].append(col)
        elif any(h in cl for h in STATUS_HINTS):
            roles["status"].append(col)
        elif any(h in cl for h in TECH_HINTS):
            roles["technician"].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            roles["numeric"].append(col)
        elif df[col].dtype == object and df[col].str.len().mean() > 30:
            roles["description"].append(col)
        else:
            roles["other"].append(col)
    return roles


# ─── Row → Natural Language ────────────────────────────────────────────────────
def row_to_text(row: pd.Series, col_roles: dict) -> str:
    """Convert a DataFrame row to a natural-language sentence for embedding."""
    parts = []
    for col, val in row.items():
        if pd.isna(val) or str(val).strip() == "":
            continue
        parts.append(f"{col}: {val}")
    return " | ".join(parts)


def df_to_texts(df: pd.DataFrame, col_roles: dict, batch_size: int = 1000) -> list:
    """Convert entire dataframe to list of text strings."""
    texts = []
    for _, row in df.iterrows():
        texts.append(row_to_text(row, col_roles))
    return texts


# ─── Number formatting ─────────────────────────────────────────────────────────
def fmt_number(n, prefix="", suffix=""):
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    if abs(n) >= 1_000_000:
        return f"{prefix}{n/1_000_000:.2f}M{suffix}"
    if abs(n) >= 1_000:
        return f"{prefix}{n/1_000:.1f}K{suffix}"
    return f"{prefix}{n:.2f}{suffix}"


# ─── Save / Load pickle ────────────────────────────────────────────────────────
def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── Column suggestion (if CSV doesn't have headers) ──────────────────────────
def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        re.sub(r"[^\w]", "_", str(c)).strip("_").lower()
        for c in df.columns
    ]
    # Deduplicate
    seen = {}
    new_cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols
    return df