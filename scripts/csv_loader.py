"""
csv_loader.py — Handles CSV ingestion, cleaning, and preprocessing.
Supports up to 40,000 rows with intelligent column detection.
"""
import io
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from scripts.utils import (
    sanitize_columns, detect_column_roles, file_hash,
    DATA_DIR, save_pickle, load_pickle
)


# ─── Core Loader ──────────────────────────────────────────────────────────────
def load_csv(uploaded_file) -> tuple[pd.DataFrame, str]:
    """
    Load and clean CSV file.
    Returns (cleaned_df, file_hash_id)
    """
    raw_bytes = uploaded_file.read()
    fhash = file_hash(raw_bytes)

    # Try multiple encodings
    for enc in ["utf-8", "latin-1", "cp1252", "utf-8-sig"]:
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, low_memory=False)
            break
        except Exception:
            continue
    else:
        raise ValueError("Could not decode CSV. Try UTF-8 or Latin-1 encoding.")

    df = sanitize_columns(df)
    df = clean_dataframe(df)

    # Save to disk
    df.to_parquet(DATA_DIR / f"{fhash}.parquet", index=False)

    return df, fhash


def load_cached(fhash: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{fhash}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


# ─── Cleaning Pipeline ────────────────────────────────────────────────────────
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Drop fully-empty rows/cols
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    # Trim whitespace in string cols
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # Auto-parse datetime cols
    df = auto_parse_dates(df)

    # Auto-convert numeric cols stored as strings
    df = auto_parse_numerics(df)

    # Fill remaining NaN in string cols with "Unknown"
    for col in df.select_dtypes(include="object").columns:
        df[col].fillna("Unknown", inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df


def auto_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    date_hints = ["date", "time", "timestamp", "created", "recorded", "updated"]
    for col in df.columns:
        if any(h in col.lower() for h in date_hints):
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            except Exception:
                pass
    return df


def auto_parse_numerics(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object").columns:
        try:
            converted = pd.to_numeric(df[col].str.replace(",", "").str.replace("$", "").str.strip(), errors="coerce")
            if converted.notna().sum() / len(df) > 0.6:  # >60% parseable → numeric
                df[col] = converted
        except Exception:
            pass
    return df


# ─── Summary ──────────────────────────────────────────────────────────────────
def get_csv_summary(df: pd.DataFrame, col_roles: dict) -> dict:
    """Return a compact summary dict of the loaded dataset."""
    summary = {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "columns": list(df.columns),
        "col_roles": col_roles,
        "missing_pct": round(df.isnull().mean().mean() * 100, 2),
        "date_range": None,
        "numeric_summary": {},
    }
    if col_roles["datetime"]:
        dt_col = col_roles["datetime"][0]
        try:
            mn = df[dt_col].min()
            mx = df[dt_col].max()
            summary["date_range"] = f"{mn} → {mx}"
        except Exception:
            pass
    for col in col_roles["numeric"][:8]:  # top 8 numeric cols
        try:
            summary["numeric_summary"][col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "sum": float(df[col].sum()),
            }
        except Exception:
            pass
    return summary