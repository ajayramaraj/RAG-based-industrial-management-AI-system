"""
analytics.py — Industrial KPI computation engine.
Computes: MTBF, MTTR, health scores, failure rates, cost analysis, technician metrics.
"""
import numpy as np
import pandas as pd
from scripts.utils import detect_column_roles


def compute_full_analytics(df: pd.DataFrame) -> dict:
    col_roles = detect_column_roles(df)
    result = {
        "total_rows":       len(df),
        "total_cols":       len(df.columns),
        "columns":          list(df.columns),
        "col_roles":        col_roles,
        "missing_pct":      round(df.isnull().mean().mean() * 100, 2),
        "date_range":       None,
        "machine_stats":    {},
        "machine_health":   {},
        "failure_analysis": {},
        "cost_analysis":    {},
        "technician_stats": {},
        "top_failures":     [],
        "numeric_summary":  {},
        "trend_data":       {},
    }

    # Date range
    if col_roles["datetime"]:
        dt_col = col_roles["datetime"][0]
        try:
            mn = df[dt_col].dropna().min()
            mx = df[dt_col].dropna().max()
            result["date_range"] = f"{mn} → {mx}"
        except Exception:
            pass

    # Numeric summary
    for col in col_roles["numeric"]:
        try:
            s = df[col].dropna()
            result["numeric_summary"][col] = {
                "min":  float(s.min()),
                "max":  float(s.max()),
                "mean": float(s.mean()),
                "sum":  float(s.sum()),
                "std":  float(s.std()),
            }
        except Exception:
            pass

    # Machine-level stats
    if col_roles["machine_id"]:
        mid_col = col_roles["machine_id"][0]
        ms, mh = _compute_machine_stats(df, mid_col, col_roles)
        result["machine_stats"]  = ms
        result["machine_health"] = mh

    # Failure analysis — uses value_counts(), never __dummy__
    if col_roles["status"]:
        result["failure_analysis"] = _compute_failure_analysis(df, col_roles)
        result["top_failures"]     = _get_top_failures(df, col_roles)

    # Cost analysis
    result["cost_analysis"] = _compute_cost_analysis(df, col_roles)

    # Technician stats
    if col_roles["technician"]:
        result["technician_stats"] = _compute_technician_stats(df, col_roles)

    # Trend data
    if col_roles["datetime"]:
        result["trend_data"] = _compute_trends(df, col_roles)

    return result


# ─── Machine Stats ─────────────────────────────────────────────────────────────
def _compute_machine_stats(df: pd.DataFrame, mid_col: str, col_roles: dict):
    machines      = df[mid_col].dropna().unique()
    machine_stats = {}
    machine_health = {}

    status_col   = col_roles["status"][0]   if col_roles["status"]   else None
    dt_col       = col_roles["datetime"][0]  if col_roles["datetime"] else None
    downtime_col = _find_col(df, col_roles["numeric"], ["downtime","duration","hours","minutes"])
    cost_col     = _find_col(df, col_roles["numeric"], ["cost","expense","price","amount"])

    for machine in machines:
        mdf   = df[df[mid_col] == machine]
        stats = {"total_events": len(mdf), "machine_id": machine}

        # Failure rate
        if status_col:
            fail_kw  = ["fail","fault","error","alarm","breakdown","critical","down"]
            failures = mdf[status_col].astype(str).str.lower().apply(
                lambda x: any(k in x for k in fail_kw)
            )
            stats["failure_count"] = int(failures.sum())
            stats["failure_rate"]  = round(float(failures.mean()) * 100, 2)
        else:
            stats["failure_count"] = 0
            stats["failure_rate"]  = 0.0

        # MTBF
        if dt_col and len(mdf) > 1:
            try:
                times = mdf[dt_col].dropna().sort_values()
                gaps  = times.diff().dropna().dt.total_seconds() / 3600
                stats["mtbf_hours"] = round(float(gaps.mean()), 2)
            except Exception:
                stats["mtbf_hours"] = None

        # Downtime
        if downtime_col:
            try:
                stats["total_downtime"] = float(mdf[downtime_col].sum())
                stats["avg_downtime"]   = float(mdf[downtime_col].mean())
            except Exception:
                pass

        # Cost
        if cost_col:
            try:
                stats["total_cost"] = float(mdf[cost_col].sum())
                stats["avg_cost"]   = float(mdf[cost_col].mean())
            except Exception:
                pass

        health = _compute_health_score(stats)
        stats["health_score"]          = health
        machine_health[str(machine)]   = health
        machine_stats[str(machine)]    = stats

    return machine_stats, machine_health


def _compute_health_score(stats: dict) -> float:
    score  = 100.0
    score -= min(stats.get("failure_rate", 0) * 1.5, 60)
    dt     = stats.get("avg_downtime", 0) or 0
    score -= min(dt / 10, 20)
    return max(round(score, 1), 0.0)


# ─── Failure Analysis ──────────────────────────────────────────────────────────
def _compute_failure_analysis(df: pd.DataFrame, col_roles: dict) -> dict:
    status_col = col_roles["status"][0]
    # Use value_counts() — safe, no __dummy__ needed
    series = df[status_col].astype(str).str.strip()
    counts = series.value_counts().head(20)
    return {
        "status_distribution": counts.to_dict(),
        "unique_statuses":     int(series.nunique()),
        "most_common":         str(counts.index[0]) if len(counts) else "N/A",
    }


def _get_top_failures(df: pd.DataFrame, col_roles: dict) -> list:
    status_col = col_roles["status"][0]
    counts = df[status_col].astype(str).str.strip().value_counts().head(10)
    return [f"{k}: {v} occurrences" for k, v in counts.items()]


# ─── Cost Analysis ─────────────────────────────────────────────────────────────
def _compute_cost_analysis(df: pd.DataFrame, col_roles: dict) -> dict:
    result   = {}
    cost_col = _find_col(df, col_roles["numeric"], ["cost","expense","price","amount","budget"])
    if not cost_col:
        return result
    try:
        s = df[cost_col].dropna()
        result["total_cost"] = float(s.sum())
        result["avg_cost"]   = float(s.mean())
        result["max_cost"]   = float(s.max())
        result["min_cost"]   = float(s.min())
        result["cost_col"]   = cost_col

        mid_col = col_roles["machine_id"][0] if col_roles["machine_id"] else None
        if mid_col:
            by_machine = df.groupby(mid_col)[cost_col].sum().sort_values(ascending=False)
            result["cost_by_machine"]       = by_machine.head(10).to_dict()
            result["highest_cost_machine"]  = str(by_machine.index[0]) if len(by_machine) else None

        status_col = col_roles["status"][0] if col_roles["status"] else None
        if status_col:
            by_status = df.groupby(status_col)[cost_col].sum().sort_values(ascending=False)
            result["cost_by_status"] = by_status.head(10).to_dict()
    except Exception as e:
        result["error"] = str(e)
    return result


# ─── Technician Stats ──────────────────────────────────────────────────────────
def _compute_technician_stats(df: pd.DataFrame, col_roles: dict) -> dict:
    tech_col  = col_roles["technician"][0]
    cost_col  = _find_col(df, col_roles["numeric"], ["cost","expense","price","amount"])
    dt_col    = _find_col(df, col_roles["numeric"], ["downtime","duration","hours","minutes"])
    result    = {}
    try:
        counts = df[tech_col].astype(str).value_counts()
        result["assignments"] = counts.head(10).to_dict()
        result["busiest"]     = str(counts.index[0]) if len(counts) else "N/A"
        if cost_col:
            result["cost_by_tech"] = (
                df.groupby(tech_col)[cost_col].sum()
                  .sort_values(ascending=False).head(10).to_dict()
            )
        if dt_col:
            result["avg_downtime_by_tech"] = (
                df.groupby(tech_col)[dt_col].mean()
                  .sort_values().head(10).to_dict()
            )
    except Exception as e:
        result["error"] = str(e)
    return result


# ─── Trend Data ────────────────────────────────────────────────────────────────
def _compute_trends(df: pd.DataFrame, col_roles: dict) -> dict:
    dt_col = col_roles["datetime"][0]
    result = {}
    try:
        temp = df.copy()
        temp["__month__"] = temp[dt_col].dt.to_period("M").astype(str)
        monthly = temp.groupby("__month__").size().reset_index(name="count")
        result["monthly_events"] = monthly.to_dict("records")

        cost_col = _find_col(df, col_roles["numeric"], ["cost","expense","price","amount"])
        if cost_col:
            mc = temp.groupby("__month__")[cost_col].sum().reset_index()
            mc.columns = ["__month__", "cost"]
            result["monthly_cost"] = mc.to_dict("records")
    except Exception as e:
        result["error"] = str(e)
    return result


# ─── Helper ────────────────────────────────────────────────────────────────────
def _find_col(df: pd.DataFrame, candidates: list, keywords: list):
    for col in candidates:
        if any(k in col.lower() for k in keywords):
            return col
    return candidates[0] if candidates else None


# ─── Standalone frequency helper (used by some dashboard versions) ─────────────
def failure_frequency(df: pd.DataFrame, col_roles: dict) -> pd.DataFrame:
    """Safe failure frequency — uses value_counts, never __dummy__."""
    if not col_roles.get("status"):
        return pd.DataFrame(columns=["status", "count"])
    status_col = col_roles["status"][0]
    counts = df[status_col].astype(str).str.strip().value_counts().reset_index()
    counts.columns = ["status", "count"]
    return counts


# ─── Aliases for compatibility with various dashboard versions ─────────────────
def compute_summary(df: pd.DataFrame, col_roles: dict) -> dict:
    """Alias for compute_full_analytics — for compatibility."""
    return compute_full_analytics(df)

def get_machine_health(df: pd.DataFrame, col_roles: dict) -> dict:
    mid_col = col_roles["machine_id"][0] if col_roles["machine_id"] else None
    if not mid_col:
        return {}
    ms, mh = _compute_machine_stats(df, mid_col, col_roles)
    return mh

def get_failure_stats(df: pd.DataFrame, col_roles: dict) -> dict:
    if not col_roles["status"]:
        return {}
    return _compute_failure_analysis(df, col_roles)

def get_cost_analysis(df: pd.DataFrame, col_roles: dict) -> dict:
    return _compute_cost_analysis(df, col_roles)

def get_technician_stats(df: pd.DataFrame, col_roles: dict) -> dict:
    return _compute_technician_stats(df, col_roles)

def get_trend_data(df: pd.DataFrame, col_roles: dict) -> dict:
    return _compute_trends(df, col_roles)