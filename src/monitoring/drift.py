"""Data drift and schema-change detection (deterministic, pandas/numpy only)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data_quality.rules import is_missing, parse_dates, parse_numeric

EPS = 1e-4


def psi_from_counts(ref: np.ndarray, cur: np.ndarray) -> float:
    r, c = ref / max(ref.sum(), 1), cur / max(cur.sum(), 1)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> Optional[float]:
    """Population Stability Index of `cur` vs `ref` using reference-quantile bins. <0.1 stable, 0.1-0.25 moderate, >0.25 major."""
    ref, cur = ref.dropna().astype(float), cur.dropna().astype(float)
    if len(ref) < 2 or len(cur) < 2:
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0 if ref.nunique() == cur.nunique() == 1 and ref.iloc[0] == cur.iloc[0] else psi_from_counts(
            np.array([len(ref), 0.0]), np.array([float((cur == ref.iloc[0]).sum()), float((cur != ref.iloc[0]).sum())]))
    edges[0], edges[-1] = -np.inf, np.inf
    return psi_from_counts(np.histogram(ref, edges)[0].astype(float), np.histogram(cur, edges)[0].astype(float))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> Optional[float]:
    ref, cur = ref.dropna().astype(str), cur.dropna().astype(str)
    if len(ref) < 2 or len(cur) < 2:
        return None
    cats = sorted(set(ref) | set(cur))
    return psi_from_counts(np.array([(ref == k).sum() for k in cats], float), np.array([(cur == k).sum() for k in cats], float))


def split_windows(df: pd.DataFrame, reference_fraction: float, date_col: str = "Manufacturing_Date"):
    d = df.dropna(subset=[date_col]).sort_values(date_col)
    cut = int(len(d) * reference_fraction)
    return d.iloc[:cut], d.iloc[cut:]


def drift_report(df: pd.DataFrame, mon_cfg: dict) -> pd.DataFrame:
    """PSI per monitored feature between the reference window (earliest batches) and the current window (latest)."""
    dc = mon_cfg["data"]
    ref, cur = split_windows(df, mon_cfg["windows"]["drift_reference_fraction"])
    rows = []
    if len(ref) < dc["min_rows_per_window"] or len(cur) < dc["min_rows_per_window"]:
        return pd.DataFrame(columns=["feature", "kind", "psi", "reference_rows", "current_rows", "status"])
    warn, crit = mon_cfg["slo"]["drift_psi_max"]["warn"], mon_cfg["slo"]["drift_psi_max"]["critical"]
    feats = [(c, "numeric") for c in dc["monitored_numeric"] if c in df] + [(c, "categorical") for c in dc["monitored_categories"] if c in df]
    for f, kind in feats:
        psi = psi_numeric(ref[f], cur[f], dc["drift_bins"]) if kind == "numeric" else psi_categorical(ref[f], cur[f])
        rows.append({"feature": f, "kind": kind, "psi": None if psi is None else round(psi, 4), "reference_rows": len(ref), "current_rows": len(cur),
                     "status": "unknown" if psi is None else "critical" if psi > crit else "warn" if psi > warn else "ok"})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


def missingness_drift(df: pd.DataFrame, mon_cfg: dict) -> pd.DataFrame:
    """Change in per-column missing rate between reference and current windows (percentage points)."""
    ref, cur = split_windows(df, mon_cfg["windows"]["drift_reference_fraction"])
    rows = [{"column": c, "reference_missing_pct": round(float(is_missing(ref[c]).mean() * 100), 2), "current_missing_pct": round(float(is_missing(cur[c]).mean() * 100), 2)}
            for c in df.columns if c not in ("Source_Row",) and len(ref) and len(cur)]
    out = pd.DataFrame(rows)
    if not out.empty:
        out["change_pp"] = (out["current_missing_pct"] - out["reference_missing_pct"]).round(2)
    return out.sort_values("change_pp", key=abs, ascending=False).reset_index(drop=True) if not out.empty else out


# ---- schema ---------------------------------------------------------------------------------------------
def infer_schema(raw: pd.DataFrame) -> dict:
    """Column -> inferred kind (number / date / text). Works on raw all-text frames."""
    out = {}
    for c in raw.columns:
        s, present = raw[c], ~is_missing(raw[c])
        if not present.any():
            out[c] = "empty"
        elif (parse_numeric(s)["status"].isin(["ok", "converted"]) & present).sum() >= 0.9 * present.sum():
            out[c] = "number"
        elif (parse_dates(s)["status"].isin(["ok", "converted"]) & present).sum() >= 0.9 * present.sum():
            out[c] = "date"
        else:
            out[c] = "text"
    return out


def schema_diff(current: dict, baseline: dict) -> dict:
    cols_now, cols_then = list(current), list(baseline)
    return {"added": [c for c in cols_now if c not in baseline], "removed": [c for c in cols_then if c not in current],
            "type_changes": [{"column": c, "was": baseline[c], "now": current[c]} for c in cols_now if c in baseline and baseline[c] != current[c]],
            "reordered": [c for c in cols_now if c in baseline] != [c for c in cols_then if c in current]}


def schema_change_count(diff: dict) -> int:
    return len(diff["added"]) + len(diff["removed"]) + len(diff["type_changes"]) + int(diff["reordered"])


def schema_baseline_path(root) -> Path:
    return Path(root) / "config" / "data_schema_baseline.json"


def register_schema_baseline(raw: pd.DataFrame, root) -> dict:
    b = {"columns": infer_schema(raw), "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rows": len(raw)}
    schema_baseline_path(root).write_text(json.dumps(b, indent=1))
    return b


def load_schema_baseline(root) -> Optional[dict]:
    p = schema_baseline_path(root)
    return json.loads(p.read_text()) if p.exists() else None


# ---- simulation helpers (used only for demonstrations; never written back) ---------------------------------
def simulate_drift(df: pd.DataFrame, mon_cfg: dict, strength: float = 1.0) -> pd.DataFrame:
    """Shift the latest window: hotter, more humid, lower assay, more failures, more missing dissolution."""
    d = df.copy()
    _, cur = split_windows(d, mon_cfg["windows"]["drift_reference_fraction"])
    idx = cur.index
    d.loc[idx, "Temperature"] = d.loc[idx, "Temperature"] + 2.5 * strength
    d.loc[idx, "Humidity"] = d.loc[idx, "Humidity"] + 9 * strength
    d.loc[idx, "Assay"] = d.loc[idx, "Assay"] - 1.8 * strength
    rng = np.random.default_rng(1)
    hit = rng.random(len(idx)) < 0.2 * strength
    d.loc[idx[hit], "Dissolution"] = np.nan
    d.loc[idx[rng.random(len(idx)) < 0.15 * strength], "Quality_Status"] = "Fail"
    return d


def simulate_schema_change(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop one column, rename another and add a new one."""
    d = raw.drop(columns=["Cycle_Time_Hrs"]).rename(columns={"Humidity": "Relative_Humidity"})
    d["Lot_Supplier"] = "SUP-01"
    return d
