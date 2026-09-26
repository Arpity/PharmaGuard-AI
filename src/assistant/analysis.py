"""Deterministic batch analysis. ALL numbers the assistant reports are computed here with
pandas/numpy; the LLM only receives (and must quote) these results."""
from __future__ import annotations

import difflib
import re
from typing import Optional

import numpy as np
import pandas as pd

PARAMS = ["Temperature", "Humidity", "pH", "Dissolution", "Assay", "Impurity_Level", "Yield_Percentage",
          "Cycle_Time_Hrs", "Batch_Size_Kg"]
UNITS = {"Temperature": "°C", "Humidity": "%", "pH": "", "Dissolution": "%", "Assay": "%", "Impurity_Level": "%",
         "Yield_Percentage": "%", "Cycle_Time_Hrs": "h", "Batch_Size_Kg": "kg"}
SIMILARITY_FEATURES = ["Temperature", "Humidity", "pH", "Dissolution", "Assay", "Impurity_Level",
                       "Yield_Percentage", "Cycle_Time_Hrs"]
CRITICAL = ["Dissolution", "Assay", "Impurity_Level", "Yield_Percentage"]
Z_WATCH, Z_OUTLIER = 2.0, 3.0
MIN_GROUP, MIN_EQUIPMENT, K_SIMILAR = 30, 15, 25
BATCH_ID_RE = re.compile(r"\bB\d{5}\b", re.IGNORECASE)


def extract_batch_id(text: str) -> Optional[str]:
    m = BATCH_ID_RE.search(text or "")
    return m.group(0).upper() if m else None


def suggest_batch_ids(df: pd.DataFrame, batch_id: str, n: int = 5) -> list[str]:
    return difflib.get_close_matches(str(batch_id).strip().upper(), df["Batch_ID"].astype(str).tolist(), n=n, cutoff=0.6)


def _fmt(v: float, nd: int = 2) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.{nd}f}".rstrip("0").rstrip(".") if nd else f"{v:.0f}"


def _r(v, nd: int = 3):
    return None if v is None or pd.isna(v) else round(float(v), nd)


def _fail_rate(status: pd.Series) -> Optional[float]:
    known = int((status != "Unknown").sum())
    return float((status == "Fail").sum()) / known * 100 if known else None


def _group_context(df: pd.DataFrame, col: str, value, overall: float, min_n: int) -> Optional[dict]:
    sub = df[df[col] == value]
    if value == "Unknown" or len(sub) < min_n or overall is None:
        return None
    fr = _fail_rate(sub["Quality_Status"])
    return {"group": str(value), "batches": int(len(sub)), "fail_rate_pct": _r(fr, 1),
            "overall_fail_rate_pct": _r(overall, 1), "ratio_to_overall": _r(fr / overall, 2) if overall else None}


def analyze_batch(df: pd.DataFrame, batch_id: str, cfg: dict) -> Optional[dict]:
    bid = str(batch_id).strip().upper()
    hit = df.index[df["Batch_ID"].astype(str).str.upper() == bid]
    if len(hit) == 0:
        return None
    row = df.loc[hit[0]]
    others = df.drop(index=hit[0])
    peers = others[others["Product_Name"] == row["Product_Name"]]
    specs = cfg["specs"]
    split = lambda v: [c for c in ("" if pd.isna(v) else str(v)).split(";") if c]
    imputed, flagged = split(row.get("Imputed_Columns")), split(row.get("Outlier_Columns"))

    params = []
    for p in PARAMS:
        v, pv = row[p], peers[p].dropna()
        rec = {"parameter": p, "unit": UNITS[p], "value": _r(v), "imputed": p in imputed,
               "spec_low": None, "spec_high": None, "in_spec": None, "breach": None, "breach_amount": None,
               "peer_n": int(len(pv)), "peer_mean": _r(pv.mean()), "peer_std": _r(pv.std()), "z_score": None,
               "percentile": None, "flag": "missing" if pd.isna(v) else "normal"}
        if p in specs:
            rec["spec_low"], rec["spec_high"] = specs[p]
        if not pd.isna(v):
            if p in specs:
                lo, hi = specs[p]
                rec["in_spec"] = bool(lo <= v <= hi)
                if v < lo:
                    rec["breach"], rec["breach_amount"] = "below", _r(lo - v)
                elif v > hi:
                    rec["breach"], rec["breach_amount"] = "above", _r(v - hi)
            if len(pv) > 1 and pv.std() > 0:
                rec["z_score"] = _r((v - pv.mean()) / pv.std(), 2)
            if len(pv):
                rec["percentile"] = _r((pv < v).mean() * 100, 1)
            z = abs(rec["z_score"]) if rec["z_score"] is not None else 0
            if rec["in_spec"] is False:
                rec["flag"] = "out_of_spec"
            elif z >= Z_OUTLIER or (p in flagged and not rec["imputed"]):
                rec["flag"] = "statistical_outlier"
            elif z >= Z_WATCH:
                rec["flag"] = "watch"
            if rec["imputed"]:
                rec["flag"] = "imputed_estimate"
        rec["spec_text"] = (f"{_fmt(rec['spec_low'])}-{_fmt(rec['spec_high'])}{UNITS[p]}"
                            if rec["spec_low"] is not None else "no specification")
        rec["value_text"] = "missing" if pd.isna(v) else f"{_fmt(v, 3)}{UNITS[p]}"
        params.append(rec)

    order = {"out_of_spec": 0, "statistical_outlier": 1, "watch": 2}
    abnormal = sorted([q for q in params if q["flag"] in order], key=lambda q: (order[q["flag"]], -abs(q["z_score"] or 0)))
    oos = [q["parameter"] for q in params if q["flag"] == "out_of_spec"]
    missing_cqa = [p for p in CRITICAL if pd.isna(row[p])]

    # deviations
    dv = peers["Deviation_Count"].dropna()
    dcount = None if pd.isna(row["Deviation_Count"]) else int(row["Deviation_Count"])
    deviations = {"count": dcount, "peer_mean": _r(dv.mean(), 2), "peer_n": int(len(dv)),
                  "percentile": _r((dv < dcount).mean() * 100, 1) if dcount is not None and len(dv) else None,
                  "flag": None if dcount is None else "high" if dcount >= 5 else "elevated" if dcount >= 3 else "normal"}

    # risk context (historical failure rates of the batch's groups)
    overall_fr = _fail_rate(others["Quality_Status"])
    ctx = {"plant": _group_context(others, "Plant", row["Plant"], overall_fr, MIN_GROUP),
           "shift": _group_context(others, "Operator_Shift", row["Operator_Shift"], overall_fr, MIN_GROUP),
           "product": _group_context(others, "Product_Name", row["Product_Name"], overall_fr, MIN_GROUP),
           "equipment": _group_context(others, "Equipment_ID", row["Equipment_ID"], overall_fr, MIN_EQUIPMENT)}
    indicators = []
    if oos:
        indicators.append(f"{len(oos)} attribute(s) outside specification: " + ", ".join(
            f"{q['parameter']} {q['value_text']} ({q['breach']} limit {q['spec_text']})" for q in params if q["flag"] == "out_of_spec"))
    if any(q["flag"] == "statistical_outlier" for q in params):
        indicators.append("Statistically unusual readings vs same-product history: " + ", ".join(
            f"{q['parameter']} {q['value_text']} (z={q['z_score']})" for q in params if q["flag"] == "statistical_outlier"))
    if deviations["flag"] in ("elevated", "high"):
        indicators.append(f"{dcount} deviations recorded (peer average {deviations['peer_mean']}); "
                          f"flagged '{deviations['flag']}' (3+ elevated, 5+ high)")
    if missing_cqa:
        indicators.append("Critical result(s) not available: " + ", ".join(missing_cqa))
    for name, c in ctx.items():
        if c and c["ratio_to_overall"] and c["ratio_to_overall"] >= 1.25 and c["fail_rate_pct"] > 0:
            indicators.append(f"{name.title()} {c['group']} historically fails {c['fail_rate_pct']}% of batches "
                              f"vs {c['overall_fail_rate_pct']}% overall ({c['batches']} batches)")

    # history: similar batches and outcomes for the same breaches
    similar = _similar_batches(peers, row)
    sim_fr = _fail_rate(similar["Quality_Status"]) if len(similar) else None
    oos_hist = []
    for p in oos:
        lo, hi = specs[p]
        sub = others[(others[p] < lo) | (others[p] > hi)]
        oos_hist.append({"parameter": p, "batches_out_of_spec": int(len(sub)),
                         "fail_rate_pct": _r(_fail_rate(sub["Quality_Status"]), 1) if len(sub) else None})
    history = {"peer_group": f"Product {row['Product_Name']}", "peer_batches": int(len(peers)),
               "peer_fail_rate_pct": _r(_fail_rate(peers["Quality_Status"]), 1),
               "peer_risk_rate_pct": _r(float((peers["Quality_Status"].isin(["Under Review", "Fail"])).mean() * 100), 1),
               "overall_fail_rate_pct": _r(overall_fr, 1),
               "similar_batches_used": int(len(similar)), "similar_fail_rate_pct": _r(sim_fr, 1),
               "closest_batches": [{"Batch_ID": r.Batch_ID, "Quality_Status": r.Quality_Status,
                                    "Manufacturing_Date": str(r.Manufacturing_Date)[:10], "distance": _r(r.distance, 2)}
                                   for r in similar.head(5).itertuples()],
               "out_of_spec_history": oos_hist}

    return {
        "batch": {"Batch_ID": row["Batch_ID"], "Product_Name": row["Product_Name"], "Dosage_Form": row["Dosage_Form"],
                  "Manufacturing_Date": str(row["Manufacturing_Date"])[:10], "Plant": row["Plant"],
                  "Operator_Shift": row["Operator_Shift"], "Equipment_ID": row["Equipment_ID"],
                  "Quality_Status": row["Quality_Status"]},
        "data_notes": {"imputed_parameters": imputed, "missing_critical_results": missing_cqa,
                       "statistically_flagged_columns": flagged},
        "parameters": params, "abnormal_parameters": [q["parameter"] for q in abnormal] + (
            ["Deviation_Count"] if deviations["flag"] in ("elevated", "high") or "Deviation_Count" in flagged else []),
        "out_of_spec_parameters": oos, "deviations": deviations, "risk_context": ctx,
        "risk_indicators": indicators, "history": history,
    }


def _similar_batches(peers: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    feats = SIMILARITY_FEATURES
    if len(peers) < MIN_GROUP:
        return peers.iloc[0:0].assign(distance=[])
    mu, sd = peers[feats].mean(), peers[feats].std().replace(0, np.nan)
    z_peers, z_row = (peers[feats] - mu) / sd, (row[feats].astype(float) - mu) / sd
    diff = z_peers.sub(z_row, axis=1)
    cnt = diff.notna().sum(axis=1)
    dist = np.sqrt((diff ** 2).sum(axis=1) / cnt.replace(0, np.nan))
    ok = cnt >= 5
    out = peers[ok].assign(distance=dist[ok]).sort_values("distance")
    return out.head(K_SIMILAR)
