"""Independent 'oracle' for expected findings. Deliberately re-implemented with plain pandas and NO imports from
src.assistant, so the golden dataset checks the analysis code instead of echoing it."""
from __future__ import annotations

import pandas as pd

CQA = ["Dissolution", "Assay", "Impurity_Level", "Yield_Percentage"]


def expected_analysis(df: pd.DataFrame, batch_id: str, specs: dict) -> dict:
    row = df[df["Batch_ID"] == batch_id].iloc[0]
    peers = df[(df["Batch_ID"] != batch_id) & (df["Product_Name"] == row["Product_Name"])]
    checks = []
    for p, (lo, hi) in specs.items():
        v = row[p]
        if pd.notna(v) and (v < lo or v > hi):
            checks.append({"parameter": p, "value": round(float(v), 3), "flag": "out_of_spec",
                           "breach": "below" if v < lo else "above",
                           "breach_amount": round(float(lo - v if v < lo else v - hi), 3)})
    dev = row["Deviation_Count"]
    dev_flag = None if pd.isna(dev) else "high" if dev >= 5 else "elevated" if dev >= 3 else "normal"
    known = peers[peers["Quality_Status"] != "Unknown"]
    split = lambda v: [] if pd.isna(v) else [c for c in str(v).split(";") if c]
    return {
        "status": row["Quality_Status"], "product": row["Product_Name"],
        "out_of_spec_parameters": sorted(c["parameter"] for c in checks), "checks": checks,
        "abnormal_include": sorted([c["parameter"] for c in checks] + (["Deviation_Count"] if dev_flag in ("elevated", "high") else [])),
        "deviation_count": None if pd.isna(dev) else int(dev), "deviation_flag": dev_flag,
        "peer_batches": int(len(peers)),
        "peer_fail_rate_pct": round(float((known["Quality_Status"] == "Fail").mean() * 100), 1) if len(known) else None,
        "imputed_parameters": sorted(split(row.get("Imputed_Columns"))),
        "missing_critical_results": [c for c in CQA if pd.isna(row[c])],
    }
