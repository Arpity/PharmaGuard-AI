"""Deterministic analytics on the cleaned batch data. Every number shown on the dashboard
(KPIs, chart aggregates, insight text) is computed here with plain pandas - no models, no LLM.

Conventions
- Failure rate = Fail / batches with a known Quality_Status (Unknown excluded).
- Risk batches = Under Review + Fail.
- Averages ignore missing values (critical results are never imputed, see cleaning step).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

STATUS_ORDER = ["Pass", "Under Review", "Fail", "Unknown"]
METRICS = {"Dissolution": "%", "Assay": "%", "Impurity_Level": "%", "Yield_Percentage": "%"}
MIN_GROUP = 30   # smallest group used when ranking products / plants / shifts in insights


def load_clean(path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=True)
    df["Manufacturing_Date"] = pd.to_datetime(df["Manufacturing_Date"], errors="coerce")
    for c in ("Outlier_Flag", "Has_Missing_CQA"):
        if c in df:
            df[c] = df[c].astype(str).str.lower().eq("true")
    df["Outlier_Columns"] = df.get("Outlier_Columns", pd.Series("", index=df.index)).fillna("")
    return df


def apply_filters(df: pd.DataFrame, products=None, plants=None, statuses=None,
                  date_range: Optional[tuple] = None, include_undated: bool = True) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    if products is not None:
        m &= df["Product_Name"].isin(products)
    if plants is not None:
        m &= df["Plant"].isin(plants)
    if statuses is not None:
        m &= df["Quality_Status"].isin(statuses)
    if date_range is not None:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        d = df["Manufacturing_Date"]
        m &= d.between(start, end) | (d.isna() & include_undated)
    return df[m]


def _rate(num: int, den: int) -> float:
    return float(num) / den * 100 if den else float("nan")


def compute_kpis(df: pd.DataFrame) -> dict:
    status = df["Quality_Status"]
    known = int((status != "Unknown").sum())
    n_pass, n_review, n_fail = (int((status == s).sum()) for s in ("Pass", "Under Review", "Fail"))
    mean = lambda c: float(df[c].mean()) if df[c].notna().any() else float("nan")
    return {
        "total_batches": len(df), "pass_batches": n_pass, "risk_batches": n_review + n_fail,
        "review_batches": n_review, "fail_batches": n_fail, "unknown_status": len(df) - known,
        "failure_rate": _rate(n_fail, known), "risk_rate": _rate(n_review + n_fail, known),
        "avg_dissolution": mean("Dissolution"), "avg_assay": mean("Assay"),
        "avg_impurity": mean("Impurity_Level"), "avg_yield": mean("Yield_Percentage"),
        "total_deviations": int(df["Deviation_Count"].sum()),
        "avg_deviations": mean("Deviation_Count"),
        "batches_with_deviation": int((df["Deviation_Count"] > 0).sum()),
    }


def group_summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """One row per group: batch counts by status, rates and metric averages."""
    g = df.groupby(by, dropna=False)
    out = pd.DataFrame({"batches": g.size()})
    for s in ("Pass", "Under Review", "Fail"):
        out[s] = g["Quality_Status"].apply(lambda x, s=s: int((x == s).sum()))
    known = out[["Pass", "Under Review", "Fail"]].sum(axis=1)
    out["fail_rate"] = (out["Fail"] / known.replace(0, np.nan) * 100)
    out["risk_rate"] = ((out["Under Review"] + out["Fail"]) / known.replace(0, np.nan) * 100)
    for c in list(METRICS) + ["Deviation_Count"]:
        out[f"avg_{c}"] = g[c].mean()
    out["total_deviations"] = g["Deviation_Count"].sum()
    return out.reset_index()


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["Manufacturing_Date"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["month", "batches", "fail_rate", "risk_rate"])
    d["month"] = d["Manufacturing_Date"].dt.to_period("M").dt.to_timestamp()
    out = group_summary(d, "month").sort_values("month")
    return out


def spec_compliance(df: pd.DataFrame, specs: dict) -> pd.DataFrame:
    """Share of batches (with a result) inside specification limits for each attribute."""
    rows = []
    for col in ("Dissolution", "Assay", "Impurity_Level", "Yield_Percentage", "Temperature", "Humidity", "pH"):
        lo, hi = specs[col]
        v = df[col].dropna()
        rows.append({"attribute": col, "limits": f"{lo:g} - {hi:g}", "tested": len(v),
                     "in_spec_pct": _rate(int(v.between(lo, hi).sum()), len(v))})
    return pd.DataFrame(rows)


def anomaly_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Batch_ID", "Manufacturing_Date", "Product_Name", "Plant", "Operator_Shift", "Outlier_Columns",
            "Dissolution", "Assay", "Impurity_Level", "Yield_Percentage", "Deviation_Count", "Quality_Status"]
    return df.loc[df["Outlier_Flag"], cols].sort_values("Manufacturing_Date", ascending=False)


def _pct(x: float) -> str:
    return f"{x:.1f}%"


def generate_insights(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Concise, rule-based business insights. Each dict: {level, title, text}."""
    if df.empty:
        return [{"level": "info", "title": "No data", "text": "No batches match the current filters."}]
    k, out = compute_kpis(df), []
    add = lambda level, title, text: out.append({"level": level, "title": title, "text": text})

    level = "high" if k["failure_rate"] >= 8 else "medium" if k["failure_rate"] >= 4 else "low"
    add(level, "Overall quality",
        f"{k['total_batches']:,} batches reviewed: {k['pass_batches']:,} passed, {k['review_batches']:,} are under review and "
        f"{k['fail_batches']:,} failed. Failure rate is {_pct(k['failure_rate'])} and {_pct(k['risk_rate'])} of batches carry some risk.")

    for dim, label in (("Product_Name", "product"), ("Plant", "plant"), ("Operator_Shift", "shift")):
        g = group_summary(df, dim)
        g = g[(g["batches"] >= MIN_GROUP) & (g[dim] != "Unknown")]
        if len(g) >= 2:
            worst, best = g.loc[g["fail_rate"].idxmax()], g.loc[g["fail_rate"].idxmin()]
            if worst["fail_rate"] > best["fail_rate"]:
                ratio = worst["fail_rate"] / best["fail_rate"] if best["fail_rate"] > 0 else float("inf")
                tail = f" - {ratio:.1f}x the best {label} ({best[dim]}, {_pct(best['fail_rate'])})" if np.isfinite(ratio) else \
                    f" versus {_pct(best['fail_rate'])} for {best[dim]}"
                add("high" if worst["fail_rate"] > k["failure_rate"] * 1.3 else "medium",
                    f"Highest-risk {label}",
                    f"{worst[dim]} has the highest failure rate at {_pct(worst['fail_rate'])} ({int(worst['Fail'])} of "
                    f"{int(worst['batches'])} batches){tail}.")

    tr = monthly_trend(df)
    if len(tr) >= 6:
        recent, prior = tr.tail(3), tr.iloc[-6:-3]
        r = _rate(int(recent["Fail"].sum()), int(recent[["Pass", "Under Review", "Fail"]].sum().sum()))
        p = _rate(int(prior["Fail"].sum()), int(prior[["Pass", "Under Review", "Fail"]].sum().sum()))
        if np.isfinite(r) and np.isfinite(p):
            d = r - p
            word = "up" if d > 0.5 else "down" if d < -0.5 else "stable"
            add("high" if d > 2 else "medium" if d > 0.5 else "low", "Recent trend",
                f"Failure rate over the last 3 months is {_pct(r)}, {word} {'' if word == 'stable' else f'{abs(d):.1f} pts '}"
                f"versus the prior 3 months ({_pct(p)}).")

    sc = spec_compliance(df, cfg["specs"]).set_index("attribute")
    worst_attr = sc.loc[["Dissolution", "Assay", "Impurity_Level", "Yield_Percentage"], "in_spec_pct"].idxmin()
    add("medium", "Specification compliance",
        f"{worst_attr.replace('_', ' ')} is the attribute most often out of specification: only "
        f"{_pct(sc.loc[worst_attr, 'in_spec_pct'])} of tested batches are within {sc.loc[worst_attr, 'limits']}. "
        f"Average assay is {k['avg_assay']:.1f}% and average impurity {k['avg_impurity']:.2f}%.")

    add("medium" if k["avg_deviations"] > 1 else "low", "Deviations",
        f"{k['total_deviations']:,} deviations were logged; {_pct(_rate(k['batches_with_deviation'], k['total_batches']))} of batches had at least one "
        f"(average {k['avg_deviations']:.2f} per batch).")
    dev = df.dropna(subset=["Deviation_Count"])
    if dev["Quality_Status"].eq("Fail").any() and dev["Quality_Status"].eq("Pass").any():
        f_avg = dev.loc[dev["Quality_Status"] == "Fail", "Deviation_Count"].mean()
        p_avg = dev.loc[dev["Quality_Status"] == "Pass", "Deviation_Count"].mean()
        out[-1]["text"] += f" Failed batches average {f_avg:.1f} deviations versus {p_avg:.1f} for passing batches."

    n_out = int(df["Outlier_Flag"].sum())
    if n_out:
        rate = df.groupby("Plant")["Outlier_Flag"].mean().mul(100)
        size = df["Plant"].value_counts()
        rate = rate[(size.reindex(rate.index) >= MIN_GROUP) & (rate.index != "Unknown")]
        where = f"; {rate.idxmax()} has the highest share ({_pct(rate.max())} of its batches)" if len(rate) > 1 else ""
        add("medium", "Anomalies",
            f"{n_out:,} batches ({_pct(_rate(n_out, len(df)))}) contain statistically extreme readings that were kept "
            f"for review{where}.")

    n_cqa = int(df["Has_Missing_CQA"].sum())
    if n_cqa:
        add("info", "Data coverage",
            f"{n_cqa:,} batches ({_pct(_rate(n_cqa, len(df)))}) lack at least one critical test result. Averages and "
            f"compliance figures use only the results that exist; nothing was estimated.")
    return out
