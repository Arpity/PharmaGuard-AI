"""Deterministic observability metrics computed with pandas from the trace store."""
from __future__ import annotations

import json
from collections import Counter

import pandas as pd

STATUS_ORDER = ["success", "success_flagged", "degraded", "refused", "blocked", "invalid_request", "error"]
STATUS_LABELS = {"success": "Success", "success_flagged": "Success (flagged)", "degraded": "Degraded (fallback)", "refused": "Refused (disposition)",
                 "blocked": "Blocked (guardrail)", "invalid_request": "Invalid request", "error": "Error"}


def _isum(s: pd.Series) -> int:
    return int(pd.to_numeric(s, errors="coerce").fillna(0).sum())


def kpis(df: pd.DataFrame) -> dict:
    """Success rate = requests handled without a system error / total. Guardrail blocks, refusals and invalid
    requests are correct handling, not failures; system errors are the only 'failed' requests."""
    n = len(df)
    if n == 0:
        return {"total_requests": 0, "success_rate": None, "failed_requests": 0, "avg_latency_ms": None, "p50_latency_ms": None,
                "p95_latency_ms": None, "guardrail_blocks": 0, "retrieval_failures": 0, "input_tokens": 0, "output_tokens": 0,
                "total_tokens": 0, "requests_with_usage": 0, "token_coverage": None, "clean_success_rate": None, "status_counts": {}}
    failed = int((df["final_status"] == "error").sum())
    usage = df["total_tokens"].notna()
    lat = df["latency_ms"].astype(float)
    return {
        "total_requests": n, "failed_requests": failed, "success_rate": (n - failed) / n * 100,
        "clean_success_rate": float(df["final_status"].isin(["success", "success_flagged"]).mean() * 100),
        "avg_latency_ms": float(lat.mean()), "p50_latency_ms": float(lat.quantile(.5)), "p95_latency_ms": float(lat.quantile(.95)),
        "guardrail_blocks": int(df["guardrail_blocked"].sum()), "retrieval_failures": int(df["retrieval_failed"].sum()),
        "input_tokens": _isum(df["input_tokens"]), "output_tokens": _isum(df["output_tokens"]),
        "total_tokens": _isum(df["total_tokens"]), "requests_with_usage": int(usage.sum()),
        "token_coverage": float(usage.mean() * 100),
        "status_counts": {s: int((df["final_status"] == s).sum()) for s in STATUS_ORDER if (df["final_status"] == s).any()},
    }


def requests_over_time(df: pd.DataFrame, freq: str = "H") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["bucket", "final_status", "requests"])
    t = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601").dt.tz_localize(None).dt.floor(freq)
    return df.assign(bucket=t).groupby(["bucket", "final_status"]).size().rename("requests").reset_index()


def parse_json_col(series: pd.Series) -> pd.Series:
    return series.map(lambda v: json.loads(v) if isinstance(v, str) and v else [])


def guardrail_event_counts(df: pd.DataFrame) -> pd.DataFrame:
    c = Counter(e for ev in parse_json_col(df["guardrail_events"]) for e in ev) if len(df) else Counter()
    return pd.DataFrame(sorted(c.items(), key=lambda kv: -kv[1]), columns=["event", "count"])


def stage_latency(spans: pd.DataFrame) -> pd.DataFrame:
    """Mean / p95 duration and error count per pipeline stage (root span excluded)."""
    s = spans[spans["name"] != "ai.request"]
    if s.empty:
        return pd.DataFrame(columns=["stage", "calls", "mean_ms", "p95_ms", "errors"])
    g = s.groupby("name")
    return pd.DataFrame({"calls": g.size(), "mean_ms": g["duration_ms"].mean(), "p95_ms": g["duration_ms"].quantile(.95),
                         "errors": g["status_code"].apply(lambda x: int((x == "ERROR").sum()))}).reset_index().rename(columns={"name": "stage"}) \
             .sort_values("mean_ms", ascending=False).reset_index(drop=True)


def tool_summary(spans: pd.DataFrame) -> pd.DataFrame:
    """Tool-call success by tool (spans flagged pharmaguard.tool)."""
    if spans.empty:
        return pd.DataFrame(columns=["tool", "calls", "errors", "success_rate", "mean_ms"])
    is_tool = spans["attributes"].map(lambda a: bool(json.loads(a or "{}").get("pharmaguard.tool")))
    t = spans[is_tool]
    if t.empty:
        return pd.DataFrame(columns=["tool", "calls", "errors", "success_rate", "mean_ms"])
    g = t.groupby("name")
    out = pd.DataFrame({"calls": g.size(), "errors": g["status_code"].apply(lambda x: int((x == "ERROR").sum())), "mean_ms": g["duration_ms"].mean()})
    out["success_rate"] = (1 - out["errors"] / out["calls"]) * 100
    return out.reset_index().rename(columns={"name": "tool"})[["tool", "calls", "errors", "success_rate", "mean_ms"]]


def model_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["model", "requests", "avg_latency_ms", "total_tokens"])
    g = df.groupby(["model", "model_version"])
    return g.agg(requests=("run_id", "count"), avg_latency_ms=("latency_ms", "mean"), total_tokens=("total_tokens", _isum)).reset_index()
