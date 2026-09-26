"""Metric collectors: one pure function per monitoring domain. Inputs are DataFrames already limited to the time window."""
from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from src.data_quality import checks
from src.data_quality.rules import is_missing

UNSUPPORTED_EVENTS = {"UNGROUNDED_NUMBER", "INVALID_CITATION", "UNKNOWN_BATCH_REFERENCE", "UNKNOWN_PROCEDURE_REFERENCE",
                      "UNSUPPORTED_CAUSAL_CLAIM", "MISSING_CITATION"}
SUSPICIOUS_EVENTS = {"PROMPT_INJECTION_BLOCKED", "DISPOSITION_REQUEST_REFUSED"}
UNAUTHORIZED_TYPES = {"UNAUTHORIZED_ACCESS", "PERMISSION_DENIED", "FOUR_EYES_VIOLATION"}
ASSISTED_STATUSES = {"success", "success_flagged", "degraded", "refused"}


def _pct(n: float, d: float) -> Optional[float]:
    return float(n) / d * 100 if d else None


def _events(df: pd.DataFrame) -> pd.Series:
    return df["guardrail_events"].map(lambda v: set(json.loads(v)) if isinstance(v, str) and v else set()) if len(df) else pd.Series(dtype=object)


# ---- application ---------------------------------------------------------------------------------------------
def application(req: pd.DataFrame, hours: Optional[float]) -> dict:
    n = len(req)
    errors = int((req["final_status"] == "error").sum()) if n else 0
    lat = req["latency_ms"].astype(float) if n else pd.Series(dtype=float)
    return {"requests": n, "errors": errors, "availability_pct": None if not n else 100 - _pct(errors, n), "error_rate_pct": _pct(errors, n),
            "latency_mean_ms": float(lat.mean()) if n else None, "latency_p50_ms": float(lat.quantile(.5)) if n else None,
            "latency_p95_ms": float(lat.quantile(.95)) if n else None, "latency_max_ms": float(lat.max()) if n else None,
            "requests_per_hour": (n / hours) if (n and hours) else None,
            "status_counts": req["final_status"].value_counts().to_dict() if n else {}}


# ---- AI --------------------------------------------------------------------------------------------------------------
def ai(req: pd.DataFrame, evaluation: dict) -> dict:
    reached = req[(req["mode"] != "n/a") & (req["final_status"] != "error")] if len(req) else req
    ev = _events(reached)
    n = len(reached)
    unsupported = int(ev.map(lambda s: bool(s & UNSUPPORTED_EVENTS)).sum()) if n else 0
    out_blocked = int(reached["guardrail_output"].fillna("").str.startswith("blocked").sum()) if n else 0
    m = evaluation.get("metrics", {}) if evaluation.get("available") else {}
    age = None
    if evaluation.get("available"):
        age = max((pd.Timestamp.now(tz="UTC") - pd.Timestamp(evaluation["generated_at"])).total_seconds() / 86400, 0)
    return {"answers": n, "unsupported_claim_answers": unsupported, "unsupported_claim_rate_pct": _pct(unsupported, n),
            "retrieval_failures": int(reached["retrieval_failed"].sum()) if n else 0,
            "retrieval_failure_rate_pct": _pct(int(reached["retrieval_failed"].sum()), n) if n else None,
            "guardrail_failures": out_blocked, "guardrail_failure_rate_pct": _pct(out_blocked, n),
            "answer_quality_pct": evaluation["overall_score"] * 100 if evaluation.get("available") else None, "evaluation_age_days": age,
            "eval_groundedness": m.get("groundedness"), "eval_answer_relevance": m.get("answer_relevance"),
            "eval_guardrail_success": m.get("guardrail_success"), "eval_unsupported_claim_rate": m.get("unsupported_claim_rate"),
            "eval_retrieval_relevance": m.get("retrieval_relevance"), "evaluation_available": bool(evaluation.get("available"))}


# ---- data ------------------------------------------------------------------------------------------------------------
def data(raw: pd.DataFrame, clean: pd.DataFrame, cleaning_summary: dict, drift: pd.DataFrame, schema_diff: Optional[dict],
         schema_changes: Optional[int]) -> dict:
    cells = max(raw.shape[0] * raw.shape[1], 1)
    miss_raw = int(sum(is_missing(raw[c]).sum() for c in raw.columns))
    per_col = checks.missing_report(raw) if len(raw) else pd.DataFrame(columns=["column", "missing_count", "missing_pct"])
    cq = cleaning_summary.get("score_after", {}) if cleaning_summary else {}
    rq = cleaning_summary.get("score_before", {}) if cleaning_summary else {}
    return {"rows_raw": len(raw), "rows_clean": len(clean), "missing_pct": miss_raw / cells * 100, "missing_by_column": per_col,
            "quality_score_raw": rq.get("overall"), "quality_score": cq.get("overall"), "quality_dimensions": cq.get("dimensions", {}),
            "drift": drift, "drift_psi_max": float(drift["psi"].max()) if len(drift) and drift["psi"].notna().any() else None,
            "drifting_features": int((drift["status"].isin(["warn", "critical"])).sum()) if len(drift) else 0,
            "schema_diff": schema_diff, "schema_changes": schema_changes}


# ---- security --------------------------------------------------------------------------------------------------------
def security(req: pd.DataFrame, sec: pd.DataFrame) -> dict:
    n = len(req)
    ev = _events(req)
    suspicious = int(ev.map(lambda s: bool(s & SUSPICIOUS_EVENTS)).sum()) if n else 0
    unauthorized = int(sec["event_type"].isin(UNAUTHORIZED_TYPES).sum()) if len(sec) else 0
    errors = int((req["final_status"] == "error").sum()) if n else 0
    counts: dict = {}
    for s in ev:
        for e in s:
            counts[e] = counts.get(e, 0) + 1
    for t in (sec["event_type"] if len(sec) else []):
        counts[t] = counts.get(t, 0) + 1
    return {"unauthorized_requests": unauthorized, "failed_requests": errors + unauthorized, "suspicious_inputs": suspicious,
            "suspicious_input_rate_pct": _pct(suspicious, n), "injections_blocked": int(ev.map(lambda s: "PROMPT_INJECTION_BLOCKED" in s).sum()) if n else 0,
            "disposition_refusals": int(ev.map(lambda s: "DISPOSITION_REQUEST_REFUSED" in s).sum()) if n else 0,
            "guardrail_security_events": int(sum(counts.values())), "event_counts": counts}


# ---- cost ------------------------------------------------------------------------------------------------------------
def cost(req: pd.DataFrame, spans: pd.DataFrame, mon_cfg: dict) -> dict:
    cc = mon_cfg["cost"]
    tok_in, tok_out = (int(pd.to_numeric(req[c], errors="coerce").fillna(0).sum()) if len(req) else 0 for c in ("input_tokens", "output_tokens"))
    calls = int((spans["name"] == "llm_generate").sum()) if len(spans) else 0
    failed = int(((spans["name"] == "llm_generate") & (spans["status_code"] == "ERROR")).sum()) if len(spans) else 0
    est, priced_tokens, by_model = 0.0, 0, []
    if len(req):
        used = req[pd.to_numeric(req["total_tokens"], errors="coerce").notna()]
        for (model, ver), g in used.groupby(["model", "model_version"]):
            i, o = int(g["input_tokens"].sum()), int(g["output_tokens"].sum())
            price = cc["pricing_usd_per_1m_tokens"].get(ver) or cc["pricing_usd_per_1m_tokens"].get(model)
            usd = (i * price["input"] + o * price["output"]) / 1e6 if price else None
            if usd is not None:
                est += usd
                priced_tokens += i + o
            by_model.append({"model": model, "model_version": ver, "requests": len(g), "input_tokens": i, "output_tokens": o,
                             "estimated_cost_usd": None if usd is None else round(usd, 4), "pricing": "configured" if price else "not configured"})
    total = tok_in + tok_out
    budget = cc.get("token_budget_per_day")
    return {"input_tokens": tok_in, "output_tokens": tok_out, "total_tokens": total, "llm_calls": calls, "llm_call_failures": failed,
            "avg_tokens_per_call": (total / calls) if calls else None, "estimated_cost_usd": est if priced_tokens else None,
            "priced_token_share_pct": _pct(priced_tokens, total), "by_model": pd.DataFrame(by_model),
            "token_budget_per_day": budget, "token_budget_utilisation_pct": _pct(total, budget) if (budget and total is not None) else None,
            "pricing_note": "Estimates use config/monitoring.yaml prices; models without a configured price are excluded."}


# ---- business --------------------------------------------------------------------------------------------------------
def business(req: pd.DataFrame, runs: pd.DataFrame, reviews: pd.DataFrame, mon_cfg: dict) -> dict:
    assisted = req[(req["final_status"].isin(ASSISTED_STATUSES)) & (req["mode"] != "n/a")] if len(req) else req
    dec = reviews["decision"].value_counts().to_dict() if len(reviews) else {}
    total_dec = sum(dec.values())
    ttr = None
    if len(runs) and len(reviews):
        first = reviews.groupby("run_id")["timestamp"].min()
        created = runs.set_index("run_id")["created_at"]
        j = pd.concat([first, created], axis=1, join="inner").dropna()
        if len(j):
            ttr = float((pd.to_datetime(j["timestamp"], utc=True, format="ISO8601") - pd.to_datetime(j["created_at"], utc=True, format="ISO8601")).dt.total_seconds().mean() / 3600)
    manual = mon_cfg["business"].get("manual_investigation_minutes")
    avg_s = float(assisted["latency_ms"].mean() / 1000) if len(assisted) else None
    return {"investigations_assisted": len(assisted), "unique_batches": int(assisted["batch_id"].nunique()) if len(assisted) else 0,
            "approvals": dec.get("approved", 0), "rejections": dec.get("rejected", 0), "more_analysis_requests": dec.get("more_analysis", 0),
            "human_decisions": total_dec, "approval_rate_pct": _pct(dec.get("approved", 0), total_dec),
            "review_coverage_pct": _pct(int((runs["review_count"] > 0).sum()), len(runs)) if len(runs) else None,
            "pending_review_backlog": int((runs["status"] == "pending_review").sum()) if len(runs) else 0,
            "avg_support_seconds": avg_s, "avg_time_to_first_review_hours": ttr,
            "estimated_minutes_saved": (len(assisted) * (manual - avg_s / 60)) if (manual and avg_s is not None) else None}
