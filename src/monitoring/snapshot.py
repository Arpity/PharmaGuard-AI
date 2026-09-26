"""Assembles a point-in-time monitoring snapshot: metrics by domain, status indicators, health probes and recent events."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from config import PROJECT_ROOT
from src.analytics import kpis as analytics
from src.data_quality import checks
from src.governance import metadata as gm
from src.observability.store import TraceStore
from src.review.store import ReviewStore

from . import collectors as C
from . import drift as D
from .config import ICON, Indicator, load_monitoring_config, make_indicator, worst
from .health import default_db_paths, health_checks

WINDOWS = {"1h": timedelta(hours=1), "24h": timedelta(hours=24), "7d": timedelta(days=7), "all": None}


@dataclass
class Snapshot:
    generated_at: str
    window: str
    metrics: dict
    indicators: list[Indicator]
    health: list[dict]
    events: pd.DataFrame
    overall: str = "unknown"
    notes: list[str] = field(default_factory=list)
    requests: pd.DataFrame = field(default_factory=pd.DataFrame)          # raw material for charts
    security_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    spans: pd.DataFrame = field(default_factory=pd.DataFrame)
    runs: pd.DataFrame = field(default_factory=pd.DataFrame)
    reviews: pd.DataFrame = field(default_factory=pd.DataFrame)

    def by_category(self, cat: str) -> list[Indicator]:
        return [i for i in self.indicators if i.category == cat]

    def category_status(self, cat: str) -> str:
        return worst(i.status for i in self.by_category(cat))


def _hours(window: str, req: pd.DataFrame) -> Optional[float]:
    if WINDOWS[window]:
        return WINDOWS[window].total_seconds() / 3600
    if len(req):
        t = pd.to_datetime(req["timestamp"], utc=True, format="ISO8601")
        return max((t.max() - t.min()).total_seconds() / 3600, 1 / 60)
    return None


def build_events(req: pd.DataFrame, sec: pd.DataFrame, indicators: list[Indicator], limit: int = 200) -> pd.DataFrame:
    """Recent incidents / events: critical & degraded requests, guardrail and security events, plus currently active alerts."""
    rows = []
    for r in req.itertuples():
        ev = set(json.loads(r.guardrail_events)) if isinstance(r.guardrail_events, str) and r.guardrail_events else set()
        base = {"timestamp": r.timestamp, "run_id": r.run_id, "user": r.user_name}
        if r.final_status == "error":
            rows.append({**base, "severity": "critical", "category": "Application", "event": "Internal error", "detail": f"{r.error_type} · ref {r.error_ref}"})
        if str(r.guardrail_output).startswith("blocked"):
            rows.append({**base, "severity": "warn", "category": "AI", "event": "AI output failed validation (replaced)", "detail": ", ".join(sorted(ev - {"LLM_CALL_FAILED"}))})
        if "LLM_CALL_FAILED" in ev:
            rows.append({**base, "severity": "warn", "category": "AI", "event": "LLM provider call failed (fallback used)", "detail": r.error_type})
        if r.retrieval_failed:
            rows.append({**base, "severity": "warn", "category": "AI", "event": "Knowledge retrieval failed", "detail": ", ".join(sorted(ev & {"RETRIEVAL_FAILED", "RETRIEVAL_EMPTY"})) or r.error_type})
        if "PROMPT_INJECTION_BLOCKED" in ev:
            rows.append({**base, "severity": "warn", "category": "Security", "event": "Prompt injection blocked", "detail": f"batch {r.batch_id}"})
        if "DISPOSITION_REQUEST_REFUSED" in ev:
            rows.append({**base, "severity": "info", "category": "Security", "event": "Disposition request refused", "detail": f"batch {r.batch_id}"})
    for r in sec.itertuples():
        rows.append({"timestamp": r.timestamp, "run_id": "", "user": r.user_name, "severity": "warn" if r.severity != "high" else "critical",
                     "category": "Security", "event": r.event_type.replace("_", " ").capitalize(), "detail": f"{r.source}: {r.detail}"})
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for i in indicators:
        if i.status in ("warn", "critical"):
            rows.append({"timestamp": now, "run_id": "", "user": "", "severity": i.status, "category": i.category,
                         "event": f"ACTIVE ALERT: {i.name}", "detail": f"{i.display} ({i.rule})"})
    df = pd.DataFrame(rows, columns=["timestamp", "severity", "category", "event", "detail", "run_id", "user"])
    if df.empty:
        return df
    df["_t"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    return df.sort_values("_t", ascending=False).drop(columns="_t").head(limit).reset_index(drop=True)


def build_snapshot(cfg: dict, window: str = "24h", include_synthetic: bool = True, root=None, mon_cfg: Optional[dict] = None,
                   trace_store: Optional[TraceStore] = None, review_store: Optional[ReviewStore] = None,
                   raw: Optional[pd.DataFrame] = None, clean: Optional[pd.DataFrame] = None) -> Snapshot:
    """Compute everything from local stores and data files. `raw` / `clean` can be supplied (e.g. a simulated-drift frame)."""
    root, mon = Path(root or PROJECT_ROOT), mon_cfg or load_monitoring_config()
    review_db, trace_db = default_db_paths(cfg)
    ts = trace_store or TraceStore(trace_db)
    rs = review_store or ReviewStore(review_db)
    since = (datetime.now(timezone.utc) - WINDOWS[window]).isoformat(timespec="seconds") if WINDOWS[window] else None

    req = ts.requests(since=since, include_synthetic=include_synthetic)
    sec = ts.security_events(since=since)
    if not include_synthetic and len(sec):
        sec = sec[~sec["synthetic"]]
    spans_all = ts.spans()
    spans = spans_all[spans_all["run_id"].isin(req["run_id"])] if len(req) else spans_all.iloc[0:0]
    runs, reviews = rs.list_runs(), rs.reviews()
    if since and len(reviews):
        reviews = reviews[pd.to_datetime(reviews["timestamp"], utc=True, format="ISO8601") >= pd.Timestamp(since)]
    evaluation = gm.evaluation_status(cfg, root)

    raw = raw if raw is not None else (checks.load_raw(root / cfg["dataset"]["raw_path"]) if (root / cfg["dataset"]["raw_path"]).exists() else pd.DataFrame())
    clean = clean if clean is not None else (analytics.load_clean(root / cfg["processed"]["clean_path"]) if (root / cfg["processed"]["clean_path"]).exists() else pd.DataFrame())
    summ_path = root / cfg["processed"]["summary_path"]
    summary = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    drift = D.drift_report(clean, mon) if len(clean) else pd.DataFrame(columns=["feature", "kind", "psi", "status"])
    base = D.load_schema_baseline(root)
    diff = D.schema_diff(D.infer_schema(raw), base["columns"]) if (base and len(raw)) else None

    m = {"application": C.application(req, _hours(window, req)), "ai": C.ai(req, evaluation),
         "data": C.data(raw, clean, summary, drift, diff, D.schema_change_count(diff) if diff else None),
         "security": C.security(req, sec), "cost": C.cost(req, spans, mon), "business": C.business(req, runs, reviews, mon),
         "missingness_drift": D.missingness_drift(clean, mon) if len(clean) else pd.DataFrame()}
    a, i, d, s, c, b = (m[k] for k in ("application", "ai", "data", "security", "cost", "business"))
    mk = lambda cat, key, name, v, unit="", det="": make_indicator(mon, cat, key, name, v, unit, det)
    inds = [
        mk("Application", "availability_pct", "Availability", a["availability_pct"], "%", "requests handled without an internal error"),
        mk("Application", "error_rate_pct", "Error rate", a["error_rate_pct"], "%"),
        mk("Application", "latency_p95_ms", "Latency p95", a["latency_p95_ms"], " ms"),
        mk("AI", "answer_quality_pct", "Answer quality (golden set)", i["answer_quality_pct"], "%", "latest evaluation overall score"),
        mk("AI", "evaluation_age_days", "Evaluation age", i["evaluation_age_days"], " d"),
        mk("AI", "unsupported_claim_rate_pct", "Unsupported claims (live)", i["unsupported_claim_rate_pct"], "%"),
        mk("AI", "retrieval_failure_rate_pct", "Retrieval failures", i["retrieval_failure_rate_pct"], "%"),
        mk("AI", "guardrail_failure_rate_pct", "Guardrail failures (output blocked)", i["guardrail_failure_rate_pct"], "%"),
        mk("Data", "missing_pct", "Missingness (raw)", d["missing_pct"], "%"),
        mk("Data", "data_quality_score", "Data quality score", d["quality_score"], "", "cleaned data"),
        mk("Data", "drift_psi_max", "Drift (max PSI)", d["drift_psi_max"], "", f"{d['drifting_features']} feature(s) drifting"),
        mk("Data", "schema_changes", "Schema changes", d["schema_changes"], "", "vs registered baseline" if base else "no baseline registered"),
        mk("Security", "unauthorized_requests", "Unauthorized requests", s["unauthorized_requests"], ""),
        mk("Security", "suspicious_input_rate_pct", "Suspicious inputs", s["suspicious_input_rate_pct"], "%", f"{s['suspicious_inputs']} in window"),
        mk("Cost", "token_budget_utilisation_pct", "Token budget used", c["token_budget_utilisation_pct"], "%", "of daily budget" if c["token_budget_per_day"] else "no budget set"),
        mk("Business", "pending_review_backlog", "Pending QA reviews", b["pending_review_backlog"], ""),
    ]
    health = health_checks(cfg, review_db, trace_db, root)
    events = build_events(req, sec, inds)
    snap = Snapshot(datetime.now(timezone.utc).isoformat(timespec="seconds"), window, m, inds, health, events,
                    requests=req, security_events=sec, spans=spans, runs=runs, reviews=reviews)
    snap.overall = worst([i.status for i in inds] + [h["status"] for h in health])
    if not len(req):
        snap.notes.append("No AI requests recorded in this window - request-based indicators show n/a.")
    return snap


__all__ = ["Snapshot", "build_snapshot", "build_events", "ICON", "WINDOWS"]
