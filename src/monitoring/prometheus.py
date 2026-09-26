"""Prometheus text-format exposition of the monitoring snapshot (see docs/observability-integration.md)."""
from __future__ import annotations

import re

from .config import STATUS_ORDER
from .snapshot import Snapshot

_LABEL_ESC = str.maketrans({"\\": "\\\\", '"': '\\"', "\n": "\\n"})


class _Out:
    def __init__(self):
        self.lines, self._declared = [], set()

    def add(self, name: str, help_: str, typ: str, value, labels: dict | None = None) -> None:
        if value is None:
            return
        if name not in self._declared:
            self._declared.add(name)
            self.lines += [f"# HELP {name} {help_}", f"# TYPE {name} {typ}"]
        lab = "{" + ",".join(f'{k}="{str(v).translate(_LABEL_ESC)}"' for k, v in (labels or {}).items()) + "}" if labels else ""
        self.lines.append(f"{name}{lab} {float(value):g}")


def render_prometheus(s: Snapshot) -> str:
    o, m = _Out(), s.metrics
    a, ai, d, sec, c, b = (m[k] for k in ("application", "ai", "data", "security", "cost", "business"))
    for st, n in a["status_counts"].items():
        o.add("pharmaguard_requests_total", "AI requests by final status in the window", "gauge", n, {"status": st})
    o.add("pharmaguard_availability_ratio", "Share of requests handled without an internal error", "gauge", None if a["availability_pct"] is None else a["availability_pct"] / 100)
    o.add("pharmaguard_request_errors", "Requests that ended in an internal error", "gauge", a["errors"])
    for q, k in (("0.5", "latency_p50_ms"), ("0.95", "latency_p95_ms")):
        o.add("pharmaguard_request_latency_ms", "End-to-end request latency in milliseconds", "gauge", a[k], {"quantile": q})
    o.add("pharmaguard_ai_unsupported_claim_ratio", "Share of answers with an unsupported claim", "gauge", None if ai["unsupported_claim_rate_pct"] is None else ai["unsupported_claim_rate_pct"] / 100)
    o.add("pharmaguard_ai_retrieval_failures", "Retrieval failures in the window", "gauge", ai["retrieval_failures"])
    o.add("pharmaguard_ai_guardrail_failures", "Model outputs blocked by output validation", "gauge", ai["guardrail_failures"])
    o.add("pharmaguard_ai_answer_quality_ratio", "Golden-set evaluation overall score", "gauge", None if ai["answer_quality_pct"] is None else ai["answer_quality_pct"] / 100)
    o.add("pharmaguard_data_missing_ratio", "Share of missing cells in the raw dataset", "gauge", d["missing_pct"] / 100)
    o.add("pharmaguard_data_quality_score", "Data quality score (0-100) of the cleaned dataset", "gauge", d["quality_score"])
    for r in d["drift"].itertuples() if len(d["drift"]) else []:
        o.add("pharmaguard_data_drift_psi", "Population Stability Index per feature", "gauge", r.psi, {"feature": r.feature})
    o.add("pharmaguard_data_schema_changes", "Schema differences vs registered baseline", "gauge", d["schema_changes"])
    o.add("pharmaguard_security_unauthorized_requests", "Permission denials in the window", "gauge", sec["unauthorized_requests"])
    o.add("pharmaguard_security_suspicious_inputs", "Prompt-injection / disposition attempts in the window", "gauge", sec["suspicious_inputs"])
    o.add("pharmaguard_security_events", "Guardrail and security events in the window", "gauge", sec["guardrail_security_events"])
    o.add("pharmaguard_llm_calls", "LLM provider calls in the window", "gauge", c["llm_calls"])
    o.add("pharmaguard_llm_tokens", "LLM tokens in the window", "gauge", c["input_tokens"], {"direction": "input"})
    o.add("pharmaguard_llm_tokens", "LLM tokens in the window", "gauge", c["output_tokens"], {"direction": "output"})
    o.add("pharmaguard_llm_estimated_cost_usd", "Estimated LLM cost (configured prices only)", "gauge", c["estimated_cost_usd"])
    o.add("pharmaguard_investigations_assisted", "Batch investigations assisted in the window", "gauge", b["investigations_assisted"])
    for dec, k in (("approved", "approvals"), ("rejected", "rejections"), ("more_analysis", "more_analysis_requests")):
        o.add("pharmaguard_human_reviews", "Human review decisions on AI findings", "gauge", b[k], {"decision": dec})
    o.add("pharmaguard_investigation_support_seconds", "Average AI investigation-support time", "gauge", b["avg_support_seconds"])
    o.add("pharmaguard_pending_reviews", "AI runs awaiting human review", "gauge", b["pending_review_backlog"])
    for i in s.indicators:
        o.add("pharmaguard_indicator_status", "Indicator status: 0 ok, 1 warn, 2 critical", "gauge", STATUS_ORDER[i.status], {"category": i.category.lower(), "indicator": i.key})
    for h in s.health:
        o.add("pharmaguard_health_check_up", "1 if the health probe passed", "gauge", 1 if h["status"] == "ok" else 0, {"check": re.sub(r"\W+", "_", h["check"]).strip("_").lower()})
    o.add("pharmaguard_overall_status", "Overall status: 0 ok, 1 warn, 2 critical", "gauge", STATUS_ORDER[s.overall])
    return "\n".join(o.lines) + "\n"
