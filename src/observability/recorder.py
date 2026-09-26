"""Turns a finished Investigation + its spans into one trace-store row and one structured log line."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from config import PROJECT_ROOT
from src.security.secure_logging import get_security_logger, log_event

from .store import TraceStore
from .tracing import Span, iso

DEMO_MODEL, DEMO_MODEL_VERSION = "demo-writer", "demo-writer v1"
SUCCESS_STATUSES = {"success", "success_flagged", "degraded", "refused"}


def prompt_info() -> dict:
    """Prompt version from governance metadata plus a content hash (the canary is neutralised)."""
    from src.assistant.pipeline import SYSTEM_PROMPT
    from src.guardrails import output_guard
    try:
        ver = yaml.safe_load((PROJECT_ROOT / "config" / "governance.yaml").read_text())["prompt"]["version"]
    except Exception:  # noqa: BLE001
        ver = "unknown"
    sha = hashlib.sha256(SYSTEM_PROMPT.replace(output_guard.CANARY, "<canary>").encode()).hexdigest()[:12]
    return {"version": str(ver), "sha256_12": sha}


def final_status(inv) -> str:
    g = inv.guardrail or {}
    if inv.error_ref:
        return "error"
    if inv.blocked:
        return "blocked"
    if not inv.found:
        return "invalid_request"
    if g.get("input") == "restricted_action_refused":
        return "refused"
    if str(g.get("output", "")).startswith("blocked") or inv.fallback_reason:
        return "degraded"
    return "success_flagged" if g.get("needs_verification") else "success"


def is_guardrail_block(inv) -> bool:
    g = inv.guardrail or {}
    return (inv.blocked or str(g.get("input", "")).startswith(("blocked", "restricted"))
            or str(g.get("output", "")).startswith("blocked"))


def tool_calls_from(spans: list[Span]) -> list[dict]:
    return [{"name": s.name, "status": s.status_code, "duration_ms": s.duration_ms,
             "detail": {k: v for k, v in s.attributes.items() if k.startswith("pharmaguard.tool.")}}
            for s in spans if s.attributes.get("pharmaguard.tool")]


@dataclass
class Recorder:
    store: TraceStore
    log_path: Optional[Path] = None
    service_version: str = ""

    def record(self, inv, tracer, ctx, llm_cfg) -> dict:
        spans = tracer.spans
        root = spans[0]
        pi = prompt_info()
        usage = inv.usage or {}
        errors = list(inv.errors)
        etype = ""
        if inv.error_ref:
            etype = next((e["type"] for e in inv.errors if e.get("fatal")), "InternalError")
        elif errors:
            etype = errors[0]["type"]
        retrieval = next((s for s in spans if s.name == "knowledge_retrieval"), None)
        retrieval_failed = bool(inv.found and (not inv.hits or (retrieval is not None and retrieval.status_code == "ERROR")))
        status = final_status(inv)
        row = {
            "run_id": inv.run_id, "trace_id": tracer.trace_id, "timestamp": iso(root.start_ns),
            "user_name": ctx.user, "user_role": ctx.role, "batch_id": inv.batch_id,
            "model": f"{llm_cfg.provider}/{llm_cfg.model}" if llm_cfg.is_live else DEMO_MODEL,
            "model_version": llm_cfg.model if llm_cfg.is_live else DEMO_MODEL_VERSION,
            "prompt_version": pi["version"], "mode": inv.mode if inv.found else "n/a",
            "retrieved_documents": [{"tag": f"K{i}", "document": h.chunk.doc_id, "section": h.chunk.section, "score": h.score}
                                    for i, h in enumerate(inv.hits, 1)],
            "tool_calls": tool_calls_from(spans), "latency_ms": round(root.duration_ms, 3),
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"), "total_tokens": usage.get("total_tokens"),
            "token_source": "provider" if usage else None, "errors": errors, "error_type": etype, "error_ref": inv.error_ref,
            "guardrail_input": (inv.guardrail or {}).get("input", ""), "guardrail_output": (inv.guardrail or {}).get("output", ""),
            "guardrail_events": [e["type"] for e in (inv.guardrail or {}).get("events", [])],
            "guardrail_blocked": int(is_guardrail_block(inv)), "retrieval_failed": int(retrieval_failed), "final_status": status,
            "question_len": len(inv.question or ""), "question_sha256_8": hashlib.sha256((inv.question or "").encode()).hexdigest()[:8],
            "synthetic": int(ctx.synthetic),
        }
        self.store.record_request(row, spans)
        if self.log_path:
            log_event(get_security_logger(self.log_path, "pharmaguard.observability"), "ai_request", run_id=row["run_id"],
                      trace_id=row["trace_id"], user=ctx.user, role=ctx.role, batch_id=row["batch_id"], model=row["model"],
                      model_version=row["model_version"], prompt_version=row["prompt_version"], mode=row["mode"],
                      latency_ms=row["latency_ms"], status=status, guardrail_input=row["guardrail_input"],
                      guardrail_output=row["guardrail_output"], guardrail_events=row["guardrail_events"],
                      retrieved_docs=len(inv.hits), tool_calls=[t["name"] for t in row["tool_calls"]],
                      total_tokens=row["total_tokens"], error_type=row["error_type"], error_ref=row["error_ref"],
                      retrieval_failed=bool(retrieval_failed), synthetic=ctx.synthetic, question=inv.question or "")
        return row


@dataclass
class RequestContext:
    """Who is asking, where results are persisted, and where telemetry goes."""
    user: str = "anonymous"
    role: str = "viewer"
    store: object = None                    # ReviewStore | None  (persist the answer for QA review)
    recorder: Optional[Recorder] = None     # None = no telemetry
    synthetic: bool = False
    traceparent: Optional[str] = None       # join an upstream W3C trace
    extra: dict = field(default_factory=dict)


def default_paths() -> tuple[Path, Path]:
    db = Path(os.getenv("PHARMAGUARD_OBS_DB") or PROJECT_ROOT / "data" / "app" / "observability.db")
    logs = Path(os.getenv("PHARMAGUARD_LOG_DIR") or PROJECT_ROOT / "logs")
    return db, logs / "observability.jsonl"


def get_recorder() -> Recorder:
    db, log = default_paths()
    return Recorder(TraceStore(db), log)
