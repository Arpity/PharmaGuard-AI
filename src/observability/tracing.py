"""Minimal, dependency-free tracing that mirrors the OpenTelemetry data model, so the recorded traces can be
exported to any OTel backend later without changing the instrumentation:

  * 128-bit hex trace ids, 64-bit hex span ids, parent/child spans, span kinds, UNSET/OK/ERROR status, events
  * attribute names follow OTel semantic conventions where they exist (gen_ai.*, enduser.*, exception.*)
  * to_otlp_json() renders spans in the OTLP/JSON wire format (collector-ready)
  * W3C `traceparent` helpers allow joining an upstream trace

Swapping in the real SDK later means replacing `Tracer` with `opentelemetry.trace.get_tracer(...)`; the
`start_span` / `set_attribute` / `record_exception` calls used by the pipeline are the same shape."""
from __future__ import annotations

import re
import secrets
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Optional

from src.guardrails.safe_errors import redact

SERVICE_NAME, SCOPE_NAME, SCOPE_VERSION = "pharmaguard-ai", "pharmaguard.observability", "1.0"
KINDS = {"INTERNAL": 1, "SERVER": 2, "CLIENT": 3, "PRODUCER": 4, "CONSUMER": 5}
STATUS = {"UNSET": 0, "OK": 1, "ERROR": 2}
_TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")


def new_trace_id() -> str:
    return secrets.token_hex(16)


def new_span_id() -> str:
    return secrets.token_hex(8)


def parse_traceparent(header: Optional[str]) -> Optional[tuple[str, str]]:
    """W3C traceparent -> (trace_id, parent_span_id), or None if absent / invalid / all-zero."""
    m = _TRACEPARENT.match((header or "").strip().lower())
    if not m or set(m.group(1)) == {"0"} or set(m.group(2)) == {"0"}:
        return None
    return m.group(1), m.group(2)


def format_traceparent(trace_id: str, span_id: str) -> str:
    return f"00-{trace_id}-{span_id}-01"


def _clean(v: Any) -> Any:
    """OTel attribute values are primitives or homogeneous lists of primitives."""
    if isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return redact(str(v))[:500]


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    kind: str = "INTERNAL"
    start_ns: int = 0
    end_ns: int = 0
    attributes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    status_code: str = "UNSET"
    status_message: str = ""

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = _clean(value)

    def set_attributes(self, attrs: dict) -> None:
        for k, v in attrs.items():
            self.set_attribute(k, v)

    def add_event(self, name: str, attributes: Optional[dict] = None) -> None:
        self.events.append({"name": name, "time_ns": time.time_ns(), "attributes": {k: _clean(v) for k, v in (attributes or {}).items()}})

    def set_status(self, code: str, message: str = "") -> None:
        self.status_code, self.status_message = code, redact(message)[:300]

    def record_exception(self, exc: BaseException) -> None:
        self.add_event("exception", {"exception.type": type(exc).__name__, "exception.message": str(exc)})
        self.set_status("ERROR", f"{type(exc).__name__}: {exc}")

    @property
    def duration_ms(self) -> float:
        return round((self.end_ns - self.start_ns) / 1e6, 3) if self.end_ns else 0.0


class Tracer:
    """One instance per request. Exporters receive the finished spans via export(spans)."""

    def __init__(self, service_name: str = SERVICE_NAME, exporters: tuple = (), traceparent: Optional[str] = None,
                 clock: Callable[[], int] = time.time_ns):
        parent = parse_traceparent(traceparent)
        self.service_name, self.exporters = service_name, list(exporters)
        # One monotonic timeline anchored to wall-clock at request start: children can never appear to
        # start before / end after their parent, and durations are immune to system clock adjustments.
        self._base_wall, self._base_perf = clock(), time.perf_counter_ns()
        self.trace_id = parent[0] if parent else new_trace_id()
        self._remote_parent = parent[1] if parent else None
        self.spans: list[Span] = []
        self._stack: list[Span] = []

    def _now(self) -> int:
        return self._base_wall + (time.perf_counter_ns() - self._base_perf)

    @contextmanager
    def start_span(self, name: str, kind: str = "INTERNAL", attributes: Optional[dict] = None) -> Iterator[Span]:
        parent = self._stack[-1].span_id if self._stack else self._remote_parent
        span = Span(name, self.trace_id, new_span_id(), parent, kind, self._now())
        span.set_attributes(attributes or {})
        self.spans.append(span)
        self._stack.append(span)
        try:
            yield span
        except BaseException as exc:
            span.record_exception(exc)
            raise
        finally:
            span.end_ns = max(self._now(), span.start_ns + 1)
            if span.status_code == "UNSET":
                span.set_status("OK")
            self._stack.pop()

    @property
    def current(self) -> Optional[Span]:
        return self._stack[-1] if self._stack else None

    @property
    def root(self) -> Optional[Span]:
        return self.spans[0] if self.spans else None

    def traceparent(self) -> str:
        return format_traceparent(self.trace_id, (self.current or self.root).span_id)

    def finish(self) -> list[Span]:
        for exp in self.exporters:
            exp.export(self.spans)
        return self.spans


# ---- OTLP/JSON (collector-ready) ------------------------------------------------------------------
def _any_value(v: Any) -> dict:
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, list):
        return {"arrayValue": {"values": [_any_value(x) for x in v]}}
    return {"stringValue": str(v)}


def _attrs(d: dict) -> list[dict]:
    return [{"key": k, "value": _any_value(v)} for k, v in d.items()]


def to_otlp_json(spans: list[Span], service_name: str = SERVICE_NAME, version: str = "", environment: str = "demo") -> dict:
    """ExportTraceServiceRequest in OTLP/JSON form - POST it to <collector>/v1/traces as-is."""
    return {"resourceSpans": [{
        "resource": {"attributes": _attrs({"service.name": service_name, "service.version": version, "deployment.environment": environment,
                                           "telemetry.sdk.name": "pharmaguard-minimal", "telemetry.sdk.language": "python"})},
        "scopeSpans": [{"scope": {"name": SCOPE_NAME, "version": SCOPE_VERSION}, "spans": [{
            "traceId": s.trace_id, "spanId": s.span_id, **({"parentSpanId": s.parent_span_id} if s.parent_span_id else {}),
            "name": s.name, "kind": KINDS[s.kind], "startTimeUnixNano": str(s.start_ns), "endTimeUnixNano": str(s.end_ns),
            "attributes": _attrs(s.attributes),
            "events": [{"timeUnixNano": str(e["time_ns"]), "name": e["name"], "attributes": _attrs(e["attributes"])} for e in s.events],
            "status": {"code": STATUS[s.status_code], **({"message": s.status_message} if s.status_message else {})}}
            for s in spans]}]}]}


def iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat(timespec="milliseconds")
