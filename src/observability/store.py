"""Local SQLite trace store (demo). One row per AI request plus its spans. The span table keeps OTel fields, so
export to a real backend is a straight re-serialisation (see tracing.to_otlp_json)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .tracing import Span, to_otlp_json

SCHEMA = """
CREATE TABLE IF NOT EXISTS ai_requests (
  run_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL, timestamp TEXT NOT NULL, user_name TEXT, user_role TEXT, batch_id TEXT,
  model TEXT, model_version TEXT, prompt_version TEXT, mode TEXT, retrieved_documents TEXT, tool_calls TEXT,
  latency_ms REAL, input_tokens INTEGER, output_tokens INTEGER, total_tokens INTEGER, token_source TEXT,
  errors TEXT, error_type TEXT, error_ref TEXT, guardrail_input TEXT, guardrail_output TEXT, guardrail_events TEXT,
  guardrail_blocked INTEGER, retrieval_failed INTEGER, final_status TEXT NOT NULL, question_len INTEGER, question_sha256_8 TEXT,
  synthetic INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS spans (
  trace_id TEXT NOT NULL, span_id TEXT NOT NULL, run_id TEXT NOT NULL, parent_span_id TEXT, name TEXT NOT NULL, kind TEXT,
  start_ns INTEGER, end_ns INTEGER, duration_ms REAL, status_code TEXT, status_message TEXT, attributes TEXT, events TEXT,
  PRIMARY KEY (trace_id, span_id));
CREATE TABLE IF NOT EXISTS security_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, event_type TEXT NOT NULL, severity TEXT NOT NULL,
  user_name TEXT, user_role TEXT, source TEXT, detail TEXT, synthetic INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS idx_req_ts ON ai_requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_span_run ON spans(run_id);
"""
REQUEST_COLUMNS = {
    "run_id", "trace_id", "timestamp", "user_name", "user_role", "batch_id", "model", "model_version", "prompt_version", "mode",
    "retrieved_documents", "tool_calls", "latency_ms", "input_tokens", "output_tokens", "total_tokens", "token_source", "errors",
    "error_type", "error_ref", "guardrail_input", "guardrail_output", "guardrail_events", "guardrail_blocked", "retrieval_failed",
    "final_status", "question_len", "question_sha256_8", "synthetic"}
JSON_COLS = ["retrieved_documents", "tool_calls", "errors", "guardrail_events"]


class TraceStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def record_request(self, row: dict, spans: list[Span]) -> None:
        cols = [k for k in row]
        if not set(cols) <= REQUEST_COLUMNS:                       # identifiers below are interpolated: allow-list them
            raise ValueError(f"unknown ai_requests columns: {sorted(set(cols) - REQUEST_COLUMNS)}")
        vals = [json.dumps(row[k], default=str) if k in JSON_COLS else row[k] for k in cols]
        with self._conn() as c:
            c.execute(f"INSERT INTO ai_requests ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)  # nosec B608 - allow-listed columns, bound values
            c.executemany("INSERT INTO spans VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [
                (s.trace_id, s.span_id, row["run_id"], s.parent_span_id, s.name, s.kind, s.start_ns, s.end_ns, s.duration_ms,
                 s.status_code, s.status_message, json.dumps(s.attributes, default=str), json.dumps(s.events, default=str)) for s in spans])

    def requests(self, since: Optional[str] = None, include_synthetic: bool = True, limit: int = 10000) -> pd.DataFrame:
        q, p = "SELECT * FROM ai_requests", []
        conds = []
        if since:
            conds.append("timestamp >= ?"); p.append(since)
        if not include_synthetic:
            conds.append("synthetic = 0")
        if conds:
            q += " WHERE " + " AND ".join(conds)
        with self._conn() as c:
            df = pd.read_sql_query(q + " ORDER BY timestamp DESC LIMIT ?", c, params=p + [limit])
        for col in ("guardrail_blocked", "retrieval_failed", "synthetic"):
            df[col] = df[col].astype(bool)
        return df

    def spans(self, run_id: Optional[str] = None) -> pd.DataFrame:
        q, p = "SELECT * FROM spans", ()
        if run_id:
            q, p = q + " WHERE run_id=?", (run_id,)
        with self._conn() as c:
            return pd.read_sql_query(q + " ORDER BY start_ns", c, params=p)

    def get_request(self, run_id: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM ai_requests WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        for k in JSON_COLS:
            d[k] = json.loads(d[k]) if d[k] else []
        return d

    def otlp(self, run_id: Optional[str] = None, version: str = "") -> dict:
        """OTLP/JSON for one request (or every stored span)."""
        sp = []
        for r in self.spans(run_id).itertuples():
            sp.append(Span(r.name, r.trace_id, r.span_id, r.parent_span_id, r.kind, int(r.start_ns), int(r.end_ns),
                           json.loads(r.attributes or "{}"),
                           [{"name": e["name"], "time_ns": e["time_ns"], "attributes": e["attributes"]} for e in json.loads(r.events or "[]")],
                           r.status_code, r.status_message or ""))
        return to_otlp_json(sp, version=version)

    # ---- security events (access denials etc. that never reach the AI pipeline) --------------------------
    def record_security_event(self, event_type: str, severity: str, user: str = "", role: str = "", source: str = "",
                              detail: str = "", timestamp: Optional[str] = None, synthetic: bool = False) -> None:
        from src.guardrails.safe_errors import redact
        ts = timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute("INSERT INTO security_events (timestamp,event_type,severity,user_name,user_role,source,detail,synthetic) VALUES (?,?,?,?,?,?,?,?)",
                      (ts, event_type, severity, user, role, source, redact(detail)[:300], int(synthetic)))

    def security_events(self, since: Optional[str] = None, limit: int = 5000) -> pd.DataFrame:
        q, p = "SELECT * FROM security_events", []
        if since:
            q += " WHERE timestamp >= ?"; p.append(since)
        with self._conn() as c:
            df = pd.read_sql_query(q + " ORDER BY timestamp DESC LIMIT ?", c, params=p + [limit])
        df["synthetic"] = df["synthetic"].astype(bool)
        return df

    def purge_older_than(self, days: int) -> int:
        """Retention control for the demo store."""
        cut = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        with self._conn() as c:
            ids = [r[0] for r in c.execute("SELECT run_id FROM ai_requests WHERE timestamp < ?", (cut,))]
            c.executemany("DELETE FROM spans WHERE run_id=?", [(i,) for i in ids])
            c.execute("DELETE FROM ai_requests WHERE timestamp < ?", (cut,))
            c.execute("DELETE FROM security_events WHERE timestamp < ?", (cut,))
        return len(ids)
