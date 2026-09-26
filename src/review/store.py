"""SQLite persistence for AI runs, human reviews and guardrail events (demo-grade, append-only).

Tables ai_runs / reviews / guardrail_events reject UPDATE and DELETE via triggers so the trail
cannot be silently edited. A run's status is *derived* from its latest review.
A review is a decision about the AI FINDING only - there is deliberately no batch-disposition field."""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.guardrails import roles

from .sqlite_util import connect
from src.guardrails.input_guard import normalise

DECISIONS = {"approved": "Approve AI Finding", "rejected": "Reject AI Finding", "more_analysis": "Request More Analysis"}
STATUS_LABELS = {"pending_review": "Pending review", "approved": "AI finding approved", "rejected": "AI finding rejected",
                 "more_analysis": "More analysis requested"}
COMMENT_REQUIRED = {"rejected", "more_analysis"}
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .,'_\-]{1,59}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS ai_runs (
  run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, user_name TEXT NOT NULL, user_role TEXT NOT NULL,
  batch_id TEXT NOT NULL, question TEXT NOT NULL, intents TEXT, mode TEXT, model_info TEXT, answer TEXT NOT NULL,
  analysis_json TEXT, sources_json TEXT, guardrail_json TEXT);
CREATE TABLE IF NOT EXISTS reviews (
  review_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES ai_runs(run_id),
  reviewer TEXT NOT NULL, reviewer_role TEXT NOT NULL, timestamp TEXT NOT NULL,
  decision TEXT NOT NULL CHECK (decision IN ('approved','rejected','more_analysis')), comments TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS guardrail_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, timestamp TEXT NOT NULL, user_name TEXT,
  event_type TEXT NOT NULL, severity TEXT NOT NULL, detail TEXT);
"""
TRIGGERS = "".join(
    f"CREATE TRIGGER IF NOT EXISTS {t}_no_{op.lower()} BEFORE {op} ON {t} "
    f"BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;"
    for t in ("ai_runs", "reviews", "guardrail_events") for op in ("UPDATE", "DELETE"))


class ReviewError(ValueError):
    """Validation problem that is safe to show to the user."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    return f"RUN-{datetime.now(timezone.utc):%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"


class ReviewStore:
    def __init__(self, path, allow_self_review: bool = False):
        self.path = Path(path)
        self.allow_self_review = allow_self_review
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA + TRIGGERS)

    def _conn(self) -> sqlite3.Connection:
        return connect(self.path)

    # ---- runs ------------------------------------------------------------------------------
    def save_run(self, inv, user: str, role: str) -> str:
        roles.require(role, "ask_assistant")
        run_id = getattr(inv, "run_id", "") or new_run_id()          # traced requests already carry their Run ID
        with self._conn() as c:
            c.execute("INSERT INTO ai_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                run_id, _now(), user, role, inv.batch_id, inv.question, ",".join(inv.intents), inv.mode, inv.model_info,
                inv.answer, json.dumps(inv.analysis, default=str),
                json.dumps([{"tag": f"K{i}", "document": h.chunk.doc_title, "section": h.chunk.section,
                             "file": Path(h.chunk.path).name, "score": h.score, "text": h.chunk.text}
                            for i, h in enumerate(inv.hits, 1)]),
                json.dumps(inv.guardrail, default=str)))
            for e in inv.guardrail.get("events", []):
                c.execute("INSERT INTO guardrail_events (run_id,timestamp,user_name,event_type,severity,detail) VALUES (?,?,?,?,?,?)",
                          (run_id, _now(), user, e["type"], e["severity"], e["detail"]))
        return run_id

    def log_event(self, user: str, event_type: str, severity: str, detail: str, run_id: Optional[str] = None) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO guardrail_events (run_id,timestamp,user_name,event_type,severity,detail) VALUES (?,?,?,?,?,?)",
                      (run_id, _now(), user, event_type, severity, detail[:500]))

    _STATUS_SQL = """
      SELECT r.*, COALESCE((SELECT decision FROM reviews v WHERE v.run_id=r.run_id ORDER BY review_id DESC LIMIT 1),
                           'pending_review') AS status,
             (SELECT COUNT(*) FROM reviews v WHERE v.run_id=r.run_id) AS review_count
      FROM ai_runs r"""

    def list_runs(self, status: Optional[str] = None, batch_id: Optional[str] = None, limit: int = 500) -> pd.DataFrame:
        with self._conn() as c:
            df = pd.read_sql_query(self._STATUS_SQL + " ORDER BY r.created_at DESC, r.run_id DESC LIMIT ?", c, params=(limit,))
        if status:
            df = df[df["status"] == status]
        if batch_id:
            df = df[df["batch_id"].str.upper() == batch_id.strip().upper()]
        return df.drop(columns=["analysis_json", "sources_json", "guardrail_json"]).reset_index(drop=True)

    def get_run(self, run_id: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute(self._STATUS_SQL + " WHERE r.run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        for k in ("analysis_json", "sources_json", "guardrail_json"):
            d[k.replace("_json", "")] = json.loads(d.pop(k) or "null")
        return d

    # ---- reviews ---------------------------------------------------------------------------
    def add_review(self, run_id: str, reviewer: str, role: str, decision: str, comments: str = "") -> int:
        roles.require(role, "review_ai_finding")
        if decision not in DECISIONS:
            raise ReviewError("Unknown decision.")
        name, comments = normalise(reviewer), normalise(comments)
        if not _NAME_RE.match(name):
            raise ReviewError("Enter a valid reviewer name (2-60 characters).")
        if len(comments) > 2000:
            raise ReviewError("Comments are limited to 2000 characters.")
        if decision in COMMENT_REQUIRED and len(comments) < 5:
            raise ReviewError("Comments are required (at least 5 characters) to reject a finding or request more analysis.")
        run = self.get_run(run_id)
        if run is None:
            raise ReviewError("Run not found.")
        if not self.allow_self_review and name.lower() == run["user_name"].strip().lower():
            raise ReviewError("Four-eyes rule: you cannot review an analysis you ran yourself.")
        with self._conn() as c:
            cur = c.execute("INSERT INTO reviews (run_id,reviewer,reviewer_role,timestamp,decision,comments) VALUES (?,?,?,?,?,?)",
                            (run_id, name, role, _now(), decision, comments))
            return int(cur.lastrowid)

    def reviews(self, run_id: Optional[str] = None) -> pd.DataFrame:
        q, p = "SELECT * FROM reviews", ()
        if run_id:
            q, p = q + " WHERE run_id=?", (run_id,)
        with self._conn() as c:
            return pd.read_sql_query(q + " ORDER BY review_id DESC", c, params=p)

    def events(self, limit: int = 500) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM guardrail_events ORDER BY event_id DESC LIMIT ?", c, params=(limit,))
