"""Lightweight health probes (availability signals). No network calls are made - an LLM provider is never pinged."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from config import PROJECT_ROOT
from src.assistant.knowledge import KnowledgeBase
from src.assistant.llm import LLMConfig


def _probe(name: str, fn) -> dict:
    try:
        detail = fn()
        return {"check": name, "status": "ok", "detail": detail or "ok"}
    except Exception as exc:  # noqa: BLE001
        return {"check": name, "status": "critical", "detail": f"{type(exc).__name__}: {str(exc)[:120]}"}


def _readable(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path.name)
    return f"{path.name} ({path.stat().st_size / 1024:.0f} KB)"


def _db_ok(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as c:
        c.execute("PRAGMA schema_version").fetchone()
    return f"{path.name} reachable"


def health_checks(cfg: dict, review_db: Path, trace_db: Path, root=None) -> list[dict]:
    root = Path(root or PROJECT_ROOT)
    llm = LLMConfig.from_env()

    def kb():
        n = len(KnowledgeBase.from_directory(root / "knowledge").chunks)
        if not n:
            raise RuntimeError("knowledge base is empty")
        return f"{n} chunks indexed"

    checks = [
        _probe("Raw dataset readable", lambda: _readable(root / cfg["dataset"]["raw_path"])),
        _probe("Cleaned dataset readable", lambda: _readable(root / cfg["processed"]["clean_path"])),
        _probe("Knowledge base loads", kb),
        _probe("Review database reachable", lambda: _db_ok(review_db)),
        _probe("Trace database reachable", lambda: _db_ok(trace_db)),
        _probe("Golden evaluation dataset present", lambda: _readable(root / cfg["evaluation"]["golden_path"])),
    ]
    checks.append({"check": "LLM configuration", "status": "ok",
                   "detail": f"live: {llm.provider}/{llm.model} (not pinged)" if llm.is_live else "demo mode (no LLM dependency)"})
    return checks


def default_db_paths(cfg: dict) -> tuple[Path, Path]:
    review = Path(os.getenv("PHARMAGUARD_DB_PATH") or PROJECT_ROOT / cfg["review"]["db_path"])
    trace = Path(os.getenv("PHARMAGUARD_OBS_DB") or PROJECT_ROOT / "data" / "app" / "observability.db")
    return review, trace
