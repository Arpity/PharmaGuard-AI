"""Secure logging: secrets are redacted, and only identifiers / verdicts / hashes are logged - never the full
question, answer or batch analysis."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.guardrails.safe_errors import redact

SAFE_ID_FIELDS = {"trace_id", "span_id", "run_id"}      # correlation identifiers: never secrets, must stay intact
FORBIDDEN_FIELDS = {"question", "answer", "analysis", "prompt", "api_key", "password", "token", "secret", "authorization"}


class RedactingFilter(logging.Filter):
    """Last line of defence: scrubs credential-like strings from any log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


def get_security_logger(path, name: str = "pharmaguard.security") -> logging.Logger:
    logger = logging.getLogger(name)
    if not any(getattr(h, "_pg_path", None) == str(path) for h in logger.handlers):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        h = logging.FileHandler(path, encoding="utf-8")
        h._pg_path = str(path)
        h.addFilter(RedactingFilter())
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, **fields) -> dict:
    """Write one JSON line. Free-text fields listed in FORBIDDEN_FIELDS are replaced by length + short hash."""
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event}
    for k, v in fields.items():
        if k.lower() in FORBIDDEN_FIELDS:
            text = str(v)
            rec[f"{k}_len"], rec[f"{k}_sha256_8"] = len(text), hashlib.sha256(text.encode()).hexdigest()[:8]
        else:
            rec[k] = v if k in SAFE_ID_FIELDS else (redact(str(v)) if isinstance(v, str) else v)
    logger.info(json.dumps(rec, default=str))
    return rec
