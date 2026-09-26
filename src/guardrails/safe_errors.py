"""Safe error handling: redact secrets, log details privately, show users only a reference id."""
from __future__ import annotations

import logging
import re
import traceback
import uuid
from pathlib import Path

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)(api[_-]?key|authorization|x-api-key|token|password|secret)(\s*[:=]\s*|\s+)(bearer\s+)?[^\s,;\"']{6,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
    # long opaque tokens; W3C trace / span ids in structured log lines are identifiers, not secrets
    re.compile(r"(?<!trace_id\": \")(?<!span_id\": \")(?<!run_id\": \")\b[A-Za-z0-9_\-]{32,}\b"),
]


def redact(text: str) -> str:
    out = str(text)
    for p in _SECRET_PATTERNS:
        out = p.sub(lambda m: (m.group(1) + "=" if m.lastindex else "") + "[REDACTED]", out)
    return out


def new_error_ref() -> str:
    return "ERR-" + uuid.uuid4().hex[:8].upper()


def log_exception(exc: BaseException, log_path=None) -> str:
    """Write a redacted traceback to the private error log and return a reference id for the user."""
    ref = new_error_ref()
    logger = logging.getLogger("pharmaguard.errors")
    if log_path is not None and not any(getattr(h, "_pg", False) for h in logger.handlers):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        h = logging.FileHandler(log_path, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        h._pg = True
        logger.addHandler(h)
        logger.setLevel(logging.ERROR)
    logger.error("%s %s", ref, redact("".join(traceback.format_exception(type(exc), exc, exc.__traceback__))))
    return ref
