"""Scan project files for hard-coded credentials and check secret hygiene (.env handling)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {".venv", ".git", "__pycache__", "data", "logs", "reports", "node_modules", ".pytest_cache", "tests", ".streamlit_cache"}
TEXT_EXT = {".py", ".yaml", ".yml", ".md", ".toml", ".txt", ".json", ".ini", ".cfg", ".sh", ".env", ".example", ".lock"}
PLACEHOLDER = re.compile(r"(?i)(changeme|your[_-]|example|placeholder|xxx|<.*>|\$\{.*\}|dummy|redacted|\.\.\.)")
SECRET_NAME = re.compile(r"KEY|SECRET|PASSWORD|PASSWD|(?<!MAX_)TOKEN(?!S)")   # LLM_MAX_TOKENS is a limit, not a secret
ALLOW_MARK = "pragma: allowlist secret"
RULES = [
    ("provider-style API key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("private key block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("bearer token literal", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{24,}")),
    ("credential assigned to a string literal",
     re.compile(r"(?i)\b(api[_-]?key|secret|token|passwd|password)\b\s*[:=]\s*['\"][^'\"\s]{8,}['\"]")),
]


@dataclass
class Finding:
    file: str
    line: int
    rule: str
    snippet: str


def _redact_snippet(line: str) -> str:
    return re.sub(r"([A-Za-z0-9_\-]{4})[A-Za-z0-9_\-]{6,}", r"\1****", line.strip())[:120]


def scan_text(text: str, filename: str = "<text>") -> list[Finding]:
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if ALLOW_MARK in line or "os.environ" in line or "getenv" in line:
            continue
        for name, pat in RULES:
            m = pat.search(line)
            if m and not PLACEHOLDER.search(m.group(0)) and not PLACEHOLDER.search(line[m.end():m.end() + 30] if name.startswith("cred") else ""):
                out.append(Finding(filename, i, name, _redact_snippet(line)))
                break
    return out


def scan_project(root, extra_skip: set[str] | None = None) -> tuple[list[Finding], int]:
    """Returns (findings, files_scanned). tests/ is skipped: it contains deliberately fake keys for redaction tests."""
    root, findings, n = Path(root), [], 0
    skip = SKIP_DIRS | (extra_skip or set())
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if not p.is_file() or set(rel.parts[:-1]) & skip or p.stat().st_size > 1_000_000:
            continue
        if p.suffix.lower() not in TEXT_EXT and not p.name.startswith(".env"):
            continue
        if p.name == ".env":                       # real local secrets file: never read its contents here
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        n += 1
        findings += scan_text(text, str(rel))
    return findings, n


def gitignore_covers(root, entry: str = ".env") -> bool:
    gi = Path(root) / ".gitignore"
    return gi.exists() and any(l.strip() in (entry, f"/{entry}") for l in gi.read_text().splitlines())


def env_example_is_clean(root) -> tuple[bool, list[str]]:
    """.env.example must not contain values for secret variables."""
    p, bad = Path(root) / ".env.example", []
    if not p.exists():
        return False, [".env.example missing"]
    for line in p.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            if SECRET_NAME.search(k.upper()) and v.split("#")[0].strip():
                bad.append(k.strip())
    return not bad, bad


def env_variable_status(root, environ) -> list[dict]:
    """Which documented environment variables are set. Values are NEVER returned - only set / not set."""
    p, rows = Path(root) / ".env.example", []
    if not p.exists():
        return rows
    comment = ""
    for line in p.read_text().splitlines():
        s = line.strip()
        if s.startswith("#"):
            comment = s.lstrip("# ").strip()
        elif "=" in s:
            k = s.split("=", 1)[0].strip()
            rows.append({"variable": k, "secret": bool(SECRET_NAME.search(k.upper())),
                         "status": "set" if environ.get(k) else "not set", "purpose": comment})
            comment = ""
    return rows


def config_secret_fields(obj, path: str = "") -> list[str]:
    """Names of config keys that look like credentials AND hold a non-empty value."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{path}.{k}" if path else str(k)
            if SECRET_NAME.search(str(k).upper()) and isinstance(v, str) and v.strip():
                found.append(here)
            found += config_secret_fields(v, here)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += config_secret_fields(v, f"{path}[{i}]")
    return found
