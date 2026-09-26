"""Dependency / supply-chain checks: requirement hygiene, installed-version conformance, runtime EOL, pip-audit."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from importlib import metadata
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

PYTHON_EOL = {(3, 8): date(2024, 10, 7), (3, 9): date(2025, 10, 31), (3, 10): date(2026, 10, 31),
              (3, 11): date(2027, 10, 31), (3, 12): date(2028, 10, 31), (3, 13): date(2029, 10, 31)}


def parse_requirements(path) -> list[Requirement]:
    reqs = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            try:
                reqs.append(Requirement(line))
            except InvalidRequirement:
                pass
    return reqs


def check_requirements(path, lockfile=None) -> list[dict]:
    """One row per requirement: declared spec, installed version, and hygiene notes."""
    rows = []
    for r in parse_requirements(path):
        spec = str(r.specifier)
        try:
            inst = metadata.version(r.name)
        except metadata.PackageNotFoundError:
            inst = None
        notes = []
        if not spec:
            notes.append("no version constraint")
        elif "<" not in spec and "==" not in spec:
            notes.append("no upper bound")
        if inst is None:
            notes.append("not installed")
        elif spec and not r.specifier.contains(inst, prereleases=True):
            notes.append(f"installed {inst} violates {spec}")
        rows.append({"package": r.name, "declared": spec or "(any)", "installed": inst or "-",
                     "ok": inst is not None and not any("violates" in n or n == "no version constraint" for n in notes),
                     "notes": "; ".join(notes)})
    return rows


def python_support(today: date | None = None, version=None) -> dict:
    v, today = version or sys.version_info, today or date.today()
    eol = PYTHON_EOL.get((v[0], v[1]))
    status = "unknown" if eol is None else "end-of-life" if eol < today else "supported"
    return {"version": f"{v[0]}.{v[1]}.{v[2] if len(v) > 2 else 0}", "eol_date": str(eol) if eol else "unknown", "status": status}


def run_pip_audit(lockfile, timeout: int = 180) -> dict:
    """Run pip-audit if installed (needs network to reach the vulnerability database)."""
    try:
        proc = subprocess.run([sys.executable, "-m", "pip_audit", "-r", str(lockfile), "--format", "json", "--progress-spinner", "off"],
                              capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return {"status": "unavailable", "detail": "pip-audit not installed (pip install pip-audit)"}
    except subprocess.TimeoutExpired:
        return {"status": "error", "detail": "pip-audit timed out"}
    if "No module named pip_audit" in proc.stderr:
        return {"status": "unavailable", "detail": "pip-audit not installed (pip install pip-audit)"}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"status": "error", "detail": (proc.stderr or "no output")[-300:]}
    vulns = [{"package": d["name"], "version": d["version"], "id": v["id"], "fix": ", ".join(v.get("fix_versions", [])) or "none"}
             for d in data.get("dependencies", []) for v in d.get("vulns", [])]
    return {"status": "vulnerabilities" if vulns else "clean", "packages_audited": len(data.get("dependencies", [])),
            "vulnerabilities": vulns, "detail": ""}
