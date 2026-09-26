"""AI evaluation gate: turns an evaluation report into a pass / fail decision using documented, overridable thresholds."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

LEVELS = ("critical", "required", "advisory")


@dataclass
class Check:
    name: str
    actual: Optional[float]
    op: str                     # ">=" | "<="
    threshold: float
    level: str
    passed: bool
    detail: str = ""


@dataclass
class GateResult:
    mode: str
    checks: list[Check] = field(default_factory=list)

    @property
    def blocking_failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and (c.level == "critical" or (c.level == "required" and self.mode == "enforce"))]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c not in self.blocking_failures]

    @property
    def passed(self) -> bool:
        return not self.blocking_failures


def load_gate_config(path, env: Optional[dict] = None) -> dict:
    """Load config/evaluation_gate.yaml and apply EVAL_GATE_* environment overrides."""
    env = os.environ if env is None else env
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    mode = (env.get("EVAL_GATE_MODE") or cfg.get("mode", "enforce")).strip().lower()
    if mode not in ("enforce", "warn"):
        raise ValueError(f"EVAL_GATE_MODE must be 'enforce' or 'warn', got {mode!r}")
    cfg["mode"] = mode
    for name, t in cfg["thresholds"].items():
        key = f"EVAL_GATE_{name.upper()}"
        for bound in ("min", "max"):
            v = (env.get(f"{key}_{bound.upper()}") or "").strip()
            if v:
                t[bound] = float(v)
        lvl = (env.get(f"{key}_LEVEL") or "").strip().lower()
        if lvl:
            if lvl not in LEVELS:
                raise ValueError(f"{key}_LEVEL must be one of {LEVELS}, got {lvl!r}")
            t["level"] = lvl
        if t.get("level") not in LEVELS or not ({"min", "max"} & set(t)):
            raise ValueError(f"threshold {name!r} needs min or max and a valid level")
    return cfg


def evaluate_gate(report: dict, gate: dict, current_golden_md5: Optional[str] = None) -> GateResult:
    res, s = GateResult(mode=gate["mode"]), report["summary"]
    m = s["metrics"]
    for name, t in gate["thresholds"].items():
        actual = m.get(name)
        for bound, op in (("min", ">="), ("max", "<=")):
            if bound not in t:
                continue
            ok = actual is not None and (actual >= t[bound] if op == ">=" else actual <= t[bound])
            res.checks.append(Check(name, actual, op, t[bound], t["level"], ok, "metric not available" if actual is None else ""))

    cases = report["cases"]
    for cat in gate.get("critical_categories", []):
        rs = [c for c in cases if c["category"] == cat]
        bad = [c["case_id"] for c in rs if not c["passed"]]
        res.checks.append(Check(f"critical category: {cat}", float(len(rs) - len(bad)), ">=", float(len(rs)), "critical",
                                bool(rs) and not bad, ("no cases in this category" if not rs else f"failed: {', '.join(bad)}" if bad else "")))
    res.checks.append(Check("golden cases evaluated", float(len(cases)), ">=", float(gate.get("min_cases", 0)), "critical", len(cases) >= gate.get("min_cases", 0)))
    if gate.get("require_fresh_golden", True):
        fresh = current_golden_md5 is None or report.get("golden_dataset_md5") == current_golden_md5
        res.checks.append(Check("evaluation matches current golden dataset", 1.0 if fresh else 0.0, ">=", 1.0, "critical", fresh,
                                "" if fresh else "report was produced from a different golden dataset - re-run scripts/run_evaluation.py"))
    return res


def render_markdown(res: GateResult, report: Optional[dict] = None) -> str:
    icon = {True: "✅", False: "❌"}
    lines = [f"## AI evaluation gate: {'PASSED ✅' if res.passed else 'FAILED ❌'}", "",
             f"Mode: `{res.mode}`" + (f" · {report['summary']['cases_passed']}/{report['summary']['cases']} golden cases passed · {report['mode']}" if report else ""), "",
             "| Check | Result | Required | Level | |", "|---|---|---|---|---|"]
    for c in res.checks:
        actual = "n/a" if c.actual is None else (f"{c.actual:.3f}" if c.actual < 100 else f"{c.actual:.0f}")
        thr = f"{c.threshold:.3f}" if c.threshold < 100 else f"{c.threshold:.0f}"
        mark = icon[True] if c.passed else ("❌" if c in res.blocking_failures else "⚠️")
        lines.append(f"| {c.name} | {actual} | {c.op} {thr} | {c.level} | {mark} {c.detail} |")
    if res.warnings:
        lines += ["", f"⚠️ {len(res.warnings)} non-blocking warning(s)."]
    if not res.passed:
        lines += ["", "**Blocking failures:** " + ", ".join(c.name for c in res.blocking_failures)]
    return "\n".join(lines)


def load_report(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
