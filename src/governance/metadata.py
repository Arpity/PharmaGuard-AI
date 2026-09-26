"""Governance metadata (from config/governance.yaml) plus live status computed from the running project."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

from config import PROJECT_ROOT
from src.assistant.llm import LLMConfig
from src.assistant.pipeline import SYSTEM_PROMPT
from src.guardrails import output_guard

REQUIRED = ["system", "data_sources", "data_classification", "model", "prompt", "risks", "limitations", "guardrails",
            "human_oversight", "approval", "change_history"]
SYSTEM_REQUIRED = ["name", "version", "use_case", "business_owner", "technical_owner", "intended_users"]


def load_governance(path=None) -> dict:
    p = Path(path or PROJECT_ROOT / "config" / "governance.yaml")
    meta = yaml.safe_load(p.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in meta] + [f"system.{k}" for k in SYSTEM_REQUIRED if k not in meta.get("system", {})]
    if missing:
        raise ValueError(f"governance metadata missing: {', '.join(missing)}")
    return meta


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prompt_fingerprint() -> str:
    """Hash of the system prompt with the per-process canary neutralised."""
    return _sha(SYSTEM_PROMPT.replace(output_guard.CANARY, "<canary>").encode())


def fingerprints(root=None) -> dict:
    root = Path(root or PROJECT_ROOT)
    kb = b"".join(p.read_bytes() for p in sorted((root / "knowledge").glob("*.md")))
    golden = root / "data" / "evaluation" / "golden_cases.json"
    return {"system_prompt": prompt_fingerprint(), "knowledge_base": _sha(kb),
            "config": _sha((root / "config" / "config.yaml").read_bytes()),
            "golden_dataset": _sha(golden.read_bytes()) if golden.exists() else None}


def baseline_path(root=None) -> Path:
    return Path(root or PROJECT_ROOT) / "config" / "governance_baseline.json"


def register_baseline(version: str, root=None) -> dict:
    b = {"registered_version": version, "prompt_version": load_governance()["prompt"]["version"], **fingerprints(root)}
    baseline_path(root).write_text(json.dumps(b, indent=1))
    return b


def fingerprint_status(root=None) -> list[dict]:
    cur, p = fingerprints(root), baseline_path(root)
    base = json.loads(p.read_text()) if p.exists() else {}
    labels = {"system_prompt": "System prompt", "knowledge_base": "Knowledge documents", "config": "Project configuration",
              "golden_dataset": "Golden evaluation dataset"}
    rows = []
    for k, label in labels.items():
        b = base.get(k)
        rows.append({"artifact": label, "current": (cur[k] or "missing")[:12], "baseline": (b or "-")[:12],
                     "status": "no baseline" if not b else "matches baseline" if b == cur[k] else "CHANGED since baseline"})
    return rows


def model_status(env=None) -> dict:
    env = os.environ if env is None else env
    cfg = LLMConfig.from_env(dict(env))
    if not cfg.is_live:
        return {"mode": "Demo (no LLM)", "model": "None - deterministic template writer", "model_version": "demo-writer v1",
                "pinned": "n/a", "data_leaves_environment": "No"}
    pinned = bool((env.get("LLM_MODEL") or "").strip())
    return {"mode": "Live LLM", "model": f"{cfg.provider} / {cfg.model}", "model_version": cfg.model,
            "pinned": "Yes (explicit LLM_MODEL)" if pinned else "No - using provider default alias; set LLM_MODEL",
            "data_leaves_environment": f"Yes - to {cfg.provider} ({cfg.base_url})"}


def evaluation_status(cfg: dict, root=None) -> dict:
    root = Path(root or PROJECT_ROOT)
    res, gold = Path(os.getenv("PHARMAGUARD_EVAL_PATH") or root / cfg["evaluation"]["results_path"]), root / cfg["evaluation"]["golden_path"]
    if not res.exists():
        return {"available": False, "status": "Not evaluated"}
    rep = json.loads(res.read_text())
    s = rep["summary"]
    stale = False
    if gold.exists():
        stale = rep.get("golden_dataset_md5") != json.loads(gold.read_text()).get("dataset_md5")
    return {"available": True, "generated_at": rep["generated_at"], "mode": rep["mode"], "cases": s["cases"],
            "cases_passed": s["cases_passed"], "overall_score": s["overall_score"], "targets_met": s["all_targets_met"],
            "metrics": s["metrics"], "target_status": s["target_status"], "stale": stale,
            "status": ("Passed all targets" if s["all_targets_met"] else "Targets missed") + (" (results older than current golden set)" if stale else "")}
