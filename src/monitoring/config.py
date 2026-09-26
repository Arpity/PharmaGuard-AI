"""Monitoring configuration (config/monitoring.yaml) and the status rule engine."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from config import PROJECT_ROOT

STATUS_ORDER = {"ok": 0, "unknown": 0, "warn": 1, "critical": 2}
ICON = {"ok": "🟢", "warn": "🟡", "critical": "🔴", "unknown": "⚪"}
CATEGORIES = ["Application", "AI", "Data", "Security", "Cost", "Business"]


def load_monitoring_config(path=None) -> dict:
    return yaml.safe_load(Path(path or PROJECT_ROOT / "config" / "monitoring.yaml").read_text(encoding="utf-8"))


def rule_status(value: Optional[float], rule: dict) -> str:
    """ok / warn / critical / unknown for a value against {direction, warn, critical}. Thresholds are inclusive breaches
    for 'max' when value > threshold (or >= for zero-tolerance rules), and value < threshold for 'min'."""
    if value is None or value != value:
        return "unknown"
    if rule["direction"] == "max":
        zero_tol = rule["warn"] == 0 and rule["critical"] == 0
        breach = (lambda t: value > t) if not zero_tol else (lambda t: value > 0)
        return "critical" if breach(rule["critical"]) else "warn" if breach(rule["warn"]) else "ok"
    return "critical" if value < rule["critical"] else "warn" if value < rule["warn"] else "ok"


def worst(statuses) -> str:
    return max(statuses, key=lambda s: STATUS_ORDER[s], default="unknown")


@dataclass
class Indicator:
    category: str
    key: str                  # slo key (also used as a Prometheus label)
    name: str
    value: Optional[float]
    unit: str
    status: str
    rule: str                 # human-readable threshold, e.g. "warn > 1.0, critical > 5.0"
    detail: str = ""

    @property
    def display(self) -> str:
        if self.value is None:
            return "n/a"
        v = f"{self.value:,.0f}" if abs(self.value) >= 100 else f"{self.value:.2f}".rstrip("0").rstrip(".")
        return f"{v}{self.unit}"


def make_indicator(cfg: dict, category: str, key: str, name: str, value: Optional[float], unit: str = "", detail: str = "") -> Indicator:
    rule = cfg["slo"][key]
    sym = ">" if rule["direction"] == "max" else "<"
    return Indicator(category, key, name, value, unit, rule_status(value, rule),
                     f"warn {sym} {rule['warn']}, critical {sym} {rule['critical']}", detail)
