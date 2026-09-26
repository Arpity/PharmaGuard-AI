import copy
import json
import subprocess
import sys

import pytest

from config import PROJECT_ROOT
from src.evaluation import gate as G

GATE = PROJECT_ROOT / "config" / "evaluation_gate.yaml"
REPORT = PROJECT_ROOT / "reports" / "evaluation" / "latest.json"


def good_report():
    metrics = {"analytical_correctness": 1.0, "retrieval_relevance": 0.92, "groundedness": 1.0, "answer_relevance": 1.0,
               "unsupported_claim_rate": 0.0, "guardrail_success": 1.0, "unsafe_output_catch_rate": 1.0, "tool_success": 1.0,
               "latency_p95_ms": 8.0}
    cases = [{"case_id": f"G{i:03d}", "category": c, "passed": True}
             for i, c in enumerate(["prompt_injection"] * 8 + ["disposition_request"] * 8 + ["adversarial_llm_output"] * 11 + ["investigate"] * 40)]
    return {"mode": "demo", "golden_dataset_md5": "abc", "summary": {"metrics": metrics, "cases": len(cases), "cases_passed": len(cases)}, "cases": cases}


@pytest.fixture
def gate():
    return G.load_gate_config(GATE, env={})


def run(report, gate, md5="abc"):
    return G.evaluate_gate(report, gate, md5)


def test_default_gate_passes_a_good_report(gate):
    res = run(good_report(), gate)
    assert res.passed and not res.warnings and res.mode == "enforce"
    assert {"guardrail_success", "groundedness", "analytical_correctness", "latency_p95_ms"} <= {c.name for c in res.checks}


@pytest.mark.parametrize("metric,value,blocks", [
    ("guardrail_success", 0.98, True), ("unsafe_output_catch_rate", 0.9, True), ("groundedness", 0.99, True),
    ("unsupported_claim_rate", 0.02, True), ("analytical_correctness", 0.90, True), ("tool_success", 0.9, True),
    ("retrieval_relevance", 0.5, True), ("answer_relevance", 0.7, True),
])
def test_each_threshold_blocks_when_violated(gate, metric, value, blocks):
    r = good_report(); r["summary"]["metrics"][metric] = value
    res = run(r, gate)
    assert res.passed is (not blocks) and metric in [c.name for c in res.blocking_failures]


def test_advisory_threshold_only_warns(gate):
    r = good_report(); r["summary"]["metrics"]["latency_p95_ms"] = 99999
    res = run(r, gate)
    assert res.passed and [c.name for c in res.warnings] == ["latency_p95_ms"]


def test_missing_metric_fails_closed(gate):
    r = good_report(); del r["summary"]["metrics"]["groundedness"]
    res = run(r, gate)
    assert not res.passed and "metric not available" in next(c for c in res.checks if c.name == "groundedness").detail


def test_critical_category_case_failure_blocks_even_if_scores_look_fine(gate):
    r = good_report()
    next(c for c in r["cases"] if c["category"] == "prompt_injection")["passed"] = False
    res = run(r, gate)
    assert not res.passed and any(c.name == "critical category: prompt_injection" for c in res.blocking_failures)
    r2 = good_report(); r2["cases"] = [c for c in r2["cases"] if c["category"] != "disposition_request"]
    assert not run(r2, gate).passed                                           # a critical category with no cases also fails


def test_stale_golden_and_too_few_cases_block(gate):
    assert not run(good_report(), gate, md5="different").passed
    r = good_report(); r["cases"] = r["cases"][:10]
    assert not run(r, gate).passed


def test_warn_mode_relaxes_required_checks_but_never_critical_ones():
    warn = G.load_gate_config(GATE, env={"EVAL_GATE_MODE": "warn"})
    r = good_report(); r["summary"]["metrics"]["answer_relevance"] = 0.1
    res = run(r, warn)
    assert res.passed and "answer_relevance" in [c.name for c in res.warnings]
    r["summary"]["metrics"]["guardrail_success"] = 0.5
    assert not run(r, warn).passed                                            # critical stays blocking


def test_environment_overrides_thresholds_and_levels():
    g = G.load_gate_config(GATE, env={"EVAL_GATE_ANSWER_RELEVANCE_MIN": "0.5", "EVAL_GATE_LATENCY_P95_MS_LEVEL": "required",
                                      "EVAL_GATE_LATENCY_P95_MS_MAX": "5"})
    assert g["thresholds"]["answer_relevance"]["min"] == 0.5 and g["thresholds"]["latency_p95_ms"]["level"] == "required"
    r = good_report(); r["summary"]["metrics"]["answer_relevance"] = 0.6
    assert run(r, g).passed is False                                          # latency 8ms > 5ms and now required
    r["summary"]["metrics"]["latency_p95_ms"] = 4
    assert run(r, g).passed


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        G.load_gate_config(GATE, env={"EVAL_GATE_MODE": "yolo"})
    with pytest.raises(ValueError):
        G.load_gate_config(GATE, env={"EVAL_GATE_GROUNDEDNESS_LEVEL": "optional"})
    with pytest.raises(ValueError):
        G.load_gate_config(GATE, env={"EVAL_GATE_GROUNDEDNESS_MIN": "high"})


def test_gate_config_file_is_documented_and_consistent_with_dashboard_targets(gate):
    text = GATE.read_text()
    for word in ("critical", "required", "advisory", "EVAL_GATE_MODE", "EVAL_GATE_<METRIC>_MIN"):
        assert word in text
    cfg = json.loads(json.dumps(gate))
    assert all(t["level"] in G.LEVELS for t in cfg["thresholds"].values())
    assert cfg["thresholds"]["guardrail_success"]["level"] == "critical" and cfg["thresholds"]["groundedness"]["level"] == "critical"
    from config import load_config
    t = load_config()["evaluation"]["targets"]                                # gate must not be laxer than the published targets
    assert cfg["thresholds"]["analytical_correctness"]["min"] >= t["analytical_correctness"]
    assert cfg["thresholds"]["retrieval_relevance"]["min"] >= t["retrieval_relevance"]


def test_markdown_summary_marks_blocking_and_warning_rows(gate):
    r = good_report(); r["summary"]["metrics"]["latency_p95_ms"] = 99999; r["summary"]["metrics"]["groundedness"] = 0.5
    md = G.render_markdown(run(r, gate), r)
    assert "FAILED" in md and "Blocking failures" in md and "groundedness" in md and "⚠️" in md and "❌" in md
    assert "PASSED" in G.render_markdown(run(good_report(), gate), good_report())


@pytest.mark.skipif(not REPORT.exists(), reason="evaluation report not generated")
def test_gate_script_exit_codes(tmp_path):
    script = str(PROJECT_ROOT / "scripts" / "evaluation_gate.py")
    ok = subprocess.run([sys.executable, script, "--report", str(REPORT)], capture_output=True, text=True, cwd=PROJECT_ROOT)
    assert ok.returncode == 0 and "PASSED" in ok.stdout
    bad = json.loads(REPORT.read_text())
    bad["summary"]["metrics"]["guardrail_success"] = 0.9
    p = tmp_path / "bad.json"; p.write_text(json.dumps(bad))
    fail = subprocess.run([sys.executable, script, "--report", str(p)], capture_output=True, text=True, cwd=PROJECT_ROOT)
    assert fail.returncode == 1 and "EVALUATION GATE FAILED" in fail.stderr
    summary = tmp_path / "summary.md"
    subprocess.run([sys.executable, script, "--report", str(REPORT)], capture_output=True, text=True, cwd=PROJECT_ROOT,
                   env={"GITHUB_STEP_SUMMARY": str(summary), "PATH": "/usr/bin:/bin"})
    assert "AI evaluation gate" in summary.read_text()
    assert subprocess.run([sys.executable, script, "--report", str(tmp_path / "nope.json")], capture_output=True, cwd=PROJECT_ROOT).returncode == 2
    assert subprocess.run([sys.executable, script, "--report", str(REPORT)], capture_output=True, cwd=PROJECT_ROOT,
                          env={"EVAL_GATE_MODE": "bogus", "PATH": "/usr/bin:/bin"}).returncode == 2
