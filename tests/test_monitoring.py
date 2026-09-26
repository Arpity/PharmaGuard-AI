import json
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest
import yaml

from config import PROJECT_ROOT
from src.monitoring import collectors as C
from src.monitoring import drift as D
from src.monitoring.config import Indicator, load_monitoring_config, make_indicator, rule_status, worst
from src.monitoring.health import health_checks
from src.monitoring.prometheus import render_prometheus
from src.monitoring.simulate import seed_demo_reviews, simulate_production
from src.monitoring.snapshot import build_events, build_snapshot
from src.observability.recorder import Recorder
from src.observability.store import TraceStore
from src.review.store import ReviewStore
from tests._paths import page
from tests.test_assistant import hist, kb, make_history  # noqa: F401

MON = load_monitoring_config()


# ---- status rules -------------------------------------------------------------------------------------------------
def test_rule_status_min_max_and_unknown():
    mx, mn = {"direction": "max", "warn": 1.0, "critical": 5.0}, {"direction": "min", "warn": 99.5, "critical": 99.0}
    assert [rule_status(v, mx) for v in (0.5, 1.0, 1.1, 5.0, 5.1)] == ["ok", "ok", "warn", "warn", "critical"]
    assert [rule_status(v, mn) for v in (100, 99.5, 99.4, 99.0, 98.9)] == ["ok", "ok", "warn", "warn", "critical"]
    assert rule_status(None, mx) == "unknown" and rule_status(float("nan"), mn) == "unknown"
    zero = {"direction": "max", "warn": 0, "critical": 0}
    assert rule_status(0, zero) == "ok" and rule_status(1, zero) == "critical"          # zero tolerance (schema changes)


def test_indicator_and_worst():
    i = make_indicator(MON, "Application", "error_rate_pct", "Error rate", 2.5, "%")
    assert i.status == "warn" and "critical > 5.0" in i.rule and i.display == "2.5%"
    assert make_indicator(MON, "Application", "error_rate_pct", "x", None).display == "n/a"
    assert worst(["ok", "warn", "critical"]) == "critical" and worst(["ok", "unknown"]) in ("ok", "unknown") and worst([]) == "unknown"


def test_every_slo_has_a_valid_rule():
    for k, r in MON["slo"].items():
        assert r["direction"] in ("min", "max") and {"warn", "critical"} <= set(r), k
        assert (r["critical"] >= r["warn"]) if r["direction"] == "max" else (r["critical"] <= r["warn"]), k


# ---- drift & schema -----------------------------------------------------------------------------------------------
def test_psi_numeric_behaviour():
    rng = np.random.default_rng(0)
    a, b = pd.Series(rng.normal(0, 1, 2000)), pd.Series(rng.normal(0, 1, 2000))
    assert D.psi_numeric(a, b) < 0.05 and D.psi_numeric(a, a) == pytest.approx(0, abs=1e-9)
    assert D.psi_numeric(a, b + 1.5) > 0.25 and 0.1 < D.psi_numeric(a, b + 0.35) < 0.25
    assert D.psi_numeric(pd.Series([1.0]), b) is None and D.psi_numeric(pd.Series([2.0] * 50), pd.Series([2.0] * 50)) == 0.0
    assert D.psi_numeric(pd.Series([2.0] * 50), pd.Series([9.0] * 50)) > 0.25


def test_psi_categorical():
    ref, same = pd.Series(["a"] * 50 + ["b"] * 50), pd.Series(["a"] * 48 + ["b"] * 52)
    assert D.psi_categorical(ref, same) < 0.01 and D.psi_categorical(ref, pd.Series(["a"] * 95 + ["b"] * 5)) > 0.25
    assert D.psi_categorical(ref, pd.Series(["c"] * 100)) > 0.25 and D.psi_categorical(pd.Series(["a"]), ref) is None


def _clean_like(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"Manufacturing_Date": pd.date_range("2024-01-01", periods=n, freq="D"), "Temperature": rng.normal(22.5, 1.5, n),
                         "Humidity": rng.normal(45, 5, n), "Assay": rng.normal(100, 1.5, n), "pH": rng.normal(6.5, .3, n),
                         "Plant": rng.choice(["Plant_A", "Plant_B"], n), "Quality_Status": rng.choice(["Pass", "Fail"], n, p=[.95, .05]),
                         "Dissolution": rng.normal(92, 3, n)})


def test_drift_report_stable_then_simulated_drift():
    df = _clean_like()
    stable = D.drift_report(df, MON)
    assert set(stable["status"]) == {"ok"} and stable["psi"].max() < 0.1
    drifted = D.drift_report(D.simulate_drift(df, MON), MON)
    assert drifted["psi"].max() > 0.25 and {"Temperature", "Humidity", "Assay"} <= set(drifted[drifted["status"] == "critical"]["feature"])
    assert D.drift_report(df.head(120), MON).empty                                           # too few rows -> no verdict


def test_missingness_drift_detects_new_missing_values():
    df = _clean_like()
    out = D.missingness_drift(D.simulate_drift(df, MON), MON).set_index("column")
    assert out.loc["Dissolution", "change_pp"] > 10 and D.missingness_drift(df, MON).set_index("column")["change_pp"].abs().max() < 1


def test_schema_inference_and_diff():
    raw = pd.DataFrame({"Batch_ID": ["B1", "B2"], "Date": ["2024-01-01", "05/03/2024"], "Temp": ["22.4C", "23"], "Empty": ["", "N/A"]})
    base = D.infer_schema(raw)
    assert base == {"Batch_ID": "text", "Date": "date", "Temp": "number", "Empty": "empty"}
    assert D.schema_change_count(D.schema_diff(base, base)) == 0
    changed = raw.drop(columns=["Temp"]).rename(columns={"Date": "When"})
    changed["New"] = ["x", "y"]
    diff = D.schema_diff(D.infer_schema(changed), base)
    assert diff["added"] == ["When", "New"] and diff["removed"] == ["Date", "Temp"] and diff["type_changes"] == [] and diff["reordered"] is False
    assert D.schema_change_count(diff) == 4
    retyped = D.schema_diff({**base, "Temp": "text"}, base)
    assert retyped["type_changes"] == [{"column": "Temp", "was": "number", "now": "text"}]
    assert D.schema_diff({k: base[k] for k in reversed(list(base))}, base)["reordered"] is True


def test_schema_baseline_roundtrip_and_simulated_change(tmp_path):
    (tmp_path / "config").mkdir()
    raw = pd.DataFrame({"a": ["1", "2"], "Cycle_Time_Hrs": ["1", "2"], "Humidity": ["3", "4"]})
    assert D.load_schema_baseline(tmp_path) is None
    D.register_schema_baseline(raw, tmp_path)
    base = D.load_schema_baseline(tmp_path)
    assert base["columns"]["Humidity"] == "number" and D.schema_change_count(D.schema_diff(D.infer_schema(raw), base["columns"])) == 0
    diff = D.schema_diff(D.infer_schema(D.simulate_schema_change(raw)), base["columns"])
    assert "Cycle_Time_Hrs" in diff["removed"] and "Relative_Humidity" in diff["added"] and "Lot_Supplier" in diff["added"]


def test_real_data_matches_registered_schema_and_is_not_drifting():
    from src.data_quality import checks
    from config import load_config
    cfg = load_config()
    raw = checks.load_raw(PROJECT_ROOT / cfg["dataset"]["raw_path"])
    base = D.load_schema_baseline(PROJECT_ROOT)
    assert base is not None and D.schema_change_count(D.schema_diff(D.infer_schema(raw), base["columns"])) == 0


# ---- collectors ---------------------------------------------------------------------------------------------------
def req_frame(rows):
    base = {"run_id": "R", "timestamp": "2026-01-01T00:00:00+00:00", "user_name": "u", "batch_id": "B00001", "model": "demo-writer", "model_version": "demo-writer v1",
            "mode": "demo", "latency_ms": 10.0, "input_tokens": None, "output_tokens": None, "total_tokens": None, "error_type": "", "error_ref": "",
            "guardrail_input": "passed", "guardrail_output": "passed", "guardrail_events": "[]", "guardrail_blocked": False, "retrieval_failed": False,
            "final_status": "success"}
    df = pd.DataFrame([{**base, "run_id": f"R{i}", **r} for i, r in enumerate(rows)])
    for c in ("input_tokens", "output_tokens", "total_tokens"):
        df[c] = pd.to_numeric(df[c])
    return df


def test_application_metrics():
    df = req_frame([{"latency_ms": 10}, {"latency_ms": 20}, {"latency_ms": 30, "final_status": "error"}, {"latency_ms": 100, "final_status": "blocked", "mode": "n/a"}])
    a = C.application(df, hours=2)
    assert a["requests"] == 4 and a["errors"] == 1 and a["availability_pct"] == 75.0 and a["error_rate_pct"] == 25.0
    assert a["requests_per_hour"] == 2 and a["latency_max_ms"] == 100 and a["status_counts"]["success"] == 2
    assert C.application(df.iloc[0:0], None)["availability_pct"] is None


def test_ai_metrics_from_live_events():
    ev = lambda *e: json.dumps(list(e))
    df = req_frame([{}, {"guardrail_events": ev("UNGROUNDED_NUMBER"), "guardrail_output": "blocked_replaced_with_deterministic_summary"},
                    {"guardrail_events": ev("MISSING_CITATION"), "guardrail_output": "passed_with_warnings"}, {"retrieval_failed": True},
                    {"final_status": "blocked", "mode": "n/a"}, {"final_status": "error"}])
    a = C.ai(df, {"available": True, "overall_score": 0.98, "generated_at": "2026-01-01T00:00:00+00:00",
                  "metrics": {"groundedness": 1.0, "answer_relevance": 0.95, "guardrail_success": 1.0, "unsupported_claim_rate": 0.0, "retrieval_relevance": .9}})
    assert a["answers"] == 4 and a["unsupported_claim_answers"] == 2 and a["unsupported_claim_rate_pct"] == 50.0
    assert a["guardrail_failures"] == 1 and a["retrieval_failures"] == 1 and a["retrieval_failure_rate_pct"] == 25.0
    assert a["answer_quality_pct"] == pytest.approx(98.0) and a["evaluation_age_days"] > 100
    assert C.ai(df.iloc[0:0], {"available": False})["answer_quality_pct"] is None


def test_security_metrics():
    ev = lambda *e: json.dumps(list(e))
    df = req_frame([{"guardrail_events": ev("PROMPT_INJECTION_BLOCKED")}, {"guardrail_events": ev("DISPOSITION_REQUEST_REFUSED")},
                    {"guardrail_events": ev("INPUT_REJECTED")}, {}, {"final_status": "error"}])
    sec = pd.DataFrame([{"event_type": "PERMISSION_DENIED"}, {"event_type": "UNAUTHORIZED_ACCESS"}, {"event_type": "OTHER"}])
    s = C.security(df, sec)
    assert s["suspicious_inputs"] == 2 and s["suspicious_input_rate_pct"] == 40.0 and s["injections_blocked"] == 1
    assert s["unauthorized_requests"] == 2 and s["failed_requests"] == 3 and s["event_counts"]["PERMISSION_DENIED"] == 1


def test_cost_metrics_pricing_and_budget():
    df = req_frame([{"model": "anthropic/demo-model-1", "model_version": "demo-model-1", "input_tokens": 1_000_000, "output_tokens": 100_000, "total_tokens": 1_100_000},
                    {"model": "openai/other", "model_version": "other-1", "input_tokens": 500, "output_tokens": 500, "total_tokens": 1000}, {}])
    spans = pd.DataFrame([{"name": "llm_generate", "status_code": "OK"}, {"name": "llm_generate", "status_code": "ERROR"}, {"name": "input_validation", "status_code": "OK"}])
    c = C.cost(df, spans, MON)
    assert c["llm_calls"] == 2 and c["llm_call_failures"] == 1 and c["total_tokens"] == 1_101_000
    assert c["estimated_cost_usd"] == pytest.approx(1.0 * 3.0 + 0.1 * 15.0)               # only the priced model is estimated
    bm = c["by_model"].set_index("model_version")
    assert bm.loc["other-1", "pricing"] == "not configured" and pd.isna(bm.loc["other-1", "estimated_cost_usd"])
    assert c["priced_token_share_pct"] == pytest.approx(1_100_000 / 1_101_000 * 100)
    assert c["token_budget_utilisation_pct"] == pytest.approx(1_101_000 / MON["cost"]["token_budget_per_day"] * 100)
    none = C.cost(req_frame([{}]), spans.iloc[0:0], MON)
    assert none["estimated_cost_usd"] is None and none["total_tokens"] == 0


def test_business_metrics():
    df = req_frame([{"latency_ms": 2000, "batch_id": "B1"}, {"latency_ms": 4000, "batch_id": "B1"}, {"latency_ms": 10, "final_status": "blocked", "mode": "n/a"},
                    {"final_status": "refused", "batch_id": "B2", "latency_ms": 3000}])
    runs = pd.DataFrame([{"run_id": "r1", "created_at": "2026-01-01T00:00:00+00:00", "status": "approved", "review_count": 1},
                         {"run_id": "r2", "created_at": "2026-01-01T00:00:00+00:00", "status": "pending_review", "review_count": 0}])
    reviews = pd.DataFrame([{"run_id": "r1", "decision": "approved", "timestamp": "2026-01-01T02:00:00+00:00"},
                            {"run_id": "r1", "decision": "rejected", "timestamp": "2026-01-01T05:00:00+00:00"}])
    b = C.business(df, runs, reviews, {"business": {"manual_investigation_minutes": 30}})
    assert b["investigations_assisted"] == 3 and b["unique_batches"] == 2 and b["avg_support_seconds"] == pytest.approx(3.0)
    assert (b["approvals"], b["rejections"], b["human_decisions"], b["approval_rate_pct"]) == (1, 1, 2, 50.0)
    assert b["review_coverage_pct"] == 50.0 and b["pending_review_backlog"] == 1 and b["avg_time_to_first_review_hours"] == pytest.approx(2.0)
    assert b["estimated_minutes_saved"] == pytest.approx(3 * (30 - 3.0 / 60))
    assert C.business(df.iloc[0:0], runs.iloc[0:0], reviews.iloc[0:0], MON)["estimated_minutes_saved"] is None


# ---- security events store ----------------------------------------------------------------------------------------
def test_security_events_store(tmp_path):
    st = TraceStore(tmp_path / "t.db")
    st.record_security_event("PERMISSION_DENIED", "medium", "bob", "viewer", "QA Review page", "denied; key sk-" + "abcdefghijklmnopqrstuvwx")
    st.record_security_event("UNAUTHORIZED_ACCESS", "medium", "amy", "viewer", "x", "y", timestamp="2020-01-01T00:00:00+00:00", synthetic=True)
    df = st.security_events()
    assert len(df) == 2 and "sk-abcdef" not in " ".join(df["detail"]) and df["synthetic"].tolist().count(True) == 1
    assert len(st.security_events(since="2021-01-01T00:00:00")) == 1
    assert st.purge_older_than(365) == 0 and len(st.security_events()) == 1


# ---- snapshot, events, simulation ---------------------------------------------------------------------------------
@pytest.fixture
def stores(tmp_path):
    return TraceStore(tmp_path / "t.db"), ReviewStore(tmp_path / "r.db")


def snap_for(cfg, stores, window="all", **kw):
    return build_snapshot(cfg, window, True, mon_cfg=MON, trace_store=stores[0], review_store=stores[1], **kw)


def test_empty_snapshot_is_unknown_not_broken(cfg, stores):
    s = snap_for(cfg, stores)
    cats = {i.category for i in s.indicators}
    assert cats == {"Application", "AI", "Data", "Security", "Cost", "Business"}
    st = {i.key: i.status for i in s.indicators}
    assert st["availability_pct"] == "unknown" and st["latency_p95_ms"] == "unknown" and st["unauthorized_requests"] == "ok"
    assert s.notes and "No AI requests" in s.notes[0]
    assert s.events.empty or set(s.events["event"]) <= {a for a in s.events["event"] if a.startswith("ACTIVE ALERT")}


def test_healthy_traffic_is_green_and_incident_traffic_alerts(cfg, hist, kb, stores):
    rec = Recorder(stores[0])
    simulate_production(rec, hist, kb, cfg, n=100, incident=False, seed=3)
    calm = {i.key: i for i in snap_for(cfg, stores).indicators}
    assert calm["availability_pct"].value is not None and calm["latency_p95_ms"].status == "ok"
    inc = TraceStore(stores[0].path.parent / "i.db")
    simulate_production(Recorder(inc), hist, kb, cfg, n=100, incident=True, seed=3)
    s = snap_for(cfg, (inc, stores[1]))
    assert s.overall in ("warn", "critical") and any(i.status in ("warn", "critical") for i in s.indicators)
    types = set(s.events["event"]) if len(s.events) else set()
    assert any(t.startswith("ACTIVE ALERT") for t in types) and {"Prompt injection blocked"} & types
    assert set(s.events["severity"]) <= {"critical", "warn", "info"}
    assert pd.to_datetime(s.events["timestamp"], utc=True, format="ISO8601").is_monotonic_decreasing               # newest first


def test_events_feed_classifies_incidents():
    df = req_frame([{"final_status": "error", "error_type": "KeyError", "error_ref": "ERR-1"},
                    {"guardrail_output": "blocked_x", "guardrail_events": json.dumps(["DISPOSITION_CLAIM"])},
                    {"guardrail_events": json.dumps(["PROMPT_INJECTION_BLOCKED"]), "final_status": "blocked"},
                    {"retrieval_failed": True, "guardrail_events": json.dumps(["RETRIEVAL_FAILED"])}])
    sec = pd.DataFrame([{"timestamp": "2026-01-01T00:00:01+00:00", "user_name": "bob", "severity": "medium", "event_type": "PERMISSION_DENIED", "source": "p", "detail": "d"}])
    ind = [Indicator("Data", "drift_psi_max", "Drift", 0.4, "", "critical", "r")]
    ev = build_events(df, sec, ind)
    assert {"Internal error", "AI output failed validation (replaced)", "Prompt injection blocked", "Knowledge retrieval failed", "Permission denied",
            "ACTIVE ALERT: Drift"} <= set(ev["event"])
    assert ev[ev["event"] == "Internal error"].iloc[0]["severity"] == "critical" and ev.columns.tolist()[:3] == ["timestamp", "severity", "category"]


def test_simulated_drift_and_schema_change_turn_data_indicators_red(cfg, stores):
    from src.analytics import kpis
    from src.data_quality import checks
    clean = kpis.load_clean(PROJECT_ROOT / cfg["processed"]["clean_path"])
    raw = checks.load_raw(PROJECT_ROOT / cfg["dataset"]["raw_path"])
    ok = {i.key: i.status for i in snap_for(cfg, stores, raw=raw, clean=clean).indicators}
    assert ok["drift_psi_max"] == "ok" and ok["schema_changes"] == "ok"
    bad = {i.key: i.status for i in snap_for(cfg, stores, raw=D.simulate_schema_change(raw), clean=D.simulate_drift(clean, MON)).indicators}
    assert bad["drift_psi_max"] == "critical" and bad["schema_changes"] == "critical"


def test_seed_demo_reviews_creates_runs_and_decisions(cfg, hist, kb, stores):
    out = seed_demo_reviews(stores[1], hist, kb, cfg, n=12)
    assert out["runs"] == 12 and out["approved"] + out["rejected"] + out["more_analysis"] > 0
    b = snap_for(cfg, stores).metrics["business"]
    assert b["human_decisions"] == out["approved"] + out["rejected"] + out["more_analysis"] and b["pending_review_backlog"] == 12 - sum(
        len(stores[1].reviews(r)) > 0 for r in stores[1].list_runs()["run_id"])


def test_health_checks(cfg, tmp_path):
    ok = health_checks(cfg, tmp_path / "r.db", tmp_path / "t.db")
    assert all(h["status"] == "ok" for h in ok) and {"Knowledge base loads", "LLM configuration"} <= {h["check"] for h in ok}
    bad = health_checks(cfg, tmp_path / "r.db", tmp_path / "t.db", root=tmp_path)             # empty project root
    assert any(h["status"] == "critical" for h in bad)


# ---- prometheus / grafana / otel integration artefacts -----------------------------------------------------------
@pytest.fixture
def prom(cfg, hist, kb, stores):
    simulate_production(Recorder(stores[0]), hist, kb, cfg, n=60, incident=True, seed=2)
    return render_prometheus(snap_for(cfg, stores))


def metric_names(text):
    return {l.split("{")[0].split(" ")[0] for l in text.splitlines() if l and not l.startswith("#")}


def test_prometheus_exposition_is_well_formed(prom):
    line = re.compile(r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{([a-zA-Z_][a-zA-Z0-9_]*="[^"\\]*(\\.[^"\\]*)*",?)+\})? -?[0-9.e+\-]+$')
    for l in prom.splitlines():
        assert l.startswith("#") or line.match(l), l
    helps = {l.split()[2] for l in prom.splitlines() if l.startswith("# HELP")}
    types = {l.split()[2] for l in prom.splitlines() if l.startswith("# TYPE")}
    assert helps == types == metric_names(prom)
    assert prom.endswith("\n") and "pharmaguard_indicator_status{category=\"application\",indicator=\"availability_pct\"}" in prom


def test_prometheus_covers_every_monitoring_domain(prom):
    names = metric_names(prom)
    for n in ("pharmaguard_availability_ratio", "pharmaguard_requests_total", "pharmaguard_request_latency_ms", "pharmaguard_ai_unsupported_claim_ratio",
              "pharmaguard_ai_retrieval_failures", "pharmaguard_ai_guardrail_failures", "pharmaguard_data_missing_ratio", "pharmaguard_data_quality_score",
              "pharmaguard_data_drift_psi", "pharmaguard_data_schema_changes", "pharmaguard_security_unauthorized_requests",
              "pharmaguard_security_suspicious_inputs", "pharmaguard_llm_tokens", "pharmaguard_llm_calls", "pharmaguard_investigations_assisted",
              "pharmaguard_human_reviews", "pharmaguard_indicator_status", "pharmaguard_overall_status", "pharmaguard_health_check_up"):
        assert n in names, n


def test_alert_rules_and_grafana_only_reference_exported_metrics(prom):
    names = metric_names(prom)
    rules = yaml.safe_load((PROJECT_ROOT / "monitoring/prometheus/alerts.yml").read_text())
    exprs = [r["expr"] for g in rules["groups"] for r in g["rules"]]
    assert exprs and all(r["labels"]["severity"] in ("warn", "critical") and r["annotations"]["runbook"] for g in rules["groups"] for r in g["rules"])
    dash = json.loads((PROJECT_ROOT / "monitoring/grafana/pharmaguard-dashboard.json").read_text())
    exprs += [t["expr"] for p in dash["panels"] for t in p["targets"]]
    for e in exprs:
        for m in re.findall(r"\b(pharmaguard_[a-z_]+)", e):
            assert m in names, f"{m} in '{e}' is not exported"
    assert {p["title"] for p in dash["panels"]} >= {"Availability", "Data drift (PSI) by feature", "LLM tokens", "Human review decisions"}


def test_alert_thresholds_match_slo_config():
    rules = yaml.safe_load((PROJECT_ROOT / "monitoring/prometheus/alerts.yml").read_text())
    by = {r["alert"]: r["expr"] for g in rules["groups"] for r in g["rules"]}
    assert "0.995" in by["PharmaGuardAvailabilityLow"] and str(MON["slo"]["availability_pct"]["critical"] / 100) in by["PharmaGuardAvailabilityCritical"]
    assert str(int(MON["slo"]["latency_p95_ms"]["warn"])) in by["PharmaGuardLatencyP95High"]
    assert "0.10" in by["PharmaGuardDataDrift"] and "25" in by["PharmaGuardReviewBacklog"]


def test_prometheus_and_otel_configs_parse_and_docs_exist():
    prom = yaml.safe_load((PROJECT_ROOT / "monitoring/prometheus/prometheus.yml").read_text())
    assert prom["scrape_configs"][0]["job_name"] == "pharmaguard" and "alerts.yml" in prom["rule_files"]
    otel = yaml.safe_load((PROJECT_ROOT / "monitoring/otel/otel-collector.yaml").read_text())
    assert {"otlp"} == set(otel["receivers"]) and set(otel["service"]["pipelines"]) == {"traces", "metrics", "logs"}
    doc = (PROJECT_ROOT / "docs/observability-integration.md").read_text()
    for word in ("OpenTelemetry", "Prometheus", "Grafana", "OTLP", "Datadog", "cardinality", "metrics_server.py", "real vs simulated"):
        assert word.lower() in doc.lower(), word
    assert not re.search(r"(sk-[A-Za-z0-9]{20}|AKIA[0-9A-Z]{16})", doc + json.dumps(otel))


def test_metrics_server_once_and_http(tmp_path):
    env = {"PHARMAGUARD_OBS_DB": str(tmp_path / "t.db"), "PHARMAGUARD_DB_PATH": str(tmp_path / "r.db"), "PATH": "/usr/bin:/bin"}
    out = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/metrics_server.py"), "--once"], capture_output=True, text=True, env=env, cwd=PROJECT_ROOT)
    assert out.returncode == 0 and "pharmaguard_overall_status" in out.stdout
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = subprocess.Popen([sys.executable, str(PROJECT_ROOT / "scripts/metrics_server.py"), "--port", str(port)], env=env, cwd=PROJECT_ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                body = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=3)
                break
            except OSError:
                time.sleep(0.2)
        assert body.status == 200 and "text/plain" in body.headers["Content-Type"] and b"pharmaguard_" in body.read()
        assert urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=3).read() == b"ok\n"
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nope", timeout=3)
        assert e.value.code == 404
    finally:
        srv.terminate()
        srv.wait(timeout=5)


# ---- dashboard page -----------------------------------------------------------------------------------------------
def _page(role, name="Ada"):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(page("app/pages/8_Production_Monitoring.py"), default_timeout=180)
    at.session_state["user_name"], at.session_state["user_role"] = name, role
    return at.run()


def test_monitoring_page_access_control_logs_denials(tmp_path):
    at = _page("viewer", "Vic")
    assert any("cannot view monitoring" in e.value for e in at.error) and not at.exception
    ev = TraceStore(tmp_path / "test_obs.db").security_events()                      # PHARMAGUARD_OBS_DB is set by conftest
    assert ev.iloc[0]["event_type"] == "UNAUTHORIZED_ACCESS" and ev.iloc[0]["user_name"] == "Vic" and ev.iloc[0]["user_role"] == "viewer"


def test_monitoring_page_renders_all_sections_and_simulations():
    at = _page("compliance_admin")
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Overview", "Application", "AI", "Data", "Security", "Cost", "Business", "Incidents & events", "Health", "Integration"]
    labels = [m.label for m in at.metric]
    for cat in ("Application", "AI", "Data", "Security", "Cost", "Business"):
        assert any(cat in l for l in labels), cat
    assert not [b for b in at.button if "Simulate production" in b.label][0].disabled
    at.session_state["sim_drift"], at.session_state["sim_schema"] = True, True
    at.run()
    assert not at.exception and any("SIMULATION ACTIVE" in m.value for m in at.markdown)
    assert any("CRITICAL" in m.value for m in at.markdown if "banner" in m.value)          # simulated drift + schema change -> incident


def test_monitoring_page_reviewer_cannot_simulate():
    at = _page("qa_reviewer", "Rita")
    assert not at.exception and [b for b in at.button if "Simulate production" in b.label][0].disabled


def test_permission_denials_on_other_pages_are_recorded(tmp_path):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(page("app/pages/3_Batch_Investigation.py"), default_timeout=120)
    at.session_state["user_name"], at.session_state["user_role"] = "Vic", "viewer"
    at.run()
    [t for t in at.text_input if t.label == "Batch_ID"][0].set_value("B00042").run()
    at.button[0].click().run()
    assert not at.exception
    ev = TraceStore(tmp_path / "test_obs.db").security_events()
    assert "PERMISSION_DENIED" in set(ev["event_type"]) and set(ev["source"]) == {"Batch Investigation page"}
