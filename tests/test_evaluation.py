import json
from pathlib import Path

import pytest

import src.assistant.analysis as analysis_mod
from config import PROJECT_ROOT
from src.analytics import kpis
from src.assistant.knowledge import KnowledgeBase
from src.assistant.pipeline import investigate
from src.assistant import llm as L
from src.evaluation import metrics as M
from src.evaluation import oracle
from src.evaluation.golden import build_golden, load_golden
from src.evaluation.runner import run_evaluation
from src.guardrails import input_guard as ig
from src.guardrails import output_guard as og
from tests.test_assistant import hist, kb, make_history  # noqa: F401

GOLDEN = PROJECT_ROOT / "data" / "evaluation" / "golden_cases.json"
CLEAN = PROJECT_ROOT / "data" / "processed" / "pharma_batch_clean.csv"
needs_data = pytest.mark.skipif(not (GOLDEN.exists() and CLEAN.exists()), reason="golden/cleaned data not generated")


@pytest.fixture(scope="module")
def clean_df():
    return kpis.load_clean(CLEAN)


@pytest.fixture(scope="module")
def golden():
    return load_golden(GOLDEN)


@pytest.fixture(scope="module")
def real_kb():
    return KnowledgeBase.from_directory(PROJECT_ROOT / "knowledge")


# ---- golden dataset ---------------------------------------------------------------------------
@needs_data
def test_golden_dataset_structure(golden):
    cases = golden["cases"]
    assert len(cases) >= 60 and len({c["case_id"] for c in cases}) == len(cases)
    cats = {c["category"] for c in cases}
    assert cats == {"investigate", "why_risk", "compare", "abnormal", "procedure", "prompt_injection", "disposition_request",
                    "invalid_input", "adversarial_llm_output"}
    assert len(golden["archetype_batches"]) >= 6
    for c in cases:
        assert {"question", "batch_id", "expected", "guardrail", "archetype"} <= set(c)
        assert {"input", "blocked", "llm_calls", "output"} <= set(c["guardrail"])
        assert "answer_keywords" in c["expected"] and "forbidden" in c["expected"]
    analytic = [c for c in cases if c["category"] in ("investigate", "why_risk", "compare", "abnormal", "procedure")]
    assert all(c["expected"]["analysis"]["status"] for c in analytic)
    assert all("source_docs" in c["expected"] and "acceptable_docs" in c["expected"] for c in analytic)


@needs_data
def test_golden_file_is_reproducible_from_current_data(golden, clean_df, cfg):
    """Fails when data/config change without regenerating the golden set (python scripts/build_golden.py)."""
    fresh = json.loads(json.dumps(build_golden(clean_df, cfg), default=str))
    assert fresh == golden


def test_oracle_is_independent_of_the_code_under_test():
    import ast
    tree = ast.parse((PROJECT_ROOT / "src" / "evaluation" / "oracle.py").read_text())
    imported = [n.module if isinstance(n, ast.ImportFrom) else a.name for n in ast.walk(tree)
                if isinstance(n, (ast.Import, ast.ImportFrom)) for a in getattr(n, "names", [None])]
    assert not any(str(m).startswith(("src", "config")) for m in imported)


def test_oracle_expected_findings(hist, cfg):
    exp = oracle.expected_analysis(hist, "B09999", cfg["specs"])
    assert exp["out_of_spec_parameters"] == ["Dissolution", "Impurity_Level", "Temperature"]
    assert exp["deviation_flag"] == "elevated" and exp["peer_batches"] == 60 and exp["peer_fail_rate_pct"] == 0.0
    assert exp["imputed_parameters"] == ["pH"] and exp["status"] == "Fail"
    d = next(c for c in exp["checks"] if c["parameter"] == "Dissolution")
    assert (d["breach"], d["breach_amount"]) == ("below", 10.0)


# ---- metric unit tests ------------------------------------------------------------------------
def _inv(hist, cfg, kb, question="Investigate this batch."):
    return investigate(question, "B09999", hist, kb, cfg, L.LLMConfig())


@pytest.fixture
def kb_small():
    return KnowledgeBase.from_directory(PROJECT_ROOT / "knowledge")


def test_analytical_correctness_metric(hist, cfg, kb_small):
    inv, exp = _inv(hist, cfg, kb_small), oracle.expected_analysis(hist, "B09999", cfg["specs"])
    assert M.analytical_correctness(inv, exp) == (1.0, [])
    inv.analysis["out_of_spec_parameters"] = ["Assay"]
    inv.analysis["batch"]["Quality_Status"] = "Pass"
    score, fails = M.analytical_correctness(inv, exp)
    assert score < 1 and {"status", "out_of_spec_set"} <= set(fails)


def test_retrieval_metric(hist, cfg, kb_small):
    inv = _inv(hist, cfg, kb_small)
    good = {"source_docs": ["batch_quality_review"], "acceptable_docs": ["batch_quality_review", "deviation_management",
                                                                       "dissolution_investigation", "temperature_excursion"]}
    r = M.retrieval(inv, good)
    assert r["recall"] == 1.0 and 0 < r["precision"] <= 1 and 0 < r["f1"] <= 1
    strict = M.retrieval(inv, {"source_docs": ["batch_quality_review"], "acceptable_docs": ["batch_quality_review"]})
    assert strict["precision"] < r["precision"]
    assert M.retrieval(inv, {"source_docs": ["nonexistent_doc"], "acceptable_docs": []})["recall"] == 0.0
    assert M.retrieval(inv, {"source_docs": [], "acceptable_docs": []}) is None


def test_groundedness_and_unsupported_claim_metrics(hist, cfg, kb_small):
    inv = _inv(hist, cfg, kb_small)
    assert M.groundedness(inv) == 1.0 and M.unsupported_claims(inv) == []
    inv.answer += " Failure probability is 87.3% per batch B77777 [K9]."
    assert M.groundedness(inv) < 1.0
    assert {"UNGROUNDED_NUMBER", "UNKNOWN_BATCH_REFERENCE", "INVALID_CITATION"} <= set(M.unsupported_claims(inv))


def test_answer_relevance_metric(hist, cfg, kb_small):
    inv = _inv(hist, cfg, kb_small)
    assert M.answer_relevance(inv, {"answer_keywords": [["risk"], ["B09999"]], "intent": "investigate"})[0] == 1.0
    score, missed = M.answer_relevance(inv, {"answer_keywords": [["zebra"], ["risk"]]})
    assert score == 0.5 and missed == ["zebra"]


def test_guardrail_and_tool_metrics(hist, cfg, kb_small):
    inv = _inv(hist, cfg, kb_small)
    case = {"question": "Investigate this batch.", "category": "investigate", "archetype": "x",
            "guardrail": {"input": "passed", "blocked": False, "llm_calls": 0, "output": "passed"}, "expected": {"forbidden": []}}
    assert M.guardrail_success(inv, case, 0, "")[0] == 1.0
    case["guardrail"]["llm_calls"] = 1
    score, fails = M.guardrail_success(inv, case, 0, "")
    assert score < 1 and fails == ["llm_calls"]
    assert M.tool_success(inv, case, True) == {"batch_lookup": True, "analysis": True, "retrieval": True,
                                               "answer_generation": True, "persistence": True}
    assert M.tool_success(inv, case, False)["persistence"] is False
    assert M.tool_success(inv, {**case, "category": "prompt_injection"}, None) is None


def test_percentile():
    assert M.percentile([1, 2, 3, 4, 5], .5) == 3 and M.percentile([10], .95) == 10
    assert M.percentile([0, 100], .95) == pytest.approx(95)
    assert M.percentile([], .5) != M.percentile([], .5)          # NaN


# ---- full run on the real golden set ----------------------------------------------------------
@needs_data
def test_full_evaluation_meets_all_targets(golden, clean_df, real_kb, cfg, tmp_path):
    rep = run_evaluation(golden, clean_df, real_kb, cfg, save_to=tmp_path / "r.json")
    s = rep["summary"]
    failed = [(c["case_id"], c["failures"]) for c in rep["cases"] if not c["passed"]]
    assert not failed, failed
    assert s["all_targets_met"], {k: v for k, v in s["target_status"].items() if not v}
    m = s["metrics"]
    assert m["analytical_correctness"] == 1.0 and m["guardrail_success"] == 1.0 and m["unsafe_output_catch_rate"] == 1.0
    assert m["latency_p95_ms"] <= cfg["evaluation"]["targets"]["latency_p95_ms_max"]
    assert (tmp_path / "r.json").exists() and json.loads((tmp_path / "r.json").read_text())["summary"]["cases"] == len(golden["cases"])
    assert set(s["tool_stage_success"]) >= {"batch_lookup", "analysis", "retrieval", "answer_generation", "persistence"}


# ---- the evaluation must be able to FAIL: break a component, expect a red metric --------------
@needs_data
def test_eval_detects_broken_injection_defence(monkeypatch, golden, clean_df, real_kb, cfg):
    monkeypatch.setattr(ig, "detect_injection", lambda text: False)
    s = run_evaluation(golden, clean_df, real_kb, cfg)["summary"]
    assert s["metrics"]["guardrail_success"] < 1 and not s["target_status"]["guardrail_success"]
    assert s["by_category"]["prompt_injection"]["passed"] == 0


@needs_data
def test_eval_detects_disabled_output_validation(monkeypatch, golden, clean_df, real_kb, cfg):
    monkeypatch.setattr(og, "validate_output", lambda answer, *a, **k: og.OutputCheck(answer, []))
    s = run_evaluation(golden, clean_df, real_kb, cfg)["summary"]
    assert s["metrics"]["unsafe_output_catch_rate"] < 1 and s["by_category"]["adversarial_llm_output"]["passed"] < 11


@needs_data
def test_eval_detects_wrong_analysis(monkeypatch, golden, clean_df, real_kb, cfg):
    real = analysis_mod.analyze_batch

    def broken(*a, **k):
        r = real(*a, **k)
        if r:
            r["out_of_spec_parameters"], r["deviations"]["flag"] = [], "normal"
        return r

    monkeypatch.setattr(analysis_mod, "analyze_batch", broken)
    s = run_evaluation(golden, clean_df, real_kb, cfg)["summary"]
    assert s["metrics"]["analytical_correctness"] < cfg["evaluation"]["targets"]["analytical_correctness"]
    assert not s["target_status"]["analytical_correctness"]


@needs_data
def test_eval_detects_lost_retrieval(monkeypatch, golden, clean_df, real_kb, cfg):
    monkeypatch.setattr(type(real_kb), "search", lambda self, *a, **k: [])
    s = run_evaluation(golden, clean_df, real_kb, cfg)["summary"]
    assert s["metrics"]["retrieval_recall"] == 0.0 and s["metrics"]["tool_success"] < 1


# ---- dashboard --------------------------------------------------------------------------------
@needs_data
def test_dashboard_renders_report(golden, clean_df, real_kb, cfg, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    path = tmp_path / "latest.json"
    run_evaluation(golden, clean_df, real_kb, cfg, save_to=path)
    monkeypatch.setenv("PHARMAGUARD_EVAL_PATH", str(path))
    at = AppTest.from_file("app/pages/5_Evaluation_Dashboard.py", default_timeout=120).run()
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    for k in ("Analytical correctness", "Retrieval relevance", "Groundedness", "Answer relevance", "Unsupported-claim rate",
              "Guardrail success", "Tool execution success", "Latency p95", "Overall score", "Cases passed"):
        assert k in labels
    assert labels["Cases passed"] == f"{len(golden['cases'])} / {len(golden['cases'])}"


def test_dashboard_without_results_shows_guidance(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("PHARMAGUARD_EVAL_PATH", str(tmp_path / "missing.json"))
    at = AppTest.from_file("app/pages/5_Evaluation_Dashboard.py", default_timeout=60).run()
    assert not at.exception and any("No evaluation results" in i.value for i in at.info)
