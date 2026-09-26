import json

import numpy as np
import pandas as pd
import pytest

from src.assistant import analysis as A
from src.assistant.grounding import ungrounded_numbers
from src.assistant.intents import classify, sections_for
from src.assistant.knowledge import KnowledgeBase
from src.assistant import llm as L
from src.assistant.pipeline import SYSTEM_PROMPT, build_knowledge_query, build_user_prompt, investigate
from config import PROJECT_ROOT


def make_history(cfg, n=60):
    """Deterministic synthetic history: n normal batches of one product plus one target batch."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rows.append({"Batch_ID": f"B{i + 1:05d}", "Product_Name": "P1", "Dosage_Form": "Tablet",
                     "Manufacturing_Date": "2024-01-01", "Plant": "Plant_A" if i % 2 else "Plant_B",
                     "Operator_Shift": "Morning", "Equipment_ID": "EQ-A01",
                     "Temperature": rng.normal(22.5, 0.5), "Humidity": rng.normal(45, 2), "pH": rng.normal(6.5, 0.1),
                     "Dissolution": rng.normal(92, 2), "Assay": rng.normal(100, 0.5), "Impurity_Level": abs(rng.normal(0.3, 0.05)),
                     "Yield_Percentage": rng.normal(96, 1), "Cycle_Time_Hrs": rng.normal(9, 0.3), "Batch_Size_Kg": rng.normal(400, 5),
                     "Deviation_Count": 1, "Quality_Status": "Pass", "Outlier_Flag": False, "Outlier_Columns": np.nan,
                     "Imputed_Columns": np.nan, "Has_Missing_CQA": False})
    target = {**rows[0], "Batch_ID": "B09999", "Temperature": 45.0, "Dissolution": 70.0, "Impurity_Level": 1.4,
              "Deviation_Count": 4, "Quality_Status": "Fail", "Outlier_Columns": "Temperature", "Imputed_Columns": "pH"}
    return pd.DataFrame(rows + [target])


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase.from_directory(PROJECT_ROOT / "knowledge")


@pytest.fixture
def hist(cfg):
    return make_history(cfg)


# --- deterministic analysis --------------------------------------------------------------------
def test_analysis_flags_and_numbers(hist, cfg):
    a = A.analyze_batch(hist, "b09999", cfg)                          # case-insensitive lookup
    p = {x["parameter"]: x for x in a["parameters"]}
    assert p["Dissolution"]["flag"] == "out_of_spec" and p["Dissolution"]["breach"] == "below"
    assert p["Dissolution"]["breach_amount"] == 10.0                  # 80 - 70
    assert p["Temperature"]["flag"] == "out_of_spec" and p["Temperature"]["breach_amount"] == 18.0
    assert p["Impurity_Level"]["breach"] == "above" and p["Impurity_Level"]["breach_amount"] == pytest.approx(0.4)
    assert p["pH"]["flag"] == "imputed_estimate"
    peers = hist[hist["Batch_ID"] != "B09999"]["Dissolution"]
    assert p["Dissolution"]["peer_mean"] == pytest.approx(peers.mean(), abs=1e-3)
    assert p["Dissolution"]["z_score"] == pytest.approx((70 - peers.mean()) / peers.std(), abs=0.01)
    assert p["Dissolution"]["percentile"] == 0.0
    assert set(a["out_of_spec_parameters"]) == {"Temperature", "Dissolution", "Impurity_Level"}
    assert "Deviation_Count" in a["abnormal_parameters"] and a["deviations"]["flag"] == "elevated"
    assert a["data_notes"]["imputed_parameters"] == ["pH"]
    assert a["history"]["peer_batches"] == 60 and a["history"]["peer_fail_rate_pct"] == 0.0


def test_analysis_unknown_batch_and_suggestions(hist, cfg):
    assert A.analyze_batch(hist, "B12345", cfg) is None
    assert "B09999" in A.suggest_batch_ids(hist, "B09998")
    assert A.extract_batch_id("please look at b00042, thanks") == "B00042"
    assert A.extract_batch_id("no id here") is None


def test_analysis_result_is_json_serialisable(hist, cfg):
    json.dumps(A.analyze_batch(hist, "B09999", cfg))


# --- intents & retrieval -----------------------------------------------------------------------
def test_intent_classification():
    assert classify("Investigate this batch.") == ["investigate"]
    assert "why_risk" in classify("Why is this batch showing risk?")
    assert "compare" in classify("Compare it with historical batches.")
    assert "abnormal" in classify("Which parameters are abnormal?")
    assert "procedure" in classify("What procedure should QA review?")
    assert classify("hello") == ["investigate"]
    assert sections_for(["investigate"]) == ["why_risk", "abnormal", "compare", "procedure"]
    assert sections_for(["compare"]) == ["compare"]


def test_knowledge_base_loads_all_four_documents(kb):
    assert {c.doc_id for c in kb.chunks} == {"dissolution_investigation", "temperature_excursion",
                                              "deviation_management", "batch_quality_review"}
    assert all(c.note.startswith("SYNTHETIC") for c in kb.chunks)


@pytest.mark.parametrize("query,doc", [
    ("dissolution result below specification laboratory investigation", "dissolution_investigation"),
    ("temperature excursion above the validated range", "temperature_excursion"),
    ("how to classify a deviation and CAPA", "deviation_management"),
    ("release checklist status criteria pass fail", "batch_quality_review"),
])
def test_retrieval_finds_the_right_document(kb, query, doc):
    assert kb.search(query, top_k=3)[0].chunk.doc_id == doc


def test_retrieval_query_uses_findings(hist, cfg, kb):
    a = A.analyze_batch(hist, "B09999", cfg)
    q = build_knowledge_query("Why is this risky?", a, ["why_risk"])
    assert "dissolution" in q and "temperature excursion" in q and "deviation" in q
    ids = {h.chunk.doc_id for h in kb.search(q, top_k=6)}
    assert {"dissolution_investigation", "temperature_excursion"} <= ids


# --- LLM configuration & guard-rails -----------------------------------------------------------
def test_llm_config_from_env():
    assert not L.LLMConfig.from_env({}).is_live
    assert not L.LLMConfig.from_env({"LLM_API_KEY": "changeme"}).is_live
    assert not L.LLMConfig.from_env({"LLM_PROVIDER": "mock", "LLM_API_KEY": "k"}).is_live
    c = L.LLMConfig.from_env({"LLM_API_KEY": "k"})
    assert c.is_live and c.provider == "anthropic" and c.model == "claude-sonnet-5"
    c = L.LLMConfig.from_env({"LLM_PROVIDER": "openai", "LLM_API_KEY": "k", "LLM_MODEL": "m", "LLM_BASE_URL": "http://x/v1/"})
    assert (c.model, c.base_url) == ("m", "http://x/v1")
    assert "Demo mode" in L.LLMConfig.from_env({}).describe()


def test_llm_request_shapes_and_parsing():
    seen = {}

    def fake(url, headers, payload, timeout):
        seen.update(url=url, headers=headers, payload=payload)
        return {"content": [{"type": "text", "text": "hi"}]} if "anthropic" in url else {"choices": [{"message": {"content": "yo"}}]}

    a = L.complete(L.LLMConfig.from_env({"LLM_API_KEY": "sk"}), "sys", "usr", fake)
    assert a == "hi" and seen["url"].endswith("/v1/messages") and seen["headers"]["x-api-key"] == "sk"
    assert seen["payload"]["system"] == "sys" and seen["payload"]["messages"][0]["content"] == "usr"
    o = L.complete(L.LLMConfig.from_env({"LLM_PROVIDER": "openai", "LLM_API_KEY": "sk"}), "sys", "usr", fake)
    assert o == "yo" and seen["url"].endswith("/chat/completions") and seen["headers"]["Authorization"] == "Bearer sk"
    with pytest.raises(L.LLMError):
        L.complete(L.LLMConfig(), "s", "u", fake)                      # not configured


def test_grounding_check():
    facts = ["Assay 99.65% spec 95-105, z-score 3.21, 1,234 batches"]
    assert ungrounded_numbers("Assay was 99.65 % (95-105), z 3.21 [K1] on 2024-03-05, 3 items", *facts) == []
    assert ungrounded_numbers("Assay was 99.7", *facts) == []          # rounding of a supplied number is tolerated
    assert ungrounded_numbers("Failure rate is 12.5% and 87 batches", *facts) == ["12.5", "87"]


def test_system_prompt_forbids_calculation_and_prompt_carries_data(hist, cfg, kb):
    assert "NEVER calculate" in SYSTEM_PROMPT
    a = A.analyze_batch(hist, "B09999", cfg)
    hits = kb.search("dissolution investigation", top_k=2)
    prompt = build_user_prompt("Investigate", ["investigate"], a, hits)
    assert "BATCH_ANALYSIS" in prompt and '"breach_amount": 10.0' in prompt and "[K1]" in prompt


# --- end to end --------------------------------------------------------------------------------
def test_pipeline_demo_mode_without_key(hist, cfg, kb):
    inv = investigate("Investigate this batch.", "B09999", hist, kb, cfg, L.LLMConfig())
    assert inv.found and inv.mode == "demo" and inv.hits
    assert "Dissolution" in inv.answer and "[K1]" in inv.answer and "Fail" in inv.answer
    assert [s["step"][0] for s in inv.steps] == ["1", "2", "3", "4", "5"]
    assert ungrounded_numbers(inv.answer, json.dumps(inv.analysis, default=str), " ".join(h.chunk.text for h in inv.hits)) == []


def test_pipeline_batch_id_from_question_and_not_found(hist, cfg, kb):
    inv = investigate("Why is b09999 risky?", "", hist, kb, cfg, L.LLMConfig())
    assert inv.found and inv.batch_id == "B09999"
    nf = investigate("Investigate", "B09998", hist, kb, cfg, L.LLMConfig())
    assert not nf.found and "not found" in nf.answer
    bad = investigate("Investigate", "B09998x", hist, kb, cfg, L.LLMConfig())
    assert not bad.found and "not a valid Batch_ID" in bad.answer
    assert not investigate("Investigate", "", hist, kb, cfg, L.LLMConfig()).found


def test_pipeline_live_llm_invented_numbers_are_blocked(hist, cfg, kb):
    live = L.LLMConfig.from_env({"LLM_API_KEY": "k"})
    calls = []

    def fake(url, headers, payload, timeout):
        calls.append(payload)
        return {"content": [{"type": "text", "text": "Dissolution 70% is below 80 [K1]. Failure probability is 87.3%."}]}

    inv = investigate("Why risk?", "B09999", hist, kb, cfg, live, fake)
    assert "BATCH_ANALYSIS" in calls[0]["messages"][0]["content"]
    assert inv.mode == "demo" and inv.grounding_warnings and "87.3" in inv.grounding_warnings[0]   # fail closed
    assert inv.guardrail["output"].startswith("blocked") and "87.3" not in inv.answer


def test_pipeline_live_llm_valid_answer_passes_and_gets_disclaimer(hist, cfg, kb):
    live = L.LLMConfig.from_env({"LLM_API_KEY": "k"})
    ok = {"content": [{"type": "text", "text": "Dissolution is 70% (specification 80-105%), below the limit [K1]."}]}
    inv = investigate("Why risk?", "B09999", hist, kb, cfg, live, lambda *a, **k: ok)
    assert inv.mode == "llm" and inv.guardrail["output"] == "passed" and "Decision support only" in inv.answer


def test_pipeline_falls_back_to_demo_when_llm_fails(hist, cfg, kb):
    live = L.LLMConfig.from_env({"LLM_API_KEY": "k"})

    def boom(*a, **k):
        raise L.LLMError("HTTP 500")

    inv = investigate("Investigate this batch.", "B09999", hist, kb, cfg, live, boom)
    assert inv.mode == "demo" and "HTTP 500" in inv.fallback_reason and inv.answer


def test_assistant_page_renders_in_demo_mode(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    at = AppTest.from_file("app/pages/3_Batch_Investigation.py", default_timeout=120).run()
    at.text_input[0].set_value("B00042").run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception and len(at.chat_message) == 2
    at.text_input[0].set_value("B99999").run()
    assert not at.exception and at.error
