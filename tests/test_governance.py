import json
import shutil

import pytest
import yaml

from config import PROJECT_ROOT
from src.assistant.pipeline import SYSTEM_PROMPT
from src.governance import metadata as gm
from src.governance.report import Report, build_report
from src.guardrails import output_guard
from src.security import controls, secrets_scan


@pytest.fixture(scope="module")
def meta():
    return gm.load_governance()


# ---- metadata ---------------------------------------------------------------------------------
def test_metadata_has_every_required_governance_field(meta):
    s = meta["system"]
    for k in ("use_case", "business_owner", "technical_owner", "intended_users"):
        assert s[k]
    assert meta["data_sources"] and meta["data_classification"]["current"] and meta["prompt"]["version"]
    assert meta["risks"] and meta["limitations"] and meta["guardrails"] and meta["human_oversight"] and meta["change_history"]
    for r in meta["risks"]:
        assert {"id", "risk", "likelihood", "impact", "mitigation", "controls", "residual"} <= set(r)
        assert r["likelihood"] in ("Low", "Medium", "High") and r["impact"] in ("Low", "Medium", "High", "Critical")
        assert isinstance(r["controls"], list) and r["controls"]
    assert len({r["id"] for r in meta["risks"]}) == len(meta["risks"])


def test_governance_is_honest_about_approval_and_owners(meta):
    assert "not approved" in meta["approval"]["status"].lower()
    assert all(a["status"] == "Pending" for a in meta["approval"]["approvals"])          # nothing pre-approved
    assert "role" in meta["system"]["business_owner"].lower() and "role" in meta["system"]["technical_owner"].lower()


def test_guardrail_implementations_exist_in_the_repo(meta):
    for g in meta["guardrails"]:
        assert (PROJECT_ROOT / g["implementation"]).exists(), g


def test_change_history_is_consistent(meta):
    vs = [tuple(map(int, c["version"].split("."))) for c in meta["change_history"]]
    assert vs == sorted(vs) and len(set(vs)) == len(vs)
    assert meta["system"]["version"].startswith(meta["change_history"][-1]["version"])


def test_missing_fields_are_rejected(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump({"system": {"name": "x"}}))
    with pytest.raises(ValueError, match="missing"):
        gm.load_governance(p)


# ---- live status ------------------------------------------------------------------------------
def test_model_status_demo_and_live():
    assert gm.model_status({})["mode"] == "Demo (no LLM)" and gm.model_status({})["data_leaves_environment"] == "No"
    live = gm.model_status({"LLM_API_KEY": "k"})
    assert live["mode"] == "Live LLM" and live["pinned"].startswith("No") and "anthropic" in live["data_leaves_environment"]
    assert gm.model_status({"LLM_API_KEY": "k", "LLM_MODEL": "claude-x"})["pinned"].startswith("Yes")
    assert "k" != live["model"]                                                          # the key is never exposed


def test_prompt_fingerprint_ignores_the_random_canary():
    import hashlib
    assert gm.prompt_fingerprint() == hashlib.sha256(SYSTEM_PROMPT.replace(output_guard.CANARY, "<canary>").encode()).hexdigest()
    assert output_guard.CANARY not in json.dumps(gm.fingerprints())


def test_fingerprint_baseline_detects_drift(tmp_path):
    (tmp_path / "config").mkdir(); (tmp_path / "knowledge").mkdir(); (tmp_path / "data" / "evaluation").mkdir(parents=True)
    (tmp_path / "config" / "config.yaml").write_text("a: 1\n")
    (tmp_path / "knowledge" / "x.md").write_text("# doc\n")
    (tmp_path / "data" / "evaluation" / "golden_cases.json").write_text("{}")
    assert {r["status"] for r in gm.fingerprint_status(tmp_path)} == {"no baseline"}
    gm.register_baseline("9.9.9", tmp_path)
    assert {r["status"] for r in gm.fingerprint_status(tmp_path)} == {"matches baseline"}
    (tmp_path / "knowledge" / "x.md").write_text("# doc changed\n")
    st = {r["artifact"]: r["status"] for r in gm.fingerprint_status(tmp_path)}
    assert st["Knowledge documents"] == "CHANGED since baseline" and st["Project configuration"] == "matches baseline"


def test_registered_baseline_matches_current_project():
    """Fails when the prompt, knowledge, config or golden set change without re-registering
    (python scripts/register_baseline.py after review + evaluation) - that is the drift control working."""
    assert {r["artifact"]: r["status"] for r in gm.fingerprint_status()} == {
        "System prompt": "matches baseline", "Knowledge documents": "matches baseline",
        "Project configuration": "matches baseline", "Golden evaluation dataset": "matches baseline"}


def test_evaluation_status(cfg, tmp_path, monkeypatch):
    res = PROJECT_ROOT / cfg["evaluation"]["results_path"]
    if not res.exists():
        pytest.skip("evaluation not run")
    ev = gm.evaluation_status(cfg)
    assert ev["available"] and ev["targets_met"] and not ev["stale"] and ev["status"].startswith("Passed")
    stale = json.loads(res.read_text()); stale["golden_dataset_md5"] = "different"
    (tmp_path / "r.json").write_text(json.dumps(stale))
    monkeypatch.setenv("PHARMAGUARD_EVAL_PATH", str(tmp_path / "r.json"))
    assert gm.evaluation_status(cfg)["stale"] is True
    monkeypatch.setenv("PHARMAGUARD_EVAL_PATH", str(tmp_path / "none.json"))
    assert gm.evaluation_status(cfg) == {"available": False, "status": "Not evaluated"}


# ---- report -----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def report(meta, cfg):
    env = secrets_scan.env_variable_status(PROJECT_ROOT, {"LLM_API_KEY": "SUPER-SECRET-VALUE-123"})
    return build_report(meta, gm.model_status({"LLM_API_KEY": "SUPER-SECRET-VALUE-123"}), gm.evaluation_status(cfg), gm.fingerprint_status(),
                        controls.run_controls(), env, generated_by="Rita")


def test_report_contains_all_required_sections(report, meta):
    md = report.markdown()
    for heading in ("AI system", "Data", "Model and prompt", "Risks", "Known limitations", "Guardrails", "Human oversight",
                    "Evaluation status", "Approval status", "Change history", "Security controls", "Secret-management guidance"):
        assert heading in md, heading
    for v in (meta["system"]["business_owner"], meta["system"]["technical_owner"], meta["prompt"]["version"],
              meta["data_classification"]["current"], meta["approval"]["status"], "SEC-01", "generated"):
        assert v in md, v
    assert md.startswith("# AI Governance Report") and "by Rita" in md
    assert all(r["id"] in md for r in meta["risks"]) and all(c["version"] in md for c in meta["change_history"])


def test_report_never_leaks_secret_values(report):
    for text in (report.markdown(), report.html()):
        assert "SUPER-SECRET-VALUE-123" not in text


def test_report_html_is_standalone_and_escaped():
    r = Report("T <b>")
    r.h("x")
    r.p("<script>alert(1)</script>")
    r.table(["a|b"], [["<img src=x onerror=alert(1)>"]])
    html_doc = r.html()
    assert html_doc.startswith("<!doctype html>") and "<script>" not in html_doc and "<img" not in html_doc and "&lt;script&gt;" in html_doc
    assert "a\\|b" in r.markdown() or "a|b" in r.markdown()


def test_markdown_table_escapes_pipes_and_newlines():
    r = Report("T")
    r.table(["h"], [["a|b\nc"]])
    assert "a\\|b c" in r.markdown()


# ---- page -------------------------------------------------------------------------------------
def _page(role, name="Tester"):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app/pages/6_AI_Governance.py", default_timeout=120)
    at.session_state["user_name"], at.session_state["user_role"] = name, role
    return at.run()


def test_governance_page_renders_with_all_sections():
    at = _page("viewer")
    assert not at.exception
    assert [t.label for t in at.tabs] == ["System card", "Risks & limitations", "Guardrails & oversight", "Evaluation & approval",
                                          "Change history", "Security controls", "Governance report"]
    labels = {m.label: m.value for m in at.metric}
    assert labels["Approval status"] == "Not approved for production use" and labels["Evaluation"] == "Passed"


def test_governance_page_permissions():
    viewer = _page("viewer")
    assert any("Downloading requires" in i.value for i in viewer.info)
    assert [b for b in viewer.button if "dependency audit" in b.label][0].disabled
    admin = _page("compliance_admin")
    assert not any("Downloading requires" in i.value for i in admin.info)
    assert not [b for b in admin.button if "dependency audit" in b.label][0].disabled
    assert not _page("qa_reviewer").exception


def test_ai_request_logging_omits_question_text(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("PHARMAGUARD_LOG_DIR", str(tmp_path / "logs"))
    at = AppTest.from_file("app/pages/3_Batch_Investigation.py", default_timeout=120).run()
    [t for t in at.text_input if t.label == "Batch_ID"][0].set_value("B00042").run()
    at.button[1].click().run()                               # "Why is this batch showing risk?"
    assert not at.exception
    text = (tmp_path / "logs" / "security_audit.log").read_text()
    rec = json.loads(text.splitlines()[-1])
    assert rec["event"] == "ai_request" and rec["batch_id"] == "B00042" and rec["run_id"].startswith("RUN-")
    assert "showing risk" not in text and rec["question_len"] > 0 and "question" not in rec
