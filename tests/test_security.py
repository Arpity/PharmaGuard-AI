import json
import subprocess
from datetime import date

import pytest

from config import PROJECT_ROOT
from src.guardrails import roles
from src.security import controls, dependencies, secrets_scan
from src.security.secure_logging import RedactingFilter, get_security_logger, log_event

# Fake credentials are assembled at runtime so this file itself never contains a literal secret.
FAKE_KEY = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2"
FAKE_AWS = "AKIA" + "ABCDEFGHIJKLMNOP"


# ---- secret scanning --------------------------------------------------------------------------
@pytest.mark.parametrize("line,rule", [
    (f'client = Client("{FAKE_KEY}")', "provider-style API key"),
    (f"aws = '{FAKE_AWS}'", "AWS access key id"),
    ("-----BEGIN RSA PRIVATE KEY-----", "private key block"),
    ("h = 'Authorization: Bearer " + "abcdef0123456789abcdef0123456789'", "bearer token literal"),
    ('api_key = "' + 'realvalue12345678"', "credential assigned to a string literal"),
    ("PASSWORD: 'hunter2hunter2'", "credential assigned to a string literal"),
])
def test_scanner_detects_secrets(line, rule):
    f = secrets_scan.scan_text(line, "x.py")
    assert len(f) == 1 and f[0].rule == rule and f[0].line == 1


@pytest.mark.parametrize("line", [
    'api_key = os.environ["LLM_API_KEY"]', 'key = os.getenv("LLM_API_KEY", "")', 'api_key = "changeme"', 'api_key = "your_api_key_here"',
    'token = "<token>"', 'password = "${DB_PASSWORD}"', 'api_key = "short"', "LLM_MAX_TOKENS=900",
    f'x = "{FAKE_KEY}"  # pragma: allowlist secret', "just a normal line of code",
])
def test_scanner_ignores_safe_lines(line):
    assert secrets_scan.scan_text(line) == []


def test_scanner_never_echoes_full_secret():
    f = secrets_scan.scan_text(f'k = "{FAKE_KEY}"')[0]
    assert FAKE_KEY not in f.snippet and "****" in f.snippet


def test_scan_project_skips_env_tests_and_data(tmp_path):
    (tmp_path / "src").mkdir(); (tmp_path / "tests").mkdir(); (tmp_path / "data").mkdir()
    (tmp_path / "src" / "a.py").write_text(f'k = "{FAKE_KEY}"\n')
    (tmp_path / "tests" / "t.py").write_text(f'k = "{FAKE_KEY}"\n')
    (tmp_path / "data" / "d.txt").write_text(f'k = "{FAKE_KEY}"\n')
    (tmp_path / ".env").write_text(f"LLM_API_KEY={FAKE_KEY}\n")
    findings, n = secrets_scan.scan_project(tmp_path)
    assert [f.file for f in findings] == ["src/a.py"] and n == 1                 # .env contents never read


def test_gitignore_and_env_example_hygiene(tmp_path):
    assert not secrets_scan.gitignore_covers(tmp_path)
    (tmp_path / ".gitignore").write_text("__pycache__/\n.env\n")
    assert secrets_scan.gitignore_covers(tmp_path)
    (tmp_path / ".env.example").write_text("LLM_API_KEY=\nLLM_MAX_TOKENS=900\n# c\nDB_PASSWORD=hunter2\n")
    ok, bad = secrets_scan.env_example_is_clean(tmp_path)
    assert not ok and bad == ["DB_PASSWORD"]
    assert secrets_scan.env_example_is_clean(tmp_path / "nowhere") == (False, [".env.example missing"])


def test_env_status_reports_set_or_not_but_never_values(tmp_path):
    (tmp_path / ".env.example").write_text("# provider\nLLM_PROVIDER=anthropic\n# key\nLLM_API_KEY=\nLLM_MAX_TOKENS=900\n")
    rows = secrets_scan.env_variable_status(tmp_path, {"LLM_API_KEY": "topsecretvalue", "LLM_PROVIDER": "anthropic"})
    by = {r["variable"]: r for r in rows}
    assert by["LLM_API_KEY"]["secret"] and by["LLM_API_KEY"]["status"] == "set" and not by["LLM_MAX_TOKENS"]["secret"]
    assert by["LLM_MAX_TOKENS"]["status"] == "not set" and by["LLM_API_KEY"]["purpose"] == "key"
    assert "topsecretvalue" not in json.dumps(rows)


def test_config_secret_fields():
    assert secrets_scan.config_secret_fields({"a": {"api_key": "abc", "max_tokens": 5}, "l": [{"password": "x"}]}) == ["a.api_key", "l[0].password"]
    assert secrets_scan.config_secret_fields({"api_key": "", "iqr_multiplier": 3}) == []


# ---- dependencies -----------------------------------------------------------------------------
def test_requirement_checks(tmp_path):
    req = tmp_path / "r.txt"
    req.write_text("pandas>=1.0\nnumpy>=99.0,<100\nnot-a-real-package-xyz>=1.0\nPyYAML\n# c\n")
    rows = {r["package"]: r for r in dependencies.check_requirements(req)}
    assert rows["pandas"]["ok"] and "no upper bound" in rows["pandas"]["notes"]
    assert not rows["numpy"]["ok"] and "violates" in rows["numpy"]["notes"]
    assert not rows["not-a-real-package-xyz"]["ok"] and "not installed" in rows["not-a-real-package-xyz"]["notes"]
    assert not rows["PyYAML"]["ok"] and "no version constraint" in rows["PyYAML"]["notes"]


def test_python_support_eol_logic():
    assert dependencies.python_support(date(2026, 9, 26), (3, 9, 6))["status"] == "end-of-life"
    assert dependencies.python_support(date(2026, 9, 26), (3, 12, 1))["status"] == "supported"
    assert dependencies.python_support(date(2026, 9, 26), (3, 99, 0))["status"] == "unknown"


def test_pip_audit_result_parsing(monkeypatch):
    out = json.dumps({"dependencies": [{"name": "a", "version": "1", "vulns": [{"id": "X-1", "fix_versions": ["2"]}]}, {"name": "b", "version": "1", "vulns": []}]})
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, out, ""))
    r = dependencies.run_pip_audit("x.lock")
    assert r["status"] == "vulnerabilities" and r["packages_audited"] == 2 and r["vulnerabilities"][0] == {"package": "a", "version": "1", "id": "X-1", "fix": "2"}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps({"dependencies": []}), ""))
    assert dependencies.run_pip_audit("x.lock")["status"] == "clean"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "No module named pip_audit"))
    assert dependencies.run_pip_audit("x.lock")["status"] == "unavailable"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, "garbage", "boom"))
    assert dependencies.run_pip_audit("x.lock")["status"] == "error"


# ---- secure logging ---------------------------------------------------------------------------
def test_log_event_hashes_free_text_and_redacts_secrets(tmp_path):
    lg = get_security_logger(tmp_path / "s.log", "pg.test1")
    rec = log_event(lg, "ai_request", user="rita", question="patient Jane Doe has batch problem", token="tok1234567890abcdef",  # pragma: allowlist secret
                    note=f"used {FAKE_KEY}", batch_id="B00042")
    for h in lg.handlers:
        h.flush()
    text = (tmp_path / "s.log").read_text()
    assert "Jane Doe" not in text and FAKE_KEY not in text and "tok1234567890abcdef" not in text
    assert rec["question_len"] == len("patient Jane Doe has batch problem") and len(rec["question_sha256_8"]) == 8 and rec["batch_id"] == "B00042"
    assert json.loads(text.splitlines()[0])["event"] == "ai_request"


def test_redacting_filter_scrubs_plain_messages(tmp_path):
    import logging
    lg = logging.getLogger("pg.test2")
    lg.handlers = []
    h = logging.FileHandler(tmp_path / "p.log"); h.addFilter(RedactingFilter()); lg.addHandler(h); lg.setLevel(logging.INFO)
    lg.info("failed with %s", FAKE_KEY)
    h.flush()
    assert FAKE_KEY not in (tmp_path / "p.log").read_text()


def test_logger_setup_is_idempotent(tmp_path):
    a, b = get_security_logger(tmp_path / "i.log", "pg.test3"), get_security_logger(tmp_path / "i.log", "pg.test3")
    assert a is b and len(a.handlers) == 1


# ---- executable controls ----------------------------------------------------------------------
def test_code_controls_pass_on_this_repository():
    cs = {c.id: c for c in controls.run_controls()}
    for cid in [f"SEC-{i:02d}" for i in range(1, 12)]:
        assert cs[cid].status == "pass", (cid, cs[cid].evidence)
    assert cs["SEC-12"].status == "pass"
    assert cs["SEC-15"].status == "info"                                    # no audit supplied -> "not run"
    s = controls.summary(list(cs.values()))
    assert s["total"] == 15 and s["pass"] >= 12


def test_controls_fail_on_an_insecure_project(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.yaml").write_text("llm:\n  api_key: realvalue12345678\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "bad.py").write_text(f'KEY = "{FAKE_KEY}"\n')
    (tmp_path / ".env.example").write_text("LLM_API_KEY=" + "abcdef123456\n")
    (tmp_path / "requirements.txt").write_text("pandas\n")
    cs = {c.id: c for c in controls.run_controls(tmp_path)}
    assert all(cs[i].status == "fail" for i in ("SEC-01", "SEC-02", "SEC-03", "SEC-04", "SEC-12"))
    assert FAKE_KEY not in cs["SEC-01"].evidence and cs["SEC-01"].remediation


def test_pip_audit_control_states():
    base = {c.id: c for c in controls.run_controls(pip_audit={"status": "clean", "packages_audited": 5})}
    assert base["SEC-15"].status == "pass"
    vul = {"status": "vulnerabilities", "vulnerabilities": [{"package": "x", "version": "1", "id": "A", "fix": "2"}]}
    assert {c.id: c for c in controls.run_controls(pip_audit=vul)}["SEC-15"].status == "warn"
    assert {c.id: c for c in controls.run_controls(pip_audit={"status": "unavailable", "detail": "n/a"})}["SEC-15"].status == "warn"


def test_pip_audit_results_roundtrip(tmp_path):
    assert controls.load_pip_audit(tmp_path) is None
    controls.save_pip_audit({"status": "clean", "packages_audited": 3}, tmp_path)
    assert controls.load_pip_audit(tmp_path)["status"] == "clean" and "scanned_at" in controls.load_pip_audit(tmp_path)


# ---- role concept -----------------------------------------------------------------------------
def test_admin_role_separation_of_duties(tmp_path):
    from src.review.store import ReviewStore
    assert roles.can("compliance_admin", "run_security_scan") and roles.can("compliance_admin", "export_governance_report")
    assert not roles.can("compliance_admin", "ask_assistant") and not roles.can("compliance_admin", "review_ai_finding")
    assert not roles.can("quality_analyst", "run_security_scan") and all(roles.can(r, "view_governance") for r in roles.ROLES)
    from src.assistant import llm as L
    from src.assistant.pipeline import Investigation
    with pytest.raises(roles.PermissionDenied):
        ReviewStore(tmp_path / "x.db").save_run(Investigation("q", "B1", True), "admin", "compliance_admin")
