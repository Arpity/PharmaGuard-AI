"""The Dockerfile, .dockerignore, compose file, requirements and CI workflow are configuration-as-code: test them."""
import re
from pathlib import Path

import pytest
import yaml

from config import PROJECT_ROOT as ROOT

DOCKERFILE = (ROOT / "Dockerfile").read_text()
SECRETISH = re.compile(r"(?i)(api[_-]?key|secret|password|token)")


def instructions(text=DOCKERFILE):
    """Dockerfile instructions with line continuations joined, comments removed."""
    joined = re.sub(r"\\\n", " ", text)
    return [l.strip() for l in joined.splitlines() if l.strip() and not l.strip().startswith("#")]


# ---- Dockerfile -------------------------------------------------------------------------------
def test_dockerfile_is_multistage_slim_and_pinned():
    ins = instructions()
    froms = [i for i in ins if i.startswith("FROM")]
    assert len(froms) == 2 and all("-slim" in f and ":latest" not in f for f in froms)
    assert "ARG PYTHON_VERSION=3.12" in ins and any("AS runtime" in f for f in froms)


def test_dockerfile_runs_as_non_root_and_exposes_the_streamlit_port():
    ins = instructions()
    assert "USER app" in ins and "EXPOSE 8501" in ins
    assert any(i.startswith("RUN groupadd") and "10001" in i for i in ins)
    assert ins.index("USER app") > max(n for n, i in enumerate(ins) if i.startswith("RUN"))          # no RUN after dropping privileges


def test_dockerfile_starts_streamlit_correctly_with_healthcheck():
    ins = instructions()
    assert 'CMD ["streamlit", "run", "app/streamlit_app.py"]' in ins
    hc = next(i for i in ins if i.startswith("HEALTHCHECK"))
    assert "_stcore/health" in hc
    env = " ".join(i for i in ins if i.startswith("ENV"))
    for kv in ("STREAMLIT_SERVER_HEADLESS=true", "STREAMLIT_SERVER_ADDRESS=0.0.0.0", "STREAMLIT_SERVER_PORT=8501"):
        assert kv in env
    assert (ROOT / "app" / "streamlit_app.py").exists()


def test_dockerfile_bakes_in_no_secrets():
    ins = instructions()
    for i in ins:
        if i.startswith(("ENV", "ARG")):
            assert not SECRETISH.search(i.split("=")[0]), i                       # no secret-named ENV / ARG
        assert ".env " not in i + " " or ".env.example" in i, i                    # never COPY .env
    assert not any(re.search(r"COPY .*\.env(\s|$)", i) for i in ins)
    assert not any(re.search(r"(sk-[A-Za-z0-9]{20}|AKIA[0-9A-Z]{16})", i) for i in ins)


def test_dockerfile_installs_only_runtime_requirements_reproducibly():
    ins = instructions()
    assert any(i.startswith("RUN pip install") and "-c requirements.lock" in i and "-r requirements.txt" in i for i in ins)
    assert not any("requirements-dev" in i for i in ins) and not any("pytest" in i for i in ins)
    assert all(i.startswith("COPY") is False or "--from=builder" in i or "--chown=app:app" in i or "requirements" in i for i in ins)


def test_every_copied_path_exists():
    for i in instructions():
        m = re.match(r"COPY (?:--\S+ )*(.+) (\S+)$", i)
        if m and "--from" not in i:
            for src in m.group(1).split():
                assert (ROOT / src).exists(), f"COPY source missing: {src}"


# ---- .dockerignore ----------------------------------------------------------------------------
def _ignore_regex(pattern: str) -> re.Pattern:
    """Approximate Docker's .dockerignore matching (Go filepath.Match + ** support)."""
    p = pattern.strip().lstrip("/")
    out, i = "", 0
    while i < len(p):
        if p[i:i + 3] == "**/":
            out, i = out + "(?:.*/)?", i + 3
        elif p[i:i + 2] == "**":
            out, i = out + ".*", i + 2
        elif p[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif p[i] == "[" and "]" in p[i:]:
            j = p.index("]", i)
            out, i = out + p[i:j + 1], j + 1
        elif p[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(p[i]), i + 1
    return re.compile(f"^{out}(?:/.*)?$")


def ignored(path: str, lines) -> bool:
    state = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        neg = line.startswith("!")
        if _ignore_regex(line.lstrip("!")).match(path):
            state = not neg
    return state


DOCKERIGNORE = (ROOT / ".dockerignore").read_text().splitlines()


@pytest.mark.parametrize("path", [".env", ".env.local", ".env.production", ".git/config", ".venv/bin/python", "tests/conftest.py",
                                  "data/app/pharmaguard_review.db", "logs/app_errors.log", "keys/server.pem", "id.key", "config/secrets.json",
                                  ".github/workflows/ci.yml", ".coverage", "src/__pycache__/x.pyc", "requirements-dev.txt"])
def test_dockerignore_keeps_secrets_and_dev_files_out(path):
    assert ignored(path, DOCKERIGNORE), path


def test_dockerignore_allows_env_example_and_never_hides_runtime_files():
    assert not ignored(".env.example", DOCKERIGNORE)
    needed = []
    for d in ("app", "config", "src", "scripts", "knowledge", ".streamlit"):
        needed += [str(p.relative_to(ROOT)) for p in (ROOT / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for d in ("data/raw", "data/processed", "data/evaluation"):
        needed += [str(p.relative_to(ROOT)) for p in (ROOT / d).rglob("*") if p.is_file()]
    needed += ["requirements.txt", "requirements.lock", ".env.example", ".gitignore", "README.md", "app/streamlit_app.py"]
    hidden = [n for n in needed if ignored(n, DOCKERIGNORE)]
    assert not hidden, f"needed at run time but excluded by .dockerignore: {hidden}"     # regression: '**/secrets*' hid src/security/secrets_scan.py


# ---- requirements -----------------------------------------------------------------------------
def _reqs(name):
    return [l.split("#")[0].strip() for l in (ROOT / name).read_text().splitlines() if l.split("#")[0].strip()]


def test_runtime_requirements_are_minimal_and_bounded():
    reqs = _reqs("requirements.txt")
    names = {re.split(r"[<>=!~\[ ]", r)[0].lower() for r in reqs}
    assert {"streamlit", "pandas", "numpy", "plotly", "pyyaml", "python-dotenv", "packaging"} <= names
    assert not names & {"pytest", "pytest-cov", "ruff", "bandit", "pip-audit", "scikit-learn", "joblib"}
    assert all("<" in r for r in reqs), "every runtime requirement needs an upper bound"


def test_dev_requirements_layer_on_runtime_ones():
    dev = _reqs("requirements-dev.txt")
    assert "-r requirements.txt" in dev and {"pytest", "ruff", "bandit", "pip-audit"} <= {re.split(r"[<>=]", r)[0] for r in dev if not r.startswith("-")}


def test_lockfile_pins_every_runtime_dependency_and_no_dev_tools():
    lock = [l for l in (ROOT / "requirements.lock").read_text().splitlines() if l and not l.startswith("#")]
    assert lock and all(re.match(r"^[A-Za-z0-9_.\-]+==[0-9][^ ]*$", l) for l in lock), [l for l in lock if "==" not in l]
    names = {l.split("==")[0].lower().replace("_", "-") for l in lock}
    for r in _reqs("requirements.txt"):
        assert re.split(r"[<>=!~\[ ]", r)[0].lower().replace("_", "-") in names
    assert not names & {"pytest", "pytest-cov", "ruff", "bandit", "pip-audit"}


# ---- compose ----------------------------------------------------------------------------------
def test_compose_file_is_hardened_and_secret_free():
    c = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    svc = c["services"]["pharmaguard"]
    assert "8501:8501" in svc["ports"] and svc["read_only"] is True and svc["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in svc["security_opt"] and svc["env_file"][0]["required"] is False
    assert "environment" not in svc or not any(SECRETISH.search(str(k)) and v for k, v in svc["environment"].items())
    assert {"pharmaguard_data", "pharmaguard_logs"} <= set(c["volumes"])


# ---- CI workflow ------------------------------------------------------------------------------
WF = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
WF_TEXT = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
ON = WF.get("on", WF.get(True))


def test_workflow_triggers_and_least_privilege():
    assert "push" in ON and "pull_request" in ON and "workflow_dispatch" in ON
    assert WF["permissions"] == {"contents": "read"}
    assert WF["concurrency"]["cancel-in-progress"] is True


def test_pipeline_stages_run_in_the_required_order():
    jobs = WF["jobs"]
    chain = ["quality", "unit-tests", "integration-tests", "security", "ai-evaluation-gate", "docker", "deploy"]
    assert list(jobs) == chain
    for prev, cur in zip(chain, chain[1:]):
        assert jobs[cur]["needs"] == prev, f"{cur} must need {prev}"


def test_required_jobs_cannot_be_skipped_or_softened():
    for name, job in WF["jobs"].items():
        if name == "deploy":
            continue
        assert "if" not in job and job.get("continue-on-error") is not True, name
        for step in job["steps"]:
            assert step.get("continue-on-error") is not True, (name, step.get("name"))


def test_stage_contents():
    def runs(job):
        return "\n".join(str(s.get("run", "")) + str(s.get("with", "")) + str(s.get("uses", "")) for s in WF["jobs"][job]["steps"])
    assert "ruff check" in runs("quality") and "bandit" in runs("quality")
    assert 'not integration and not e2e and not ui' in runs("unit-tests") and "--cov-fail-under=90" in runs("unit-tests")
    assert 'integration or e2e or ui' in runs("integration-tests")
    assert "run_security_checks.py" in runs("security") and "pip-audit -r requirements.lock" in runs("security")
    assert "scripts/evaluation_gate.py --run" in runs("ai-evaluation-gate")
    d = runs("docker")
    assert "hadolint" in d and "--read-only" in d and "_stcore/health" in d and "docker save" in d and "id -u" in d


def test_gate_thresholds_are_configurable_through_repository_variables():
    env = WF["env"]
    gate = yaml.safe_load((ROOT / "config" / "evaluation_gate.yaml").read_text())["thresholds"]
    gate_vars = {k for k in env if k.startswith("EVAL_GATE_")}
    assert "EVAL_GATE_MODE" in gate_vars and all("vars." in str(env[k]) for k in gate_vars)
    for k in gate_vars - {"EVAL_GATE_MODE"}:
        metric = re.match(r"EVAL_GATE_(.+)_(MIN|MAX)$", k).group(1).lower()
        assert metric in gate, f"{k} does not match a threshold in config/evaluation_gate.yaml"


def test_no_credentials_assumed_and_deployment_and_publish_are_opt_in():
    secrets_used = set(re.findall(r"secrets\.([A-Za-z0-9_]+)", WF_TEXT))
    assert secrets_used <= {"GITHUB_TOKEN", "DEPLOY_HOST", "DEPLOY_SSH_KEY", "LLM_API_KEY"}
    deploy = WF["jobs"]["deploy"]
    assert "vars.DEPLOY_ENABLED == 'true'" in deploy["if"] and "refs/heads/main" in deploy["if"] and deploy["environment"] == "production"
    publish = next(s for s in WF["jobs"]["docker"]["steps"] if "Publish" in s.get("name", ""))
    assert "vars.PUBLISH_IMAGE == 'true'" in publish["if"]
    assert not re.search(r"(sk-[A-Za-z0-9]{20}|AKIA[0-9A-Z]{16}|password:\s*\S+)", WF_TEXT)
    assert WF["jobs"]["docker"]["permissions"]["packages"] == "write" and "permissions" not in WF["jobs"]["quality"]


def test_actions_are_version_pinned():
    for use in re.findall(r"uses:\s*(\S+)", WF_TEXT):
        assert re.search(r"@v\d", use), f"unpinned action: {use}"
