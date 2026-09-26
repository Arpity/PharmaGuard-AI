# PharmaGuard AI - Production Readiness Report

Independent end-to-end review of the repository (commit history through the audit fixes). Method: fresh clone from GitHub, follow the README,
regenerate every artifact, run the test suite / linters / scanners, replay the CI stages in a clean Python 3.12 container, build and run the
Docker image, probe guardrails and cleaning with inputs the tests did not cover, read the code for each lifecycle stage.

**Verdict: a well-tested, reproducible *demonstration prototype*. Not ready for production or GxP use.** The blocking gaps are outside the
code that exists: no real authentication, no validation against real data, no live-LLM verification, no deployment target.

Status legend: **Implemented** = works and is verified here for the demo scope; **Partially Implemented** = core exists with material gaps;
**Not Implemented** = absent. "Verified" means I ran it in this environment (macOS, Python 3.9.6 local; Python 3.12 in Docker / CI replay).

## 1. Production readiness table

| Component | Implementation status | Test status | Risk / gap | Recommended action |
|---|---|---|---|---|
| **Problem definition** | Partially Implemented - use case, intended users, out-of-scope in `config/governance.yaml`; success criteria expressed as evaluation targets and SLOs | Governance metadata tests pass | Owners are role titles; no stakeholder-signed requirements or acceptance criteria | Assign named owners; have QA sign acceptance criteria |
| **Data collection** | Partially Implemented - deterministic synthetic generator only; raw folder protected by a write guard | Generation, reproducibility and guard tests pass; regeneration in a fresh clone reproduced raw and cleaned data byte-for-byte | No real source connectors, contracts or ingestion; synthetic data proves nothing about real distributions | Define real source systems, extraction contract and consent / provenance |
| **Data governance** | Partially Implemented - classification, data sources, personal-data flags, schema baseline, cleaning audit log | Tests pass | No data dictionary, per-dataset owner, retention / deletion policy (runtime DBs hold user names); dataset version not stored with each AI run | Add data dictionary and retention policy; record dataset and knowledge-base fingerprints per run |
| **Data quality** | Implemented - 6-dimension score, per-issue reports, page | Unit + page tests pass; fixed a false positive on typed values earlier | Thresholds untuned for real data | Calibrate thresholds on real data |
| **Data cleaning** | Implemented - controlled rules, audit log, raw never written; **audit fix:** schema change / empty input now fails with a clear error (previously cryptic pandas errors) | Tests pass incl. new schema-error test; replay reproduces audit log (timestamps differ only) | Imputation and duplicate-resolution rules are heuristics needing QA approval | QA review and sign-off of cleaning rules |
| **Analytics** | Implemented - deterministic KPIs, charts, rule-based insights | Exact-value tests pass | None material | - |
| **AI / LLM** | Partially Implemented - Anthropic and OpenAI-compatible adapters, env-based config, demo fallback, base-URL scheme check, token capture | Adapters tested **only with mocked transports**; a real provider was never called (no key) | No retry / backoff / rate-limit / circuit-breaker; model not pinned unless `LLM_MODEL` is set; live behaviour and cost unverified | Run the evaluation with a real key; add retries and a circuit breaker; pin the model |
| **RAG** | Partially Implemented - TF-IDF over 4 synthetic documents (21 chunks), cited sources, poisoned-chunk drop | Retrieval metrics on the golden set (F1 0.92, recall 1.0) pass | Lexical retrieval, tiny synthetic corpus, no document versioning or approval workflow | Load approved real SOPs with versioning; consider embeddings + reranking |
| **Guardrails** | Partially Implemented - input validation, injection and disposition detection, output validation, grounding, fail-closed replacement. **Audit fixes:** synonyms (ship, accept, good to go, override), leetspeak / homoglyph / paraphrase injections, bare-verdict answers | 321 tests incl. new regression tests; golden gate 100% on injection, disposition and adversarial-output cases | Regex-based: other languages and novel paraphrases still evade the input guard (documented). Mitigated by defence in depth - the AI has no disposition capability and output validation blocks decision claims | Red-team with real attackers; add a model-based classifier or provider moderation |
| **Human in the loop** | Partially Implemented - approve / reject / request-more-analysis, mandatory comments, four-eyes, append-only tables, permission enforced in the store | Tests pass | Identity is a name + role dropdown: **no authentication**, four-eyes is spoofable; SQLite triggers deter but are not a validated audit trail | SSO / IAM; move to a controlled database with signed audit records |
| **Evaluation** | Partially Implemented - 68-case golden set with independent oracle, 8 metrics, sensitivity tests, dashboard, CI gate | Gate passes; deliberately broken components turn the metrics red | Synthetic and small; answer relevance is a lexical proxy; **live-LLM wording never evaluated**; no human-graded set | Add human-reviewed cases from real usage; evaluate the live model in CI (secret-gated) |
| **Software testing** | Implemented - 321 tests, ~98% coverage of `src`; unit / integration / UI / e2e split | 321 pass locally (Py 3.9) and in a clean Python 3.12 replay (307 unit + 14 integration) | UI tested with Streamlit AppTest, **not in a real browser**; no load / performance / concurrency tests | Add a browser smoke test and a basic load test |
| **AI governance** | Partially Implemented - system card, risk register (11 risks), limitations, approvals table (all *Pending*, correctly), drift detection of prompt / knowledge / config / golden set, downloadable report | Tests pass | No formal validation (e.g. CSV / GxP), no signed approvals, owners are roles | Complete validation and approvals before any real use |
| **Security** | Partially Implemented - no secrets in tree or git history (verified), secret scan, RBAC concept, secure logging, bandit clean, `pip-audit` clean, Trivy: 0 HIGH/CRITICAL in Python packages, `LLM_BASE_URL` scheme validation | 15 executable security controls: 14 pass in CI environment (SEC-14 fails locally only because Python 3.9 is end-of-life) | Real authN/Z absent; no TLS / reverse-proxy config; no rate limiting; DBs unencrypted at rest; **44 HIGH Debian base-image CVEs with no fix available yet**; local Python 3.9 is EOL | Front with SSO + TLS proxy; rate-limit; rebuild on patched base images regularly; use Python >= 3.12 locally |
| **Observability** | Implemented (local) - Run ID + OTel-shaped trace per request, structured privacy-safe logs, token capture, dashboard, OTLP JSON | Tests pass | Local SQLite only; OTLP export never sent to a real collector; SQLite connections now close properly (audit fix) | Wire a real OTel collector before production |
| **Traceability** | Partially Implemented - Run ID, trace id, user / role, model + version, prompt version + hash, retrieved chunks, guardrail result, final status | Tests pass | **Per-run record lacks dataset version and knowledge-base version**; model version is the alias unless pinned | Store dataset / KB fingerprints and pinned model per run |
| **Docker** | Implemented - multi-stage, slim, non-root numeric uid, no secrets, healthcheck, read-only friendly, compose. **Audit fixes:** numeric `USER`, exec-form healthcheck, builder `WORKDIR`, configurable compose port | **Verified:** builds; hadolint clean; healthy under `--read-only --cap-drop ALL`; compose up healthy; earlier run verified all 9 pages load in the container; databases written to mounted volumes | Image ~0.9 GB; runs single instance with SQLite state; no image signing / SBOM | Add SBOM + signing; move state to a managed DB for multi-replica |
| **CI/CD** | Implemented - 8 staged jobs with real gates, configurable AI evaluation gate, artifacts, opt-in publish and deploy. **Audit adds:** weekly scheduled run, Dependabot | **Verified on GitHub Actions:** the audited commit (980f511) passed all seven required stages incl. Docker smoke test; actionlint clean; local Python 3.12 replay also green | Actions pinned to major versions, not commit SHAs; no image scan / signing in CI; Deploy stage is skipped by design | Pin actions by SHA; add Trivy + signing steps |
| **Deployment readiness** | Not Implemented for real deployment - the deploy job is an opt-in skeleton that fails loudly until a target is written | Not applicable | No Kubernetes / cloud manifests, TLS, authentication, backup / restore, migrations or runbooks | Choose a target; add manifests, secrets management, backups, runbooks |
| **Production monitoring** | Partially Implemented - 16 SLO indicators, incidents feed, health probes, drift + schema checks, security / cost / business signals, Prometheus exporter, alert rules, Grafana + OTel configs, simulation | 32 tests incl. exposition-format checks and alert-rule / metric-name consistency; exporter HTTP served and queried | **Prometheus / Grafana / Alertmanager were never run against it**; availability is a proxy (no external probes); alert routing absent | Deploy Prometheus + Grafana against the exporter; add external synthetic probes and paging |
| **Continuous improvement** | Partially Implemented - CI gate, weekly scheduled CI, Dependabot, evaluation dashboard, change history | Config tests pass | Human rejections / comments are **not fed back** into the golden set, knowledge base or prompt; no incident post-mortem process | Add a feedback loop: reviewed failures become golden cases |
| **Documentation / README** | Implemented after fixes - README previously said "Step 1 of N" and listed stale details; rewritten with setup, structure, page map; `.env.example` dead variables removed and missing one added | Commands below verified | README still layered by feature; some numbers (e.g. test counts) will drift | Keep counts out of prose |

## 2. Issues found and fixed in this review
1. Stray `actionlint_py-1.7.12.25.tar.gz` committed at repo root - removed and ignored.
2. Cleaning pipeline crashed with cryptic pandas errors on a missing column or empty input (the schema-change scenario the monitoring page advertises) - now raises a clear `ValueError`; tests added.
3. Input guard missed common disposition synonyms (ship, accept, "pass this batch", "good to go", override status) and obfuscated injections (leetspeak, look-alike letters, "respond only with APPROVED", prompt-extraction phrasing) - patterns and folding added; false-positive probes (e.g. "pass rate", "shipment lots") verified; regression tests added.
4. Output guard did not block a bare verdict answer ("APPROVED") - added.
5. SQLite connections used `with sqlite3.connect()`, which never closes them - replaced with a closing connection; 1,200 store calls showed no descriptor growth.
6. Dockerfile: non-numeric `USER` (breaks Kubernetes `runAsNonRoot`), shell-form healthcheck, missing builder `WORKDIR` - fixed; hadolint now exits 0.
7. `docker-compose.yml` hard-coded host port 8501 - it collided with a process on this machine - now `PHARMAGUARD_PORT`; compose retested healthy.
8. `.env.example` documented two unused variables (`LOG_LEVEL`, `PROCESSED_DATA_DIR`) and omitted `PHARMAGUARD_EVAL_PATH` - corrected.
9. README opening and the app home page were stale (Step 1 of N, missing pages) - rewritten.
10. Governance baseline recorded version 0.8.0 - re-registered (fingerprints were unchanged); version bumped to 0.10.1 with change history.
11. Added weekly scheduled CI and Dependabot. Side effect observed: Dependabot immediately opened 10 pull requests (including major bumps such as pandas 3, which the CI gate correctly failed). The config now groups minor / patch updates and ignores major versions, and CI runs on `main` + pull requests only (no duplicate push + PR runs). The 10 existing PRs were left open for the owner to close or merge.

## 3. What was and was not verified
**Verified in this environment:** fresh-clone install + full test suite; deterministic regeneration of data, cleaned data, golden set and evaluation
(byte-identical apart from timestamps / latency); imports covered by `requirements.txt` (AST scan); `pip check`; ruff, bandit, actionlint, hadolint;
`pip-audit` on the lockfile (no known vulnerabilities); Trivy image scan; a real GitHub Actions run of the audited commit (all required stages green); no secrets in tracked files or git history; Docker build and hardened run;
compose up; local `streamlit run` health; every CI stage replayed in a clean Python 3.12 container.

**Not verified:** any live LLM call; a browser rendering the pages (AppTest and server health only); Prometheus / Grafana / OTel collector receiving data;
Kubernetes or any cloud deployment; behaviour under load or many concurrent users; the scheduled weekly workflow (has not fired yet).

## 4. Commands (all verified as written unless noted)

**Set up the environment** (Python 3.10+; 3.12 recommended)
```bash
git clone https://github.com/Arpity/PharmaGuard-AI.git && cd PharmaGuard-AI
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt      # runtime + test / lint / security tools
cp .env.example .env                     # optional; leave LLM_API_KEY blank for demo mode
```
**Run tests**
```bash
pytest                                                                   # everything
pytest --cov=src --cov=config --cov-report=term                          # with coverage
pytest -m "not integration and not e2e and not ui" --cov=src --cov-fail-under=90   # CI unit stage
pytest -m "integration or e2e or ui"                                     # CI integration stage
ruff check . && bandit -q -r src app scripts config -ll                  # lint + static security
python scripts/run_security_checks.py && python scripts/evaluation_gate.py --run   # security controls + AI gate
```
**Start locally**
```bash
streamlit run app/streamlit_app.py       # http://localhost:8501  (add --server.port 8600 if 8501 is busy)
```
**Build the Docker image**
```bash
docker build -t pharmaguard-ai:local .
```
**Run the Docker container**
```bash
docker run --rm -p 8501:8501 pharmaguard-ai:local                        # demo mode

docker run -d --name pharmaguard -p 8501:8501 --env-file .env \
  --read-only --tmpfs /tmp --tmpfs /home/app/.streamlit --cap-drop ALL --security-opt no-new-privileges:true \
  -v pharmaguard_data:/app/data/app -v pharmaguard_logs:/app/logs pharmaguard-ai:local     # hardened, persistent, live LLM if .env has a key

PHARMAGUARD_PORT=8600 docker compose up --build                          # compose alternative
curl http://localhost:8501/_stcore/health                                # -> ok
```
