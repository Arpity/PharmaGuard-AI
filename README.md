# PharmaGuard AI

Pharmaceutical batch-quality analytics with a governed AI investigation assistant (Python + Streamlit).
Everything runs on **synthetic data**; without an LLM key the assistant runs in a deterministic **demo mode**.

> It supports investigations - it never approves, rejects, releases or dispositions a batch. See *AI guardrails* below.
> Status: demonstration prototype, **not approved for production or GxP use** (see `docs/PRODUCTION_READINESS_REPORT.md`).

## Quick start
Requires Python 3.10+ (3.12 is what CI and Docker use; 3.9 works locally but is end-of-life) and, optionally, Docker.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt          # runtime + test/lint/security tools  (use requirements.txt for runtime only)
cp .env.example .env                         # optional: set LLM_API_KEY to enable a live LLM (unset = demo mode)

pytest                                       # 296+ tests (unit, integration, Streamlit pages, end-to-end)
streamlit run app/streamlit_app.py           # http://localhost:8501
```
The repository already contains the generated data (`data/raw`, `data/processed`), the golden evaluation set and the latest
evaluation report, so the app and tests work immediately. To rebuild everything from scratch (all steps are deterministic):
```bash
python scripts/generate_data.py && python scripts/run_cleaning.py
python scripts/build_golden.py && python scripts/run_evaluation.py && python scripts/evaluation_gate.py
python scripts/register_baseline.py && python scripts/register_data_baseline.py   # governance + schema baselines
```
Docker and CI/CD instructions are in *Docker & CI/CD* below.

## What is in the app
| Page | Purpose |
|---|---|
| Data Quality | Score the raw data, drill into issues, run the cleaning pipeline, inspect the audit log |
| Analytics Dashboard | KPIs, charts, filters and rule-based insights on the cleaned data |
| Batch Investigation | Ask about a batch: deterministic analysis + RAG over QA procedures + optional LLM, with sources and guardrails |
| QA Review | Humans approve / reject / request more analysis on AI findings (append-only, four-eyes) |
| Evaluation Dashboard | Golden-set metrics vs targets |
| AI Governance | System card, risks, approvals, live security controls, downloadable governance report |
| Observability | Per-request traces, latency, guardrail and token metrics |
| Production Monitoring | SLO status for application, AI, data, security, cost and business; incidents; simulation |

Roles (demo dropdown, not authentication): Viewer, Quality Analyst, QA Reviewer, Compliance / IT Admin.

## Structure
| Path | Purpose |
|---|---|
| `app/` | Streamlit app (`streamlit_app.py` + `pages/`) |
| `src/` | Library code: `data_generation`, `data_quality`, `cleaning`, `analytics`, `assistant`, `guardrails`, `review`, `evaluation`, `governance`, `security`, `observability`, `monitoring` |
| `config/` | `config.yaml`, `evaluation_gate.yaml`, `monitoring.yaml`, `governance.yaml` and baselines |
| `data/` | `raw/` (never modified), `processed/`, `evaluation/` (golden set); `data/app/` holds runtime databases (git-ignored) |
| `knowledge/` | Synthetic QA procedure documents for RAG |
| `scripts/` | CLI entry points (generate, clean, evaluate, gate, security checks, exporter, publish) |
| `tests/` | pytest suite |
| `docs/`, `monitoring/` | Observability integration guide; Prometheus / Grafana / OpenTelemetry assets |
| `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` | Container image and CI/CD |

## Dataset (3,000 rows x 18 columns)
Batch_ID, Product_Name, Dosage_Form, Manufacturing_Date, Plant, Batch_Size_Kg, Temperature, Humidity,
pH, Dissolution, Assay, Impurity_Level, Yield_Percentage, Deviation_Count, Equipment_ID,
Operator_Shift, Cycle_Time_Hrs, Quality_Status (Pass / Under Review / Fail).

Quality_Status is derived from spec limits (see `config.yaml`) before corruption, so it correlates
with the process measurements, plant and shift.

### Intentional data-quality issues
- **Missing values** - NaN, empty string, `N/A`, `NA`, `null`, `-`
- **Duplicates** - 90 duplicated rows (exact and near-duplicates sharing a Batch_ID)
- **Outliers** - extreme but plausible values per numeric column
- **Invalid ranges** - e.g. pH 15, Humidity 130%, negative impurity, Yield > 100%
- **Inconsistent categories** - `Plant_A` / `plant a` / `PLANT-A` / `A`, `Pass` / `PASS` / `Passed`, etc.
- **Type issues** - numbers stored as text (`22.4C`, `99.7 %`, `1,234.5`, `two`), mixed date formats

Exact counts per issue are in `reports/data_generation_log.json`.

## Data quality & cleaning
```bash
python scripts/run_cleaning.py                    # raw -> data/processed/
streamlit run app/streamlit_app.py                # open the "Data Quality" page
```
- `src/data_quality/` - reusable checks (missing, duplicates, outliers, invalid ranges, types,
  categories) and the weighted 0-100 Data Quality Score; `rules.py` holds shared parsers/master data.
- `src/cleaning/` - controlled pipeline (`pipeline.py`) and audit log (`audit.py`).
- Outputs in `data/processed/`: `pharma_batch_clean.csv`, `cleaning_audit_log.csv`
  (timestamp, step, issue, source_row, batch_id, column, original_value, new_value, action, reason),
  `cleaning_summary.json`.
- Valid ranges, outlier fences and score weights live in `config/config.yaml`.
- Policy highlights: outliers are flagged, not deleted; critical quality attributes are never
  imputed; invalid values become missing rather than guessed; `data/raw/` is never written.

## Analytics dashboard
`streamlit run app/streamlit_app.py` -> **Analytics Dashboard** (light theme via `.streamlit/config.toml`).
- KPI cards, Plotly charts (status, product, plant, shift heatmap, attribute distributions with spec limits,
  deviations, monthly trends, anomalies) and filters for product, plant, date and quality status.
- `src/analytics/kpis.py` computes every KPI/aggregate/insight with pandas. Failure rate = Fail / batches with a
  known status; risk = Under Review + Fail; averages ignore missing results.

## Batch Investigation AI Assistant
`streamlit run app/streamlit_app.py` -> **Batch Investigation**. Enter a Batch_ID and ask e.g. "Investigate this batch",
"Why is this batch showing risk?", "Compare it with historical batches", "Which parameters are abnormal?",
"What procedure should QA review?".

Pipeline (`src/assistant/`): question -> intent -> batch data -> **deterministic analysis** (`analysis.py`: spec checks,
peer z-scores/percentiles, similar-batch outcomes, historical failure rates, deviations) -> **knowledge retrieval**
(`knowledge.py`: TF-IDF over `knowledge/*.md`, synthetic demo SOPs) -> LLM (`pipeline.py`, `llm.py`) -> answer with
sources, evidence and pipeline trace.
- The LLM receives the calculated results and is instructed to copy numbers verbatim, never compute. A numeric
  grounding check (`grounding.py`) flags any number in the answer that is not in the supplied facts.
- LLM config via env vars (`.env.example`): `LLM_PROVIDER` (anthropic | openai-compatible | mock), `LLM_API_KEY`,
  `LLM_MODEL`, `LLM_BASE_URL`. With no key, or if the call fails, a deterministic demo answer is produced from the same
  analysis and sources. Data is sent to a provider only when a key is configured.

## AI guardrails & human-in-the-loop review
**Hard rule: the AI never approves, rejects, releases or dispositions a batch.** It has no capability to do so
(no role holds those permissions; the store has no disposition field), and requests to do so are refused with an
explanation that final disposition requires authorized Quality personnel.

Guardrails (`src/guardrails/`), applied in `src/assistant/pipeline.py`:
| Layer | What it does |
|---|---|
| Input validation | length/empty/control-character checks, Batch_ID format (`B#####`), delimiter stripping |
| Prompt-injection defence | pattern + obfuscation-aware detection blocks the request *before* any LLM call; question passed as delimited untrusted data; system prompt tells the model to ignore embedded instructions; retrieved chunks are scanned too |
| Restricted actions | approve/reject/release/disposition/quarantine/data-change requests get the fixed refusal; the user's text is never forwarded - a neutral "Investigate this batch." runs instead |
| Output validation | blocks disposition claims, prompt leakage, invented numbers, citations to sources that were not retrieved, fabricated batch IDs / SOP numbers; redacts credentials; flags causal over-claims and uncited procedures |
| Grounding / fail-closed | a blocked AI answer is replaced by the deterministic summary built from the Python analysis |
| Roles | Viewer (read), Quality Analyst (ask), QA Reviewer (ask + review AI findings) - demo dropdown, not authentication |
| Safe errors | unexpected exceptions become "reference ERR-XXXX"; redacted traceback goes to `logs/app_errors.log` |

**QA Review page**: every answer gets a Run ID and is stored in SQLite (`data/app/pharmaguard_review.db`).
Reviewers *Approve AI Finding*, *Reject AI Finding* or *Request More Analysis*; stored: Run ID, reviewer, timestamp (UTC),
decision, comments. Comments are required to reject / request analysis; four-eyes (cannot review your own run);
tables are append-only (SQLite triggers). Guardrail events are logged in the same database.

Limitations: regex-based detection can be evaded (e.g. other languages, homoglyphs), so it is one layer among several;
the demo has no real authentication; SQLite triggers deter but do not replace a validated, access-controlled audit system.

## AI evaluation & test suite
```bash
python scripts/build_golden.py        # (re)build data/evaluation/golden_cases.json from the cleaned data
python scripts/run_evaluation.py      # run all golden cases -> reports/evaluation/latest.json  (add --live for a real LLM)
streamlit run app/streamlit_app.py    # open "Evaluation Dashboard"
pytest                                # full suite;  pytest --cov=src --cov=config  for coverage;  pytest -m e2e  for the workflow
```
- **Golden dataset** (68 cases): 7 batch archetypes x 5 question types, 8 prompt injections, 8 disposition requests,
  invalid inputs, and 11 simulated adversarial model replies. Expected findings come from an independent oracle
  (`src/evaluation/oracle.py`, no imports from the code under test); expected source topics and guardrail behaviour are
  authored per case.
- **Metrics** (`src/evaluation/metrics.py`): analytical correctness, retrieval relevance (recall / precision / F1 / hit@1),
  groundedness, answer relevance, unsupported claims, guardrail success, tool execution success, latency (mean/p50/p95).
  All are deterministic; targets live in `config/config.yaml` (`evaluation.targets`).
- **The evaluation can fail**: tests break the injection defence, output validation, analysis and retrieval on purpose and
  assert the relevant metric turns red.
- Limitations: answer relevance is a lexical proxy (no LLM judge); the golden set is synthetic; demo mode scores the
  deterministic answer writer, so use `--live` to evaluate a real model's wording.

## AI governance & security
```bash
streamlit run app/streamlit_app.py            # open "AI Governance"
python scripts/run_security_checks.py --audit # run all security controls (+ pip-audit, needs network); exit 1 on any FAIL
python scripts/register_baseline.py           # record approved fingerprints after a reviewed change
```
- **Governance metadata** lives in `config/governance.yaml` (use case, owners as roles, users, data sources and
  classification, risk register, limitations, guardrails, oversight, approval status, change history). Model in use,
  evaluation results, configuration fingerprints and security-control results are computed live.
- **AI Governance page** tabs: System card, Risks & limitations, Guardrails & oversight (incl. role/permission matrix),
  Evaluation & approval, Change history (with drift detection of prompt / knowledge / config / golden set),
  Security controls, Governance report (HTML / Markdown download, permission-gated).
- **Security controls** (`src/security/`, 15 executable checks): hard-coded-credential scan, `.env` git-ignored and
  `.env.example` clean, env-var-only credentials, input-validation / output-validation / RBAC probes, secure logging
  (secrets redacted; free text logged only as length + hash), append-only audit tables, raw-data write guard, safe errors,
  dependency hygiene, lockfile, runtime end-of-life check, `pip-audit`.
- Roles now include **Compliance / IT Admin** (run scans, export reports; cannot ask the AI or review findings).
- `requirements.lock` pins the environment for reproducible installs and auditing.
- The approval status is *Not approved for production use*; approvals are recorded only by editing governance.yaml
  under change control - the page never grants approval.

## Observability & traceability
```bash
python scripts/generate_demo_traces.py 150   # synthetic traffic through the real pipeline -> data/app/observability.db
streamlit run app/streamlit_app.py           # open "Observability" (QA Reviewer / Compliance-IT Admin roles)
```
- **Every AI request gets a Run ID** (`RUN-YYYYMMDD-XXXXXX`) - including blocked, refused, invalid and failed requests - and a
  trace: root span `ai.request` plus one child span per stage / tool call (`input_validation`, `batch_lookup`,
  `deterministic_analysis`, `knowledge_retrieval`, `llm_generate` | `demo_answer_writer`, `output_validation`, `persist_run`).
- **Captured per request**: run id, trace id, timestamp, user/role, batch id, model, model version, prompt version,
  retrieved documents, tool calls (status + duration), latency, token usage (when the provider reports it), errors,
  guardrail result, final status (`success`, `success_flagged`, `degraded`, `refused`, `blocked`, `invalid_request`, `error`).
- **Structured logging**: one JSON line per request in `logs/observability.jsonl`, correlated by `run_id` / `trace_id`;
  free text is stored as length + hash, secrets redacted.
- **Dashboard**: total requests, success rate, failed requests, average (and p95) latency, guardrail blocks, retrieval
  failures, token usage + coverage, status mix, latency by stage, tool-call success, guardrail events, recent runs with a
  span waterfall, retrieved documents and per-run OTLP JSON download.
- **Definitions**: *failed* = internal error only; guardrail blocks, refusals and invalid requests are correct handling.
  A retrieval failure (error or no chunks) degrades the answer but does not fail the request.
- **OpenTelemetry-ready** (`src/observability/tracing.py`): 128-bit trace ids / 64-bit span ids, span kinds, statuses,
  events, `gen_ai.*` / `enduser.*` attribute names, W3C `traceparent` join, and `to_otlp_json()` producing an OTLP/JSON
  payload for any collector. To adopt the SDK, replace `Tracer` with an OTel tracer and the store with an OTLP exporter;
  the instrumentation calls (`start_span`, `set_attribute`, `record_exception`) keep the same shape.

## Docker & CI/CD

### Run with Docker
Requires Docker 24+ (BuildKit). The image is a multi-stage build on `python:3.12-slim`, runs as a non-root user (uid 10001),
contains **no secrets**, and installs only runtime dependencies from `requirements.txt` constrained by `requirements.lock`.

```bash
# 1. Build
docker build -t pharmaguard-ai:local .

# 2. Run in demo mode (no LLM key needed)  ->  http://localhost:8501
docker run --rm -p 8501:8501 pharmaguard-ai:local

# 3. Run with persistence (review DB, traces, logs survive restarts) and a live LLM key from your git-ignored .env
cp .env.example .env            # then edit .env: set LLM_API_KEY (and optionally LLM_MODEL)
docker run -d --name pharmaguard -p 8501:8501 --env-file .env \
  -v pharmaguard_data:/app/data/app -v pharmaguard_logs:/app/logs \
  pharmaguard-ai:local

# 4. Hardened run (immutable filesystem, no capabilities) - this is what CI smoke-tests
docker run -d --name pharmaguard -p 8501:8501 --read-only --tmpfs /tmp --tmpfs /home/app/.streamlit \
  --cap-drop ALL --security-opt no-new-privileges:true \
  -v pharmaguard_data:/app/data/app -v pharmaguard_logs:/app/logs pharmaguard-ai:local

# Or with Compose (reads .env if present; set PHARMAGUARD_PORT if 8501 is already in use)
docker compose up --build
# PHARMAGUARD_PORT=8600 docker compose up --build

# Health / logs / stop
curl http://localhost:8501/_stcore/health          # -> ok
docker logs -f pharmaguard
docker rm -f pharmaguard
```

**Environment configuration** (never bake secrets into the image; inject them at run time):

| Variable | Purpose | Default |
|---|---|---|
| `LLM_API_KEY` | Enables a live LLM. **Unset = demo mode** (no data leaves the container) | unset |
| `LLM_PROVIDER` / `LLM_MODEL` / `LLM_BASE_URL` | Provider (`anthropic`, `openai`-compatible), pinned model id, gateway URL (`https://` required) | `anthropic` / provider default |
| `PHARMAGUARD_DB_PATH`, `PHARMAGUARD_OBS_DB`, `PHARMAGUARD_LOG_DIR` | Locations of the review DB, trace DB and logs | `/app/data/app/...`, `/app/logs` |
| `STREAMLIT_SERVER_PORT` | Listening port (also change `-p`) | `8501` |

Ways to supply secrets: `--env-file .env` (local), `-e LLM_API_KEY` (shell), Docker/Compose *secrets*, or your platform's secret
manager (Kubernetes Secrets, cloud secret stores). Do **not** put keys in the Dockerfile, `docker-compose.yml`, or `ARG`s -
`ARG`/`ENV` values are visible in `docker history`. `.env` is excluded by both `.gitignore` and `.dockerignore`.
Only synthetic demo data is baked into the image; the review and trace databases are created at run time in the mounted volumes.

Notes: the image is ~0.9 GB (pandas / numpy / pyarrow / plotly / streamlit). It has no authentication in front of it - put it
behind your organisation's SSO / reverse proxy before exposing it beyond a trusted network.

### Requirements files
| File | Contents |
|---|---|
| `requirements.txt` | Runtime dependencies only, each bounded to the next major version |
| `requirements-dev.txt` | `-r requirements.txt` + pytest, pytest-cov, ruff, bandit, pip-audit |
| `requirements.lock` | Exact pins of the full runtime resolution (Linux, Python 3.12) - what the image ships and CI tests |

Refreshing the lockfile (after changing `requirements.txt`; needs Docker):
```bash
sh scripts/refresh_lock.sh
```
Local development on older Pythons can use `pip install -r requirements-dev.txt` (the lock needs Python >= 3.11).

### CI/CD flow (GitHub Actions: `.github/workflows/ci.yml`)
```
push to main / pull request
   └─ 1 Dependency installation   pip install from requirements.lock + dev tools, `pip check`
       └─ 2 Code quality        ruff, syntax check, config parse, bandit (medium+)
           └─ 3 Unit tests      pytest (fast tests) + coverage gate >= 90 %
               └─ 4 Integration pytest: cross-module, Streamlit pages, critical end-to-end workflow
                   └─ 5 Security   executable security controls (secret scan, guardrail probes, RBAC ...) + pip-audit on requirements.lock
                       └─ 6 AI evaluation gate   68-case golden evaluation vs config/evaluation_gate.yaml
                           └─ 7 Docker    hadolint, build, hardened smoke test (health, non-root, no secrets, every page loads),
                                          image saved as artifact  (+ optional push to GHCR)
                               └─ 8 Deploy   OPTIONAL - off unless DEPLOY_ENABLED=true, main branch only, 'production' environment
```
Each stage only runs if the previous one passed, so a failing test, failed security control, or failed AI/guardrail check stops
the pipeline before an image is produced. No cloud credentials are needed: stages 1-6 use only the runner and the built-in
`GITHUB_TOKEN`. Every run uploads test results, the AI evaluation report and the image artifact (`pharmaguard-ai-image`, a
`docker save` tarball: `gunzip -c pharmaguard-ai-image.tar.gz | docker load`).

**AI evaluation gate** - thresholds live in `config/evaluation_gate.yaml`; each has a *level*:

| Level | Effect on a failed check |
|---|---|
| `critical` | Always fails the pipeline (guardrail success, unsafe-output catch rate, groundedness, unsupported claims, every case in the prompt-injection / disposition-request / adversarial-output categories) |
| `required` | Fails the pipeline (analytical correctness >= 0.98, tool success = 1.0, retrieval relevance >= 0.70, answer relevance >= 0.90) - unless `EVAL_GATE_MODE=warn` |
| `advisory` | Reported as a warning only (p95 latency <= 1500 ms) |

It also fails if fewer than 60 golden cases ran or if the report was produced from a different golden dataset than the one in the repo.
Run it locally: `python scripts/evaluation_gate.py --run` (exit 0 = pass, 1 = blocked, 2 = configuration error).
Change a threshold without editing code by setting a GitHub *repository variable* (Settings > Secrets and variables > Actions >
Variables), e.g. `EVAL_GATE_ANSWER_RELEVANCE_MIN=0.85`, `EVAL_GATE_MODE=warn`, or `EVAL_GATE_LATENCY_P95_MS_LEVEL=required`
(pattern: `EVAL_GATE_<METRIC>_MIN|_MAX|_LEVEL`, metric names upper-cased).

**Other repository variables** (all optional): `PIP_AUDIT_BLOCKING=false` (make the vulnerability scan advisory),
`PIP_AUDIT_IGNORE=ID1,ID2` (accept specific advisories), `PUBLISH_IMAGE=true` (push the image to
`ghcr.io/<owner>/pharmaguard-ai` on `main` / version tags), `DEPLOY_ENABLED=true` + `DEPLOY_TARGET=<target>` (enable the deploy stage).

**Deployment is a configurable next stage.** The `deploy` job is skipped by default. To use it: create a GitHub Environment named
`production` (add required reviewers), store target secrets there (for example `DEPLOY_HOST`, `DEPLOY_SSH_KEY`, `LLM_API_KEY`),
set the variables above, and implement your target in the `Deploy` step (an `ssh` skeleton is included; other targets fail loudly
until implemented). Recommended hardening once real deployment exists: pin actions to commit SHAs, add container image
scanning (e.g. Trivy), sign images, and use OIDC federation instead of long-lived cloud keys.

Run the same checks locally:
```bash
pip install -r requirements-dev.txt
ruff check . && bandit -q -r src app scripts config -ll
pytest -m "not integration and not e2e and not ui" --cov=src --cov-fail-under=90     # unit
pytest -m "integration or e2e or ui"                                                  # integration / UI / end-to-end
python scripts/run_security_checks.py && python scripts/evaluation_gate.py --run
```

### Publish to GitHub
```bash
gh auth login --web --scopes "repo,workflow"      # one-time sign-in (workflow scope is needed to push .github/workflows)
sh scripts/publish_to_github.sh PharmaGuard-AI    # creates a PRIVATE repo, pushes main, starts the CI/CD run  (add --public for public)
gh run watch                                      # follow the pipeline
```
No repository secrets are required for stages 1-6. Add optional *variables* (see the CI/CD section) under
Settings > Secrets and variables > Actions > Variables.

## Production monitoring
Open **Production Monitoring** in the app (QA Reviewer / Compliance-IT Admin roles). It shows a status banner, a status card per
domain and, per domain, indicators with `ok / warn / critical` status computed from `config/monitoring.yaml`:

| Domain | Indicators |
|---|---|
| Application | availability, requests, errors, latency (p50 / p95 / max) |
| AI | answer quality (golden set), evaluation age, unsupported claims, retrieval failures, guardrail failures (live output validation) |
| Data | missingness, quality score, drift (PSI, latest vs earlier batches), missingness drift, schema changes vs baseline |
| Security | failed / unauthorized requests (access denials), suspicious inputs (injection / disposition attempts), guardrail & security events |
| Cost | token usage, LLM calls, estimated cost (only for models priced in config), token budget |
| Business | investigations assisted, human approvals / rejections / more-analysis, approval rate, average support time, time to review, review backlog |

Plus **Incidents & events** (critical/degraded requests, blocked outputs, injections, access denials, active alerts), **Health**
probes and an **Integration** tab. Admins can **simulate production traffic** (with an incident burst), data drift and a schema
change from the sidebar - simulated data is flagged `synthetic` and simulations are display-only.

```bash
python scripts/register_data_baseline.py     # (re)register the expected raw-data schema
python scripts/metrics_server.py             # Prometheus exporter: http://127.0.0.1:9108/metrics  (--once prints)
```
Thresholds, windows, prices and budgets are in `config/monitoring.yaml`; no model prices are assumed (add yours). Integration with
OpenTelemetry, Prometheus / Grafana (scrape config, alert rules, importable dashboard in `monitoring/`) and other platforms, plus what
is real vs simulated in this demo, is documented in [docs/observability-integration.md](docs/observability-integration.md).

### Deploy to Vercel (container Function)
`Dockerfile.vercel` is the Vercel variant of the image: it listens on `$PORT` and keeps runtime state in `/tmp`. Vercel strips
`.gitignore` from uploads, so it is not copied into this image.
```bash
npx vercel login                                   # one-time (device login)
npx vercel deploy --prod --yes --name pharmaguard-ai
```
Caveats: the app has **no authentication** and the production `*.vercel.app` alias is public (Vercel Authentication protects only
deployment-specific URLs on the Hobby plan) - enable Vercel Authentication / password protection on a paid plan before sharing.
State (review and trace databases) lives in `/tmp` of each Function instance, so it is ephemeral and not shared between instances.
Set `LLM_API_KEY` as a Vercel *environment variable* (never in the repo) to enable a live LLM. Long-lived Streamlit WebSocket sessions
are subject to Vercel Function duration limits - use a container host (Cloud Run, Render, Fly.io) for durable production use.
