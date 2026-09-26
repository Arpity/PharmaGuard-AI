## Integrating with enterprise observability

The local demo stores telemetry in SQLite and computes indicators in Python. Everything is shaped so it can be exported to standard
platforms without changing the application logic. **What is real vs simulated in this demo:**

| Signal | Source in the demo | Real or simulated |
|---|---|---|
| Requests, errors, latency, guardrail results, retrieval failures, tokens | Traces recorded for every AI request (`data/app/observability.db`) | Real for requests you make; *simulated traffic* is flagged `synthetic` and can be generated from the sidebar |
| Answer quality | Latest golden-set evaluation (`reports/evaluation/latest.json`) | Real (deterministic demo mode) |
| Data quality, missingness, drift, schema | Computed from `data/raw`, `data/processed`; drift compares early vs latest batches (PSI) | Real calculations on synthetic data; *drift / schema-change scenarios can be simulated* (display only) |
| Unauthorized requests | Access denials logged by the pages (`security_events`) | Real for denials you trigger; some simulated |
| Cost | Provider-reported tokens x prices in `config/monitoring.yaml` | Only for live LLM calls; demo prices are illustrative |
| Human approvals / time to review | QA Review database | Real (demo reviews can be seeded) |
| Availability | Share of requests handled without an internal error + local health probes | Local proxy; production needs external synthetic probes |

Thresholds and status rules live in `config/monitoring.yaml` (`ok` / `warn` / `critical`).

### 1. OpenTelemetry (traces, metrics, logs)

Traces are already OTel-shaped (`src/observability/tracing.py`): 128-bit trace ids, 64-bit span ids, span kinds, status, events,
`gen_ai.*` / `enduser.*` attributes, W3C `traceparent` propagation, and an OTLP/JSON serializer (`to_otlp_json`, also downloadable
per run on the Observability page).

| Step | How |
|---|---|
| Send traces to a collector today | POST the OTLP/JSON from `TraceStore.otlp(run_id)` to `http://<collector>:4318/v1/traces` |
| Use the real SDK | `pip install opentelemetry-sdk opentelemetry-exporter-otlp`; replace `Tracer` with `opentelemetry.trace.get_tracer("pharmaguard")`. The pipeline calls (`start_span`, `set_attribute`, `record_exception`, `add_event`) have the same names |
| Metrics | Create OTel instruments and map them from `Snapshot` (table below); export via OTLP to the collector |
| Logs | Structured JSON lines already carry `trace_id` / `run_id`; ship `logs/observability.jsonl` with the collector `filelog` receiver or the OTel logging handler |
| Config | Standard environment variables: `OTEL_SERVICE_NAME=pharmaguard-ai`, `OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317`, `OTEL_RESOURCE_ATTRIBUTES=deployment.environment=prod` |

A starter collector configuration is in `monitoring/otel/otel-collector.yaml` (OTLP in; Prometheus + debug/OTLP out).

| Monitoring indicator | OTel instrument (suggested) | Name |
|---|---|---|
| Requests, errors | Counter | `pharmaguard.requests` (attribute `status`) |
| Latency | Histogram | `pharmaguard.request.duration` (ms) |
| Guardrail blocks / failures | Counter | `pharmaguard.guardrail.events` (attribute `type`) |
| Retrieval failures | Counter | `pharmaguard.retrieval.failures` |
| Tokens / LLM calls | Counter | `gen_ai.client.token.usage` (attribute `gen_ai.token.type`), `pharmaguard.llm.calls` |
| Data drift, quality score | Gauge (observable) | `pharmaguard.data.drift.psi` (attribute `feature`), `pharmaguard.data.quality.score` |
| Human review decisions | Counter | `pharmaguard.review.decisions` (attribute `decision`) |

### 2. Prometheus + Grafana

The demo ships a read-only exporter:

```bash
python scripts/metrics_server.py --port 9108          # GET http://127.0.0.1:9108/metrics
python scripts/metrics_server.py --once               # print once
```

* `monitoring/prometheus/prometheus.yml` - scrape config (15s interval, job `pharmaguard`).
* `monitoring/prometheus/alerts.yml` - alert rules mirroring the SLO thresholds (availability, latency, errors, unsupported claims,
  guardrail failures, drift, schema change, unauthorized requests, token budget, review backlog, indicator status).
* `monitoring/grafana/pharmaguard-dashboard.json` - importable dashboard (Dashboards > Import) with panels per domain.

Metric names are prefixed `pharmaguard_`; the exporter reports window aggregates as gauges (window set with `--window`, default
24h). In production, emit true monotonic counters from the application (`prometheus_client` or OTel metrics) and use `rate()` /
`increase()` in PromQL, for example:

```promql
sum(rate(pharmaguard_requests_total{status="error"}[5m])) / sum(rate(pharmaguard_requests_total[5m]))   # error ratio
histogram_quantile(0.95, sum(rate(pharmaguard_request_duration_ms_bucket[5m])) by (le))                 # p95 latency
max(pharmaguard_data_drift_psi)                                                                          # worst drift
```

### 3. Other platforms

| Platform | Route |
|---|---|
| Datadog / New Relic / Dynatrace / Honeycomb | Point the OTLP exporter (or their collector distribution) at the vendor endpoint; map indicator status to monitors |
| Azure Monitor / Application Insights | Azure Monitor OpenTelemetry distro; alerts on custom metrics |
| AWS CloudWatch / X-Ray | AWS Distro for OpenTelemetry (ADOT) collector; CloudWatch alarms on the same metrics |
| Google Cloud Monitoring / Trace | OTLP to Google Cloud Ops via the collector |
| Splunk / Elastic / Loki | Ship `logs/observability.jsonl` (JSON lines with `run_id` / `trace_id`); build dashboards from the fields |
| LLM observability (Langfuse, LangSmith, Arize Phoenix) | Export `gen_ai.*` spans over OTLP; keep prompts out of traces unless approved (see privacy) |
| PagerDuty / Opsgenie / Teams / Slack | Route Alertmanager or vendor monitors; use the `severity` label |

### 4. Production hardening checklist

* **Privacy**: traces and logs store lengths and hashes of free text, not the text; keep it that way when adding LLM span content.
  Restrict dashboards (roles `view_observability`), set retention (`TraceStore.purge_older_than`), and review PII in user names.
* **Cardinality**: keep metric labels bounded (status, category, feature); never label by run id, batch id or user.
* **Sampling**: sample high-volume successful traces (tail-based sampling that always keeps errors, blocks and slow requests).
* **SLOs and alerting**: alert on burn rate of the availability / latency SLOs, route `critical` to on-call, `warn` to a queue; write a
  runbook per alert (each indicator key in `config/monitoring.yaml` is a natural runbook id).
* **Availability**: add external synthetic probes (blackbox exporter) against `/_stcore/health` and a scripted investigation.
* **Cost**: price each model in `config/monitoring.yaml`, set token/cost budgets, alert at 80% / 100%.
* **Data monitoring**: run drift and schema checks on a schedule against each new data load and store the results; page on schema breaks.
* **Security**: forward guardrail and access-denial events to the SIEM; alert on bursts of injection or unauthorized attempts.
* **Exposure**: the demo exporter binds to `127.0.0.1` without authentication - put it behind network policy / mTLS before scraping remotely.
