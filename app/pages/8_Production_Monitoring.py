"""Production Monitoring: application, AI, data, security, cost and business health with status indicators and incidents."""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app import components as ui  # noqa: E402
from src.analytics import kpis as analytics  # noqa: E402
from src.assistant.knowledge import KnowledgeBase  # noqa: E402
from src.data_quality import checks  # noqa: E402
from src.guardrails import roles  # noqa: E402
from src.monitoring import drift as D  # noqa: E402
from src.monitoring.config import CATEGORIES, ICON, load_monitoring_config  # noqa: E402
from src.monitoring.prometheus import render_prometheus  # noqa: E402
from src.monitoring.simulate import seed_demo_reviews, simulate_production  # noqa: E402
from src.monitoring.snapshot import WINDOWS, build_snapshot  # noqa: E402
from src.observability import metrics as OM  # noqa: E402
from src.review.store import DECISIONS  # noqa: E402

st.set_page_config(page_title="Production Monitoring | PharmaGuard AI", page_icon="🩺", layout="wide")
st.markdown("""<style>
[data-testid="stMetric"] {background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:12px 16px;}
.banner {padding:12px 18px; border-radius:12px; font-weight:600; margin-bottom:10px;}
.banner.ok {background:#dcfce7; color:#166534;} .banner.warn {background:#fef3c7; color:#92400e;}
.banner.critical {background:#fee2e2; color:#991b1b;} .banner.unknown {background:#f1f5f9; color:#475569;}
.sim {background:#ede9fe; color:#5b21b6; padding:6px 12px; border-radius:8px; font-weight:600; margin-bottom:8px;}
.ind {border:1px solid #e2e8f0; border-radius:12px; padding:10px 14px; background:#fff; margin-bottom:8px;}
.ind b {font-size:.85rem; color:#475569;} .ind .v {font-size:1.25rem; font-weight:700; color:#0f172a;} .ind small {color:#64748b;}
</style>""", unsafe_allow_html=True)

cfg = ui.get_config()
mon = load_monitoring_config()
user, role = ui.identity_sidebar()
st.title("🩺 Production Monitoring")
st.caption("Application, AI, data, security, cost and business indicators with SLO status. Runs on local demo telemetry; "
           "the same indicators map to OpenTelemetry / Prometheus (see the Integration tab).")
if not roles.can(role, "view_observability"):
    ui.log_security_event("UNAUTHORIZED_ACCESS", "Production Monitoring page", f"view_observability denied for role {role}")
    st.error("Your role cannot view monitoring data. Switch to **QA Reviewer** or **Compliance / IT Admin** in the sidebar.")
    st.stop()

rec = ui.get_recorder()
review_store = ui.get_store()
can_sim = roles.can(role, "generate_demo_traces")
ss = st.session_state

with st.sidebar:
    st.header("View")
    window = st.selectbox("Time window", list(WINDOWS), index=list(WINDOWS).index(mon["windows"]["default"]),
                          format_func={"1h": "Last hour", "24h": "Last 24 hours", "7d": "Last 7 days", "all": "All time"}.get)
    include_syn = st.checkbox("Include synthetic demo traffic", value=True)
    st.header("Simulation (demo)")
    if not can_sim:
        st.caption("Simulation requires the Compliance / IT Admin role.")
    n_sim = st.slider("Requests to simulate", 50, 500, 200, 50, disabled=not can_sim)
    incident = st.checkbox("Include an incident burst (last 45 min)", value=True, disabled=not can_sim,
                           help="Injection spree, LLM provider failures, unsafe model replies, retrieval failures, errors, access denials")
    seed_reviews = st.checkbox("Also seed demo QA reviews", value=False, disabled=not can_sim,
                               help="Adds clearly named demo runs + decisions to the QA Review queue")
    if st.button("▶ Simulate production traffic", disabled=not can_sim):
        clean_p = ui.ROOT / cfg["processed"]["clean_path"]
        if not clean_p.exists():
            st.error("Cleaned data not found.")
        else:
            df_ = analytics.load_clean(clean_p)
            kb_ = KnowledgeBase.from_directory(ui.ROOT / "knowledge")
            with st.spinner("Simulating..."):
                simulate_production(rec, df_, kb_, cfg, n=n_sim, incident=incident, seed=int(datetime.now().timestamp()) % 10_000)
                if seed_reviews:
                    seed_demo_reviews(review_store, df_, kb_, cfg)
            st.rerun()
    ss["sim_drift"] = st.checkbox("Simulate data drift", value=ss.get("sim_drift", False), disabled=not can_sim,
                                  help="Display-only: shifts the latest window of the cleaned data")
    ss["sim_schema"] = st.checkbox("Simulate schema change", value=ss.get("sim_schema", False), disabled=not can_sim,
                                   help="Display-only: drops, renames and adds a column in the raw data")

raw_p, clean_p = ui.ROOT / cfg["dataset"]["raw_path"], ui.ROOT / cfg["processed"]["clean_path"]
raw = checks.load_raw(raw_p) if raw_p.exists() else pd.DataFrame()
clean = analytics.load_clean(clean_p) if clean_p.exists() else pd.DataFrame()
simulated = []
if ss.get("sim_drift") and len(clean):
    clean, _ = D.simulate_drift(clean, mon, 1.0), simulated.append("data drift")
if ss.get("sim_schema") and len(raw):
    raw, _ = D.simulate_schema_change(raw), simulated.append("schema change")

snap = build_snapshot(cfg, window, include_syn, mon_cfg=mon, trace_store=rec.store, review_store=review_store, raw=raw, clean=clean)
M = snap.metrics
LABEL = {"ok": "ALL SYSTEMS NORMAL", "warn": "DEGRADED - WARNINGS ACTIVE", "critical": "INCIDENT - CRITICAL ALERTS ACTIVE", "unknown": "NO DATA"}
n_crit = sum(i.status == "critical" for i in snap.indicators) + sum(h["status"] == "critical" for h in snap.health)
n_warn = sum(i.status == "warn" for i in snap.indicators)
if simulated:
    st.markdown(f"<div class='sim'>SIMULATION ACTIVE: {', '.join(simulated)} (display only, not saved)</div>", unsafe_allow_html=True)
st.markdown(f"<div class='banner {snap.overall}'>{ICON[snap.overall]} {LABEL[snap.overall]} · {n_crit} critical · {n_warn} warning · "
            f"window {window} · updated {snap.generated_at[11:19]} UTC</div>", unsafe_allow_html=True)
for note in snap.notes:
    st.caption(note)

cols = st.columns(6)
for c, cat in zip(cols, CATEGORIES):
    inds = snap.by_category(cat)
    c.metric(f"{ICON[snap.category_status(cat)]} {cat}", snap.category_status(cat).upper(),
             f"{sum(i.status in ('warn', 'critical') for i in inds)} of {len(inds)} need attention", delta_color="off")


def cards(cat: str, per_row: int = 4) -> None:
    inds = snap.by_category(cat)
    for i in range(0, len(inds), per_row):
        for col, ind in zip(st.columns(per_row), inds[i:i + per_row]):
            col.markdown(f"<div class='ind'><b>{ICON[ind.status]} {ind.name}</b><br><span class='v'>{ind.display}</span><br>"
                         f"<small>{ind.rule}{' · ' + ind.detail if ind.detail else ''}</small></div>", unsafe_allow_html=True)


def pct(v):
    return "n/a" if v is None else f"{v:.1f}%"


LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10), height=320)
req = snap.requests
tabs = st.tabs(["Overview", "Application", "AI", "Data", "Security", "Cost", "Business", "Incidents & events", "Health", "Integration"])

with tabs[0]:
    for cat in CATEGORIES:
        st.markdown(f"**{ICON[snap.category_status(cat)]} {cat}**")
        cards(cat)
    st.markdown("#### Active alerts")
    active = [i for i in snap.indicators if i.status in ("warn", "critical")] 
    if active:
        st.dataframe(pd.DataFrame([{"": ICON[i.status], "category": i.category, "indicator": i.name, "value": i.display, "threshold": i.rule, "detail": i.detail}
                                   for i in sorted(active, key=lambda x: x.status != "critical")]), hide_index=True, width="stretch")
    else:
        st.success("No active alerts.")
    st.markdown("#### Latest incidents & events")
    if len(snap.events):
        st.dataframe(snap.events.head(8), hide_index=True, width="stretch")
    else:
        st.caption("No events in this window.")

with tabs[1]:
    a = M["application"]
    cards("Application", 3)
    k = st.columns(4)
    k[0].metric("Requests", f"{a['requests']:,}")
    k[1].metric("Errors", a["errors"])
    k[2].metric("Latency p50 / max", f"{a['latency_p50_ms'] or 0:.1f} / {a['latency_max_ms'] or 0:.1f} ms")
    k[3].metric("Requests / hour", "n/a" if a["requests_per_hour"] is None else f"{a['requests_per_hour']:.1f}")
    if len(req):
        ot = OM.requests_over_time(req, "h" if WINDOWS[window] is None or WINDOWS[window].total_seconds() > 10800 else "min")
        fig = px.bar(ot, x="bucket", y="requests", color="final_status", title="Requests over time by outcome", labels={"bucket": ""},
                     category_orders={"final_status": OM.STATUS_ORDER})
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        t = pd.to_datetime(req["timestamp"], utc=True, format="ISO8601").dt.tz_localize(None).dt.floor("h")
        lat = req.assign(bucket=t).groupby("bucket")["latency_ms"].quantile([.5, .95]).unstack().rename(columns={.5: "p50", .95: "p95"}).reset_index()
        fig = px.line(lat, x="bucket", y=["p50", "p95"], title="Latency over time (ms)", labels={"value": "ms", "bucket": ""})
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    ai = M["ai"]
    cards("AI", 3)
    k = st.columns(4)
    k[0].metric("Live answers checked", ai["answers"])
    k[1].metric("Unsupported-claim answers", ai["unsupported_claim_answers"])
    k[2].metric("Retrieval failures", ai["retrieval_failures"])
    k[3].metric("Guardrail failures (output blocked)", ai["guardrail_failures"])
    if ai["evaluation_available"]:
        ev = pd.DataFrame({"metric": ["Groundedness", "Answer relevance", "Retrieval relevance", "Guardrail success", "Overall"],
                           "score": [ai["eval_groundedness"], ai["eval_answer_relevance"], ai["eval_retrieval_relevance"], ai["eval_guardrail_success"],
                                     ai["answer_quality_pct"] / 100]})
        fig = px.bar(ev, x="score", y="metric", orientation="h", range_x=[0, 1.05], title="Answer quality: latest golden-set evaluation",
                     text=ev["score"].map(lambda v: f"{v:.3f}"), color_discrete_sequence=["#2563eb"])
        fig.update_layout(yaxis=dict(autorange="reversed"), **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No evaluation report - run scripts/run_evaluation.py.")
    st.caption("Unsupported claims and guardrail failures are counted on LIVE traffic from output-validation events; "
               "answer quality comes from the golden-set evaluation (run by CI).")

with tabs[3]:
    d = M["data"]
    cards("Data", 4)
    k = st.columns(4)
    k[0].metric("Rows raw / cleaned", f"{d['rows_raw']:,} / {d['rows_clean']:,}")
    k[1].metric("Quality score raw → cleaned", f"{d['quality_score_raw'] or 0:.1f} → {d['quality_score'] or 0:.1f}")
    k[2].metric("Drifting features", d["drifting_features"])
    k[3].metric("Schema changes", "n/a" if d["schema_changes"] is None else d["schema_changes"])
    a_, b_ = st.columns(2)
    with a_:
        mb = d["missing_by_column"].head(12)
        fig = px.bar(mb, x="missing_pct", y="column", orientation="h", title="Missing values by column (raw, %)", color_discrete_sequence=["#f59e0b"])
        fig.update_layout(yaxis=dict(autorange="reversed"), **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with b_:
        dr = d["drift"]
        if len(dr):
            colors = {"ok": "#16a34a", "warn": "#f59e0b", "critical": "#dc2626", "unknown": "#94a3b8"}
            fig = go.Figure(go.Bar(x=dr["psi"], y=dr["feature"], orientation="h", marker_color=[colors[s] for s in dr["status"]]))
            for v in (mon["slo"]["drift_psi_max"]["warn"], mon["slo"]["drift_psi_max"]["critical"]):
                fig.add_vline(x=v, line_dash="dash", line_color="#64748b")
            fig.update_layout(title="Drift: PSI, latest vs earlier batches (dashed = warn / critical)", yaxis=dict(autorange="reversed"), **LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Schema check** (raw data vs registered baseline)")
    diff = d["schema_diff"]
    if diff is None:
        st.info("No schema baseline registered. Run `python scripts/register_data_baseline.py`.")
    elif not D.schema_change_count(diff):
        st.success("Schema matches the baseline (same columns, order and inferred types).")
    else:
        st.error("Schema changed vs baseline.")
        st.json(diff, expanded=True)
    st.markdown("**Missingness drift** (per column, percentage-point change between windows)")
    st.dataframe(M["missingness_drift"].head(8), hide_index=True, width="stretch")
    if d["quality_dimensions"]:
        qd = pd.DataFrame({"dimension": list(d["quality_dimensions"]), "score": list(d["quality_dimensions"].values())})
        st.dataframe(qd, hide_index=True, width="stretch")

with tabs[4]:
    s = M["security"]
    cards("Security", 2)
    k = st.columns(4)
    k[0].metric("Failed / unauthorized requests", s["failed_requests"], f"{s['unauthorized_requests']} unauthorized", delta_color="off")
    k[1].metric("Suspicious inputs", s["suspicious_inputs"], pct(s["suspicious_input_rate_pct"]), delta_color="off")
    k[2].metric("Injections blocked", s["injections_blocked"])
    k[3].metric("Guardrail / security events", s["guardrail_security_events"])
    if s["event_counts"]:
        ec = pd.DataFrame(sorted(s["event_counts"].items(), key=lambda kv: -kv[1]), columns=["event", "count"])
        fig = px.bar(ec, x="count", y="event", orientation="h", title="Guardrail & security events by type", color_discrete_sequence=["#6366f1"])
        fig.update_layout(yaxis=dict(autorange="reversed"), **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Access denials / unauthorized attempts**")
    if len(snap.security_events):
        st.dataframe(snap.security_events[["timestamp", "event_type", "severity", "user_name", "user_role", "source", "detail"]].head(50),
                     hide_index=True, width="stretch")
    else:
        st.success("None in this window.")

with tabs[5]:
    c = M["cost"]
    cards("Cost", 2)
    k = st.columns(4)
    k[0].metric("Total tokens", f"{c['total_tokens']:,}", f"in {c['input_tokens']:,} · out {c['output_tokens']:,}", delta_color="off")
    k[1].metric("LLM calls", c["llm_calls"], f"{c['llm_call_failures']} failed", delta_color="off")
    k[2].metric("Estimated cost", "not configured" if c["estimated_cost_usd"] is None else f"${c['estimated_cost_usd']:.4f}",
                None if c["priced_token_share_pct"] is None else f"{c['priced_token_share_pct']:.0f}% of tokens priced", delta_color="off")
    k[3].metric("Avg tokens / call", "n/a" if c["avg_tokens_per_call"] is None else f"{c['avg_tokens_per_call']:,.0f}")
    if len(req) and c["total_tokens"]:
        t = pd.to_datetime(req["timestamp"], utc=True, format="ISO8601").dt.tz_localize(None).dt.floor("h")
        tk = req.assign(bucket=t).groupby("bucket")[["input_tokens", "output_tokens"]].sum().reset_index()
        fig = px.bar(tk, x="bucket", y=["input_tokens", "output_tokens"], title="Token usage over time", labels={"value": "tokens", "bucket": ""})
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(c["by_model"], hide_index=True, width="stretch")
    st.caption(c["pricing_note"] + " Demo mode makes no LLM calls, so it costs nothing; the demo model rate is illustrative.")

with tabs[6]:
    b = M["business"]
    cards("Business", 2)
    k = st.columns(4)
    k[0].metric("Investigations assisted", b["investigations_assisted"], f"{b['unique_batches']} unique batches", delta_color="off")
    k[1].metric("Human decisions", b["human_decisions"], f"✔ {b['approvals']} · ✖ {b['rejections']} · ↻ {b['more_analysis_requests']}", delta_color="off")
    k[2].metric("Approval rate", pct(b["approval_rate_pct"]), f"review coverage {pct(b['review_coverage_pct'])}", delta_color="off")
    k[3].metric("Avg support time", "n/a" if b["avg_support_seconds"] is None else f"{b['avg_support_seconds']:.2f} s",
                "time to first review: " + ("n/a" if b["avg_time_to_first_review_hours"] is None else f"{b['avg_time_to_first_review_hours']:.1f} h"), delta_color="off")
    if b["estimated_minutes_saved"] is not None:
        st.metric("Estimated time saved (assumption-based)", f"{b['estimated_minutes_saved'] / 60:.1f} h")
    if len(req):
        ass = req[(req["final_status"].isin(["success", "success_flagged", "degraded", "refused"])) & (req["mode"] != "n/a")]
        day = ass.assign(day=pd.to_datetime(ass["timestamp"], utc=True, format="ISO8601").dt.tz_localize(None).dt.floor("D")).groupby("day").size().reset_index(name="investigations")
        fig = px.bar(day, x="day", y="investigations", title="Investigations assisted per day", color_discrete_sequence=["#2563eb"], labels={"day": ""})
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    if len(snap.reviews):
        rv = snap.reviews["decision"].map(DECISIONS).value_counts().reset_index()
        rv.columns = ["decision", "count"]
        st.plotly_chart(px.bar(rv, x="decision", y="count", title="Human decisions on AI findings", color_discrete_sequence=["#16a34a"]).update_layout(**LAYOUT),
                        use_container_width=True)
    else:
        st.caption("No human decisions in this window (use QA Review, or seed demo reviews from the sidebar).")

with tabs[7]:
    ev = snap.events
    sev = st.multiselect("Severity", ["critical", "warn", "info"], default=["critical", "warn", "info"])
    cat = st.multiselect("Category", CATEGORIES)
    v = ev[ev["severity"].isin(sev)] if len(ev) else ev
    if cat and len(v):
        v = v[v["category"].isin(cat)]
    st.caption(f"{len(v)} events (newest first)")
    st.dataframe(v.assign(severity=v["severity"].map({"critical": "🔴 critical", "warn": "🟡 warn", "info": "🔵 info"})) if len(v) else v,
                 hide_index=True, width="stretch")

with tabs[8]:
    hc = pd.DataFrame(snap.health)
    st.dataframe(hc.assign(status=hc["status"].map(lambda s: f"{ICON[s]} {s}")), hide_index=True, width="stretch")
    st.caption("Probes read local files and databases only; the LLM provider is never pinged (avoids cost and data egress).")

with tabs[9]:
    doc = ui.ROOT / "docs" / "observability-integration.md"
    st.markdown("#### Live Prometheus exposition of this snapshot")
    prom = render_prometheus(snap)
    st.code("\n".join(prom.splitlines()[:40]) + "\n...", language="text")
    st.download_button("⬇ Full /metrics text", prom.encode(), "pharmaguard_metrics.prom", "text/plain")
    st.caption("Serve it continuously with `python scripts/metrics_server.py` (http://127.0.0.1:9108/metrics).")
    if doc.exists():
        st.markdown(doc.read_text(encoding="utf-8"))
    else:
        st.info("docs/observability-integration.md not found.")
