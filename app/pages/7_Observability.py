"""Observability Dashboard: request volume, success, latency, guardrail activity, retrieval health, token usage, traces."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from app import components as ui  # noqa: E402
from src.analytics import kpis as analytics  # noqa: E402
from src.assistant.knowledge import KnowledgeBase  # noqa: E402
from src.guardrails import roles  # noqa: E402
from src.observability import metrics as M  # noqa: E402
from src.observability.demo import generate_demo_traffic  # noqa: E402

st.set_page_config(page_title="Observability | PharmaGuard AI", page_icon="📡", layout="wide")
st.markdown("""<style>
[data-testid="stMetric"] {background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:12px 16px;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600;}
</style>""", unsafe_allow_html=True)

cfg = ui.get_config()
user, role = ui.identity_sidebar()
st.title("📡 Observability")
st.caption("Every AI request gets a Run ID and a trace (OpenTelemetry-compatible spans). Telemetry is stored locally for the demo.")
if not roles.can(role, "view_observability"):
    ui.log_security_event("UNAUTHORIZED_ACCESS", "Observability page", f"view_observability denied for role {role}")
    st.error("Your role cannot view observability data (it contains user names and batch IDs). "
             "Switch to **QA Reviewer** or **Compliance / IT Admin** in the sidebar.")
    st.stop()

rec = ui.get_recorder()
STATUS_COLORS = {"success": "#16a34a", "success_flagged": "#84cc16", "degraded": "#f59e0b", "refused": "#38bdf8",
                 "blocked": "#6366f1", "invalid_request": "#94a3b8", "error": "#dc2626"}

with st.sidebar:
    st.header("Filters")
    window = st.selectbox("Time window", ["Last hour", "Last 24 hours", "Last 7 days", "All time"], index=3)
    include_syn = st.checkbox("Include synthetic demo traffic", value=True)
    st.header("Demo data")
    n_demo = st.number_input("Requests to generate", 10, 500, 100, 10)
    if st.button("▶ Generate demo traffic", disabled=not roles.can(role, "generate_demo_traces"),
                 help="Runs synthetic requests through the real pipeline (Compliance / IT Admin only)"):
        clean = ui.ROOT / cfg["processed"]["clean_path"]
        if not clean.exists():
            st.error("Cleaned data not found. Run scripts/run_cleaning.py first.")
        else:
            with st.spinner("Generating..."):
                generate_demo_traffic(int(n_demo), analytics.load_clean(clean), KnowledgeBase.from_directory(ui.ROOT / "knowledge"),
                                      cfg, rec, seed=int(datetime.now().timestamp()) % 10_000)
            st.rerun()
    if not roles.can(role, "generate_demo_traces"):
        st.caption("Generating demo traffic requires the Compliance / IT Admin role.")

delta = {"Last hour": timedelta(hours=1), "Last 24 hours": timedelta(hours=24), "Last 7 days": timedelta(days=7)}.get(window)
since = (datetime.now(timezone.utc) - delta).isoformat(timespec="seconds") if delta else None
df = rec.store.requests(since=since, include_synthetic=include_syn)
if df.empty:
    st.info("No AI requests recorded in this window. Ask the assistant a question on **Batch Investigation**, or (as Compliance / IT Admin) "
            "generate demo traffic from the sidebar.")
    st.stop()

k = M.kpis(df)
c = st.columns(4)
c[0].metric("Total requests", f"{k['total_requests']:,}")
c[1].metric("Success rate", f"{k['success_rate']:.1f}%", f"{k['clean_success_rate']:.1f}% clean answers", delta_color="off",
            help="Requests handled without a system error. Guardrail blocks, refusals and invalid requests are correct handling, not failures.")
c[2].metric("Failed requests", f"{k['failed_requests']:,}", help="Requests that ended in an internal error (a reference id was shown to the user).")
c[3].metric("Avg latency", f"{k['avg_latency_ms']:.1f} ms", f"p95 {k['p95_latency_ms']:.1f} ms", delta_color="off")
c = st.columns(4)
c[0].metric("Guardrail blocks", f"{k['guardrail_blocks']:,}", help="Blocked injections / invalid input, refused disposition requests, and AI answers replaced by output validation.")
c[1].metric("Retrieval failures", f"{k['retrieval_failures']:,}", help="Found batches where knowledge retrieval errored or returned nothing.")
c[2].metric("Token usage", f"{k['total_tokens']:,}", f"in {k['input_tokens']:,} · out {k['output_tokens']:,}", delta_color="off")
c[3].metric("Token coverage", f"{k['token_coverage']:.0f}%", f"{k['requests_with_usage']} of {k['total_requests']} requests report usage", delta_color="off",
            help="Tokens are recorded only when a live LLM provider reports them; demo mode uses no model.")

tab_over, tab_perf, tab_guard, tab_runs = st.tabs(["Overview", "Latency & tools", "Guardrails & retrieval", "Recent runs & traces"])
LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10))

with tab_over:
    a, b = st.columns([2, 1])
    with a:
        span_h = (pd.to_datetime(df["timestamp"], utc=True, format="ISO8601").max() - pd.to_datetime(df["timestamp"], utc=True, format="ISO8601").min()).total_seconds() / 3600
        ot = M.requests_over_time(df, "h" if span_h > 3 else "min")
        ot["status"] = ot["final_status"].map(M.STATUS_LABELS)
        fig = px.bar(ot, x="bucket", y="requests", color="final_status", color_discrete_map=STATUS_COLORS, title="Requests over time by final status",
                     category_orders={"final_status": M.STATUS_ORDER}, labels={"bucket": "", "final_status": "Status"})
        fig.update_layout(height=340, **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        sc = pd.DataFrame({"status": [M.STATUS_LABELS[s] for s in k["status_counts"]], "key": list(k["status_counts"]), "n": list(k["status_counts"].values())})
        fig = px.pie(sc, names="status", values="n", hole=.55, color="key", color_discrete_map=STATUS_COLORS, title="Final status mix")
        fig.update_traces(sort=False).update_layout(height=340, showlegend=False, **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Models**")
    st.dataframe(M.model_breakdown(df).round(2), hide_index=True, width="stretch")

sp_all = rec.store.spans()
sp = sp_all[sp_all["run_id"].isin(df["run_id"])]
with tab_perf:
    a, b = st.columns(2)
    with a:
        fig = px.histogram(df, x="latency_ms", nbins=40, color_discrete_sequence=["#2563eb"], title="Request latency distribution (ms)")
        fig.update_layout(height=320, **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        sl = M.stage_latency(sp)
        fig = px.bar(sl, x="mean_ms", y="stage", orientation="h", color_discrete_sequence=["#2563eb"], title="Mean time per pipeline stage (ms)",
                     text=sl["mean_ms"].map(lambda v: f"{v:.2f}"))
        fig.update_layout(height=320, yaxis=dict(autorange="reversed"), **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Tool-call success**")
    st.dataframe(M.tool_summary(sp).round(2), hide_index=True, width="stretch")

with tab_guard:
    a, b = st.columns(2)
    with a:
        ev = M.guardrail_event_counts(df)
        if ev.empty:
            st.success("No guardrail events in this window.")
        else:
            fig = px.bar(ev, x="count", y="event", orientation="h", color_discrete_sequence=["#6366f1"], title="Guardrail events by type")
            fig.update_layout(height=340, yaxis=dict(autorange="reversed"), **LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
    with b:
        rf = df[df["retrieval_failed"]]
        st.markdown(f"**Retrieval failures: {len(rf)}**")
        st.dataframe(rf[["timestamp", "run_id", "batch_id", "error_type", "final_status"]], hide_index=True, width="stretch") if len(rf) else st.success("No retrieval failures.")
        er = df[df["error_type"].fillna("") != ""]
        st.markdown(f"**Requests with errors: {len(er)}**")
        st.dataframe(er[["timestamp", "run_id", "error_type", "error_ref", "final_status"]], hide_index=True, width="stretch") if len(er) else st.success("No errors.")

with tab_runs:
    f = st.columns(4)
    st_f = f[0].multiselect("Final status", [s for s in M.STATUS_ORDER if s in set(df["final_status"])], format_func=M.STATUS_LABELS.get)
    u_f = f[1].multiselect("User", sorted(df["user_name"].dropna().unique()))
    g_f = f[2].checkbox("Guardrail blocks only")
    b_f = f[3].text_input("Batch_ID / Run ID contains").strip().upper()
    v = df
    if st_f: v = v[v["final_status"].isin(st_f)]
    if u_f: v = v[v["user_name"].isin(u_f)]
    if g_f: v = v[v["guardrail_blocked"]]
    if b_f: v = v[v["batch_id"].fillna("").str.upper().str.contains(b_f) | v["run_id"].str.contains(b_f)]
    show = v.assign(status=v["final_status"].map(M.STATUS_LABELS), guardrail=v["guardrail_input"] + " / " + v["guardrail_output"],
                    docs=v["retrieved_documents"].map(lambda x: len(json.loads(x or "[]"))),
                    tools=v["tool_calls"].map(lambda x: len(json.loads(x or "[]"))), synthetic_=v["synthetic"].map({True: "yes", False: ""}))
    st.caption(f"{len(v):,} of {len(df):,} requests (newest first, showing up to 200)")
    st.dataframe(show[["run_id", "timestamp", "user_name", "user_role", "batch_id", "model", "prompt_version", "latency_ms", "guardrail", "docs",
                       "tools", "total_tokens", "status", "error_type", "synthetic_"]].head(200), hide_index=True, width="stretch")
    if len(v):
        rid = st.selectbox("Inspect run", v["run_id"].head(200).tolist())
        req = rec.store.get_request(rid)
        spans = rec.store.spans(rid)
        st.markdown(f"#### {rid}")
        m = st.columns(4)
        m[0].metric("Status", M.STATUS_LABELS[req["final_status"]])
        m[1].metric("Latency", f"{req['latency_ms']:.1f} ms")
        m[2].metric("Model", req["model_version"], req["prompt_version"] and f"prompt v{req['prompt_version']}", delta_color="off")
        m[3].metric("Tokens", req["total_tokens"] if req["total_tokens"] is not None else "n/a", req["token_source"] or "not reported", delta_color="off")
        st.caption(f"Trace `{req['trace_id']}` · {req['user_name']} ({req['user_role']}) · batch {req['batch_id']} · {req['timestamp']} UTC"
                   + (" · synthetic" if req["synthetic"] else ""))
        if len(spans):
            t0 = spans["start_ns"].min()
            wf = spans.assign(start=pd.to_datetime(spans["start_ns"], unit="ns"), finish=pd.to_datetime(spans["end_ns"], unit="ns"))
            fig = px.timeline(wf, x_start="start", x_end="finish", y="name", color="status_code", title="Span waterfall",
                              color_discrete_map={"OK": "#2563eb", "ERROR": "#dc2626", "UNSET": "#94a3b8"})
            fig.update_yaxes(autorange="reversed", title="").update_layout(height=max(260, 34 * len(wf) + 90), **LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Retrieved documents**")
            docs = pd.DataFrame(req["retrieved_documents"])
            st.dataframe(docs, hide_index=True, width="stretch") if len(docs) else st.warning("None retrieved.")
            st.markdown("**Guardrail result**")
            st.write(f"input: `{req['guardrail_input']}` · output: `{req['guardrail_output']}`" + (" · **blocked**" if req["guardrail_blocked"] else ""))
            if req["guardrail_events"]:
                st.write(", ".join(f"`{e}`" for e in req["guardrail_events"]))
            st.markdown("**Errors**")
            st.json(req["errors"]) if req["errors"] else st.caption("None")
        with d2:
            st.markdown("**Tool calls**")
            tc = pd.DataFrame(req["tool_calls"])
            st.dataframe(tc.drop(columns=["detail"], errors="ignore"), hide_index=True, width="stretch") if len(tc) else st.caption("None")
            with st.expander("Span attributes"):
                for r in spans.itertuples():
                    st.markdown(f"**{r.name}** · {r.duration_ms:.2f} ms · {r.status_code}")
                    st.json(json.loads(r.attributes or "{}"), expanded=False)
        with st.expander("OTLP/JSON (ready for an OpenTelemetry collector)"):
            otlp = rec.store.otlp(rid, version=cfg.get("version", ""))
            st.json(otlp, expanded=False)
            st.download_button("⬇ Trace (OTLP JSON)", json.dumps(otlp, indent=1).encode(), f"{rid}.otlp.json", "application/json")
    log_path = Path(ui.os.getenv("PHARMAGUARD_LOG_DIR") or ui.ROOT / "logs") / "observability.jsonl"
    st.caption(f"Structured log: `{log_path.name}` (one JSON line per request, correlated by run_id / trace_id; free text stored as length + hash).")
