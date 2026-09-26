"""Batch Investigation AI Assistant: question -> batch data -> Python analysis -> knowledge -> LLM -> answer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app import components as ui  # noqa: E402
from src.analytics import kpis  # noqa: E402
from src.assistant import analysis as A  # noqa: E402
from src.assistant.knowledge import KnowledgeBase  # noqa: E402
from src.assistant.llm import LLMConfig  # noqa: E402
from src.assistant.pipeline import investigate  # noqa: E402
from src.guardrails import roles  # noqa: E402
from src.observability.recorder import RequestContext  # noqa: E402
from src.security.secure_logging import get_security_logger, log_event  # noqa: E402

st.set_page_config(page_title="Batch Investigation | PharmaGuard AI", page_icon="🔎", layout="wide")
st.markdown("""<style>
[data-testid="stMetric"] {background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:12px 16px;}
.src {border:1px solid #e2e8f0; border-radius:10px; padding:10px 14px; margin:6px 0; background:#f8fafc; font-size:.9rem;}
.badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem; font-weight:600;}
.badge.llm {background:#dcfce7; color:#166534;} .badge.demo {background:#fef3c7; color:#92400e;}
</style>""", unsafe_allow_html=True)

cfg = ui.get_config()
clean_path = ui.ROOT / cfg["processed"]["clean_path"]
st.title("🔎 Batch Investigation Assistant")
st.caption("Ask about a batch. Numbers come from deterministic Python analysis; procedures come from the knowledge base; "
           "the language model only explains them.")
if not clean_path.exists():
    st.warning("Cleaned data not found. Run `python scripts/run_cleaning.py` first.")
    st.stop()


@st.cache_data(show_spinner=False)
def _load(path: str, mtime: float) -> pd.DataFrame:
    return kpis.load_clean(path)


@st.cache_resource(show_spinner=False)
def _kb(_dir: str) -> KnowledgeBase:
    return KnowledgeBase.from_directory(_dir)


df = _load(str(clean_path), clean_path.stat().st_mtime)
kb = _kb(str(ui.ROOT / "knowledge"))
llm_cfg = LLMConfig.from_env()
user, role = ui.identity_sidebar()
store = ui.get_store()

QUESTIONS = ["Investigate this batch.", "Why is this batch showing risk?", "Compare it with historical batches.",
             "Which parameters are abnormal?", "What procedure should QA review?"]

with st.sidebar:
    st.header("Assistant")
    st.info("Investigation support only. The assistant never approves, rejects, releases or dispositions a batch.")
    if llm_cfg.is_live:
        st.success(f"LLM: {llm_cfg.provider} / {llm_cfg.model}")
        st.caption("Batch analysis and retrieved chunks are sent to this provider.")
    else:
        st.warning("Demo mode - no LLM key configured. Answers come from a deterministic template using the same "
                   "analysis and sources. Set `LLM_API_KEY` in `.env` to enable a live model.")
    st.markdown("**Knowledge base** (synthetic demo documents)")
    for doc in sorted({c.doc_title for c in kb.chunks}):
        st.markdown(f"- {doc}")
    st.caption(f"{len(kb.chunks)} chunks indexed")

# Batch selection -------------------------------------------------------------------------------
examples = df[df["Quality_Status"].isin(["Fail", "Under Review"]) & df["Outlier_Flag"]].sort_values("Batch_ID")["Batch_ID"].head(200)
c1, c2 = st.columns([2, 3])
batch_id = c1.text_input("Batch_ID", value=st.session_state.get("batch_id", ""), placeholder="e.g. B02343").strip().upper()
pick = c2.selectbox("...or pick a flagged example", [""] + examples.tolist(), format_func=lambda v: v or "Select a batch")
if pick:
    batch_id = pick
if batch_id != st.session_state.get("batch_id"):
    st.session_state.update(batch_id=batch_id, history=[])

if not batch_id:
    st.info("Enter a Batch_ID to begin.")
    st.stop()

base = A.analyze_batch(df, batch_id, cfg)
if base is None:
    st.error(f"Batch {batch_id} not found.")
    sug = A.suggest_batch_ids(df, batch_id)
    if sug:
        st.caption("Did you mean: " + ", ".join(sug))
    st.stop()

# Batch context card ----------------------------------------------------------------------------
b = base["batch"]
m = st.columns(5)
m[0].metric("Batch", b["Batch_ID"], b["Manufacturing_Date"], delta_color="off")
m[1].metric("Product", b["Product_Name"], b["Dosage_Form"], delta_color="off")
m[2].metric("Plant / Shift", f"{b['Plant']}", b["Operator_Shift"], delta_color="off")
m[3].metric("Quality status", b["Quality_Status"], b["Equipment_ID"], delta_color="off")
m[4].metric("Abnormal signals", len(base["abnormal_parameters"]), f"{len(base['out_of_spec_parameters'])} out of spec", delta_color="off")

with st.expander("Batch parameters vs historical batches of the same product", expanded=False):
    p = pd.DataFrame(base["parameters"])
    z = p.dropna(subset=["z_score"])
    fig = go.Figure(go.Bar(x=z["z_score"], y=z["parameter"], orientation="h",
                           marker_color=["#dc2626" if f in ("out_of_spec", "statistical_outlier") else "#f59e0b" if f == "watch" else "#93c5fd"
                                         for f in z["flag"]]))
    fig.add_vline(x=3, line_dash="dash", line_color="#dc2626"); fig.add_vline(x=-3, line_dash="dash", line_color="#dc2626")
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=40, b=10),
                      title="Deviation from peer mean (z-score; dashed = ±3)", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(p[["parameter", "value_text", "spec_text", "flag", "z_score", "percentile", "peer_mean", "peer_n"]]
                 .rename(columns={"value_text": "value", "spec_text": "specification", "peer_mean": "peer mean"}),
                 hide_index=True, width="stretch")

# Chat ------------------------------------------------------------------------------------------
st.subheader("Ask")
qcols = st.columns(len(QUESTIONS))
clicked = next((q for col, q in zip(qcols, QUESTIONS) if col.button(q, width="stretch")), None)
typed = st.chat_input("Ask a question about this batch...")
question = clicked or typed
can_ask = roles.can(role, "ask_assistant")
if not can_ask:
    if question:
        ui.log_security_event("PERMISSION_DENIED", "Batch Investigation page", f"ask_assistant denied for role {role}")
    st.info(f"The **{roles.ROLES[role]}** role is read-only. Switch role in the sidebar to ask the assistant.")
elif question:
    with st.spinner("Analysing batch and retrieving procedures..."):
        ctx = RequestContext(user=user, role=role, store=store, recorder=ui.get_recorder())
        inv = investigate(question, batch_id, df, kb, cfg, llm_cfg, ctx=ctx)
        if not inv.found:                       # blocked / invalid requests are logged as guardrail events
            try:
                for e in inv.guardrail.get("events", []):
                    store.log_event(user, e["type"], e["severity"], e["detail"], run_id=None)
            except Exception:  # noqa: BLE001
                pass
        try:                # secure audit line: identifiers and verdicts only; the question is logged as length + hash
            log_dir = Path(ui.os.getenv("PHARMAGUARD_LOG_DIR") or ui.ROOT / "logs")
            log_event(get_security_logger(log_dir / "security_audit.log"), "ai_request", run_id=inv.run_id, user=user, role=role,
                      batch_id=inv.batch_id, question=inv.question, input_verdict=inv.guardrail.get("input"),
                      output_verdict=inv.guardrail.get("output"), mode=inv.mode, found=inv.found)
        except OSError:
            pass
        st.session_state.history.append(inv)


def render(inv) -> None:
    with st.chat_message("user"):
        st.write(inv.question)
    with st.chat_message("assistant"):
        badge = f"<span class='badge {inv.mode}'>{'LLM' if inv.mode == 'llm' else 'DEMO MODE'}</span> {inv.model_info}"
        st.markdown(badge, unsafe_allow_html=True)
        if inv.run_id:
            st.caption(f"Run ID **{inv.run_id}** · saved for QA review (status: pending)")
        if inv.notice:
            st.warning(inv.notice)
        if inv.blocked:
            st.error(inv.answer)
            return
        if inv.error_ref:
            st.error(inv.answer)
            return
        if inv.guardrail.get("needs_verification"):
            st.warning("Some statements could not be fully verified against the data - review carefully.")
        if inv.fallback_reason:
            st.warning(f"The LLM call failed, so the demo answer is shown. ({inv.fallback_reason})")
        st.markdown(inv.answer)
        if inv.grounding_warnings:
            st.warning("Numeric check: these figures in the answer were not found in the calculated facts - verify before "
                       "use: " + ", ".join(inv.grounding_warnings))
        elif inv.mode == "llm":
            st.caption("✓ Output validation passed: numbers, citations and references trace to the analysis or sources.")
        with st.expander(f"Sources ({len(inv.hits)} knowledge chunks retrieved)"):
            for i, h in enumerate(inv.hits, 1):
                st.markdown(f"<div class='src'><b>[K{i}] {h.chunk.doc_title}</b> › {h.chunk.section} "
                            f"<span style='color:#64748b'>· relevance {h.score} · <code>{Path(h.chunk.path).name}</code></span><br>"
                            f"<span style='color:#475569'>{h.chunk.text[:420]}{'…' if len(h.chunk.text) > 420 else ''}</span></div>",
                            unsafe_allow_html=True)
            if inv.hits:
                st.caption(inv.hits[0].chunk.note)
        with st.expander("Evidence: deterministic analysis"):
            a = inv.analysis
            st.markdown("**Risk indicators**")
            for line in a["risk_indicators"] or ["None triggered."]:
                st.markdown(f"- {line}")
            st.json({"history": a["history"], "deviations": a["deviations"], "data_notes": a["data_notes"]}, expanded=False)
        with st.expander("Pipeline trace"):
            for s in inv.steps:
                st.markdown(f"**{s['step']}** - {s['detail']}")
            st.caption(f"Knowledge query: {inv.knowledge_query}")
            st.caption(f"Run {inv.run_id or '-'} · trace {inv.trace_id or '-'} · latency {inv.latency_ms:.0f} ms"
                       + (f" · tokens {inv.usage['total_tokens']}" if inv.usage else " · tokens n/a"))
            g = inv.guardrail
            st.markdown(f"**Guardrails** - input: `{g.get('input')}` · output: `{g.get('output')}`")
            for e in g.get("events", []):
                st.markdown(f"- `{e['type']}` ({e['severity']}): {e['detail']}")


for inv in st.session_state.history:
    render(inv)
