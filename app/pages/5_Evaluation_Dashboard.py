"""AI Evaluation Dashboard: overall and per-metric results from the golden test dataset."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app import components as ui  # noqa: E402
from src.analytics import kpis  # noqa: E402
from src.assistant.knowledge import KnowledgeBase  # noqa: E402
from src.evaluation.golden import load_golden  # noqa: E402
from src.evaluation.runner import run_evaluation  # noqa: E402

st.set_page_config(page_title="AI Evaluation | PharmaGuard AI", page_icon="🧪", layout="wide")
st.markdown("""<style>
[data-testid="stMetric"] {background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:12px 16px;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600;}
.pill.ok {background:#dcfce7; color:#166534;} .pill.bad {background:#fee2e2; color:#991b1b;}
</style>""", unsafe_allow_html=True)

cfg = ui.get_config()
golden_path = ui.ROOT / cfg["evaluation"]["golden_path"]
results_path = Path(ui.os.getenv("PHARMAGUARD_EVAL_PATH") or ui.ROOT / cfg["evaluation"]["results_path"])
clean_path = ui.ROOT / cfg["processed"]["clean_path"]
BLUE, GREEN, RED = "#2563eb", "#16a34a", "#dc2626"

st.title("🧪 AI Evaluation Dashboard")
st.caption("Quality of the batch-investigation assistant, measured on a synthetic golden test dataset. "
           "Scores are deterministic - the same code and data always give the same result.")

with st.sidebar:
    st.header("Run evaluation")
    live = st.checkbox("Use live LLM (if configured)", value=False,
                       help="Off = deterministic demo mode, no network. On = the configured LLM answers non-adversarial cases.")
    if st.button("▶ Run evaluation now", type="primary", width="stretch"):
        if not (golden_path.exists() and clean_path.exists()):
            st.error("Golden dataset or cleaned data missing. Run `scripts/build_golden.py` and `scripts/run_cleaning.py`.")
        else:
            with st.spinner("Running golden cases..."):
                run_evaluation(load_golden(golden_path), kpis.load_clean(clean_path),
                               KnowledgeBase.from_directory(ui.ROOT / "knowledge"), cfg, live=live, save_to=results_path)
            st.rerun()

if not results_path.exists():
    st.info("No evaluation results yet. Click **Run evaluation now** in the sidebar, or run `python scripts/run_evaluation.py`.")
    st.stop()

rep = json.loads(results_path.read_text())
S, M, T, ST = rep["summary"], rep["summary"]["metrics"], rep["summary"]["targets"], rep["summary"]["target_status"]
cases = pd.DataFrame(rep["cases"])

top = st.columns([1, 1, 1, 1])
top[0].metric("Overall score", f"{S['overall_score'] * 100:.1f}%")
top[1].metric("Cases passed", f"{S['cases_passed']} / {S['cases']}")
top[2].metric("Targets met", f"{sum(ST.values())} / {len(ST)}")
top[3].metric("Mode", "Live LLM" if rep["mode"].startswith("live") else "Demo")
st.markdown(f"<span class='pill {'ok' if S['all_targets_met'] else 'bad'}'>"
            f"{'ALL TARGETS MET' if S['all_targets_met'] else 'TARGETS MISSED'}</span> &nbsp; "
            f"<span style='color:#64748b'>{rep['mode']} · generated {rep['generated_at']} UTC</span>", unsafe_allow_html=True)


def card(col, label, key, fmt, help_=None):
    op, val = T[key]["op"], T[key]["value"]
    ok = ST[key]
    col.metric(label, fmt(M[key]), f"{'✓' if ok else '✗'} target {op} {fmt(val)}", delta_color="normal" if ok else "inverse", help=help_)
    

pct = lambda v: f"{v * 100:.1f}%"
ms = lambda v: f"{v:,.0f} ms"
r1, r2 = st.columns(4), st.columns(4)
card(r1[0], "Analytical correctness", "analytical_correctness", pct, "Deterministic analysis vs independently computed expected findings.")
card(r1[1], "Retrieval relevance", "retrieval_relevance", pct, "F1 of retrieved knowledge documents vs expected source topics.")
card(r1[2], "Groundedness", "groundedness", pct, "Numbers, citations and batch/SOP ids in answers trace to the supplied facts.")
card(r1[3], "Answer relevance", "answer_relevance", pct, "Expected content for the question type is present (lexical check).")
card(r2[0], "Unsupported-claim rate", "unsupported_claim_rate", pct, "Delivered answers containing an unflagged unsupported claim. Lower is better.")
card(r2[1], "Guardrail success", "guardrail_success", pct, "Expected input/output guardrail behaviour occurred.")
card(r2[2], "Tool execution success", "tool_success", pct, "Batch lookup, analysis, retrieval, answer generation and run persistence.")
card(r2[3], "Latency p95", "latency_p95_ms", ms, "95th percentile time of the full pipeline per question.")

tab_over, tab_cat, tab_ret, tab_lat, tab_cases, tab_method = st.tabs(
    ["Metrics vs targets", "By category", "Retrieval & tools", "Latency", "Case explorer", "Methodology"])

with tab_over:
    keys = [("Analytical correctness", "analytical_correctness"), ("Retrieval relevance", "retrieval_relevance"),
            ("Groundedness", "groundedness"), ("Answer relevance", "answer_relevance"),
            ("Guardrail success", "guardrail_success"), ("Tool success", "tool_success")]
    df = pd.DataFrame({"metric": [k[0] for k in keys], "score": [M[k[1]] * 100 for k in keys],
                       "target": [T[k[1]]["value"] * 100 for k in keys], "met": [ST[k[1]] for k in keys]})
    fig = go.Figure()
    fig.add_bar(y=df["metric"], x=df["score"], orientation="h", marker_color=[GREEN if m else RED for m in df["met"]],
                text=df["score"].map(lambda v: f"{v:.1f}%"), textposition="outside", name="Score", cliponaxis=False)
    fig.add_scatter(y=df["metric"], x=df["target"], mode="markers", marker=dict(symbol="line-ns", size=22, color="#0f172a", line=dict(width=3)),
                    name="Target")
    fig.update_layout(template="plotly_white", height=380, xaxis=dict(range=[0, 108], title="Score (%)"), yaxis=dict(autorange="reversed"),
                      margin=dict(l=10, r=30, t=40, b=10), title="Score vs target (black tick = target)", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)
    c = st.columns(3)
    c[0].metric("Unsafe-output catch rate", pct(M["unsafe_output_catch_rate"]) if M["unsafe_output_catch_rate"] is not None else "n/a",
                help="Adversarial model replies (disposition claims, invented numbers, fake citations, prompt leaks) that were handled correctly.")
    c[1].metric("Answers flagged for verification", pct(M["flagged_answer_rate"]),
                help="Answers delivered with an explicit 'needs verification' flag (warn-level issues).")
    c[2].metric("Retrieval hit@1", pct(M["retrieval_hit_at_1"]), help="Top-ranked chunk comes from an expected source document.")

with tab_cat:
    cat = pd.DataFrame(S["by_category"]).T.reset_index().rename(columns={"index": "category"})
    st.dataframe(cat.assign(pass_rate=(cat["passed"] / cat["cases"]).round(3)).round(3), hide_index=True, width="stretch")
    cols = [c for c in ["analytical_correctness", "retrieval_f1", "groundedness", "answer_relevance", "guardrail_success", "tool_success"] if c in cat]
    heat = cat.set_index("category")[cols].astype(float)
    fig = px.imshow(heat.fillna(float("nan")) * 100, text_auto=".0f", color_continuous_scale="RdYlGn", zmin=0, zmax=100, aspect="auto",
                    labels={"color": "%"}, title="Metric score (%) by question category (blank = not applicable)")
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

with tab_ret:
    a, b = st.columns(2)
    with a:
        rd = pd.DataFrame({"metric": ["Recall", "Precision", "Hit@1", "F1 (relevance)"],
                           "score": [M["retrieval_recall"], M["retrieval_precision"], M["retrieval_hit_at_1"], M["retrieval_relevance"]]})
        fig = px.bar(rd, x="metric", y=rd["score"] * 100, text=rd["score"].map(pct), color_discrete_sequence=[BLUE],
                     title="Knowledge retrieval quality", labels={"y": "%"})
        fig.update_layout(template="plotly_white", height=340, yaxis=dict(range=[0, 110]))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        ts = pd.DataFrame({"stage": list(S["tool_stage_success"]), "success": [v * 100 for v in S["tool_stage_success"].values()]})
        fig = px.bar(ts, x="stage", y="success", text=ts["success"].map(lambda v: f"{v:.0f}%"), color_discrete_sequence=[GREEN],
                     title="Tool / pipeline stage success")
        fig.update_layout(template="plotly_white", height=340, yaxis=dict(range=[0, 110], title="%"))
        st.plotly_chart(fig, use_container_width=True)

with tab_lat:
    l = st.columns(4)
    l[0].metric("Mean", ms(M["latency_mean_ms"]))
    l[1].metric("p50", ms(M["latency_p50_ms"]))
    l[2].metric("p95", ms(M["latency_p95_ms"]))
    l[3].metric("Max", ms(M["latency_max_ms"]))
    fig = px.box(cases, x="category", y="latency_ms", points="all", color_discrete_sequence=[BLUE], title="Latency per case by category (ms)")
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    if rep["mode"].startswith("demo"):
        st.caption("Demo mode has no network call, so this is the cost of retrieval, analysis and guardrails. "
                   "A live LLM adds provider latency on top.")

with tab_cases:
    f = st.columns(3)
    cat_f = f[0].multiselect("Category", sorted(cases["category"].unique()))
    res_f = f[1].selectbox("Result", ["All", "Failed only", "Passed only"])
    q_f = f[2].text_input("Search question / batch")
    v = cases
    if cat_f: v = v[v["category"].isin(cat_f)]
    if res_f == "Failed only": v = v[~v["passed"]]
    if res_f == "Passed only": v = v[v["passed"]]
    if q_f: v = v[v["question"].str.contains(q_f, case=False) | v["batch_id"].str.contains(q_f, case=False)]
    show = v.assign(result=v["passed"].map({True: "✓ pass", False: "✗ fail"}), issues=v["failures"].map(lambda x: "; ".join(x)))
    st.caption(f"{len(v)} of {len(cases)} cases")
    st.dataframe(show[["case_id", "result", "category", "archetype", "batch_id", "question", "latency_ms", "guardrail", "issues"]],
                 hide_index=True, width="stretch")
    if len(v):
        cid = st.selectbox("Case detail", v["case_id"].tolist())
        row = cases[cases["case_id"] == cid].iloc[0]
        st.json({"scores": row["scores"], "tools": row["tools"], "failures": row["failures"]}, expanded=False)
        st.markdown("**Answer excerpt**")
        st.text(row["answer_excerpt"])

with tab_method:
    st.markdown("""
**Golden dataset** (`data/evaluation/golden_cases.json`, built by `scripts/build_golden.py`): representative questions for
7 batch archetypes × 5 question types, plus hostile inputs (prompt injection, disposition requests), invalid inputs, and
simulated adversarial model replies. Expected findings come from an independent oracle (`src/evaluation/oracle.py`).

| Metric | How it is measured |
|---|---|
| Analytical correctness | analysis output equals expected status, out-of-spec set, breach amounts, deviation flag, peer statistics, data notes |
| Retrieval relevance | recall / precision / F1 of retrieved documents vs expected source topics |
| Groundedness | numbers, citations and batch/SOP ids in the answer traceable to supplied facts |
| Answer relevance | expected content for the question type is present (keyword groups + intent recognition) |
| Unsupported claims | delivered answers with unflagged violations (invented numbers, bad citations, fabricated ids, disposition claims) |
| Guardrail success | expected input verdict, LLM-call count, output verdict, no forbidden text, disposition refusals |
| Tool success | batch lookup, analysis, retrieval, answer generation, run persistence |
| Latency | wall-clock time of the full pipeline per case (p50 / p95) |

**Limitations**: relevance is a lexical proxy, not human or LLM-judged; the golden set is synthetic and modest in size;
demo mode evaluates the deterministic answer writer, so run with a live LLM to evaluate model wording.
""")
