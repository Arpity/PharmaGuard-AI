"""Human-in-the-Loop QA Review: humans approve / reject / request more analysis on AI FINDINGS."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import components as ui  # noqa: E402
from src.guardrails import roles  # noqa: E402
from src.review.store import DECISIONS, STATUS_LABELS, ReviewError  # noqa: E402

st.set_page_config(page_title="QA Review | PharmaGuard AI", page_icon="✅", layout="wide")
st.markdown("""<style>
[data-testid="stMetric"] {background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:12px 16px;}
.src {border:1px solid #e2e8f0; border-radius:10px; padding:10px 14px; margin:6px 0; background:#f8fafc; font-size:.9rem;}
</style>""", unsafe_allow_html=True)

user, role = ui.identity_sidebar()
store = ui.get_store()

st.title("✅ QA Review of AI Findings")
st.warning("**You are reviewing an AI finding, not disposing a batch.** Approving a finding means the investigation "
           "support is accurate and useful. It does **not** approve, reject or release the batch - batch disposition is "
           "made only by authorized Quality personnel in the quality system (QMS).")

runs = store.list_runs()
k = st.columns(4)
counts = runs["status"].value_counts() if len(runs) else pd.Series(dtype=int)
for c, key in zip(k, STATUS_LABELS):
    c.metric(STATUS_LABELS[key], int(counts.get(key, 0)))

tab_queue, tab_audit = st.tabs(["Review queue", "Audit trail"])
if "review_flash" in st.session_state:
    st.success(st.session_state.pop("review_flash"))

with tab_queue:
    f1, f2 = st.columns([1, 1])
    status_f = f1.selectbox("Status", ["(all)"] + list(STATUS_LABELS), index=1, format_func=lambda v: v if v == "(all)" else STATUS_LABELS[v])
    batch_f = f2.text_input("Batch_ID contains", placeholder="B00042").strip().upper()
    view = runs if status_f == "(all)" else runs[runs["status"] == status_f]
    if batch_f:
        view = view[view["batch_id"].str.contains(batch_f)]
    if view.empty:
        st.info("No AI runs match. Ask the assistant a question about a batch on the **Batch Investigation** page - each "
                "answer is saved here with a Run ID.")
        st.stop()
    st.dataframe(view[["run_id", "created_at", "batch_id", "question", "user_name", "mode", "status", "review_count"]],
                 hide_index=True, width="stretch")
    run_id = st.selectbox("Open a run", view["run_id"].tolist(),
                          format_func=lambda r: f"{r} - {view.set_index('run_id').loc[r, 'batch_id']} - "
                                                f"{STATUS_LABELS[view.set_index('run_id').loc[r, 'status']]}")
    run = store.get_run(run_id)

    st.subheader(f"{run['run_id']} · Batch {run['batch_id']}")
    st.caption(f"Asked by **{run['user_name']}** ({roles.ROLES.get(run['user_role'], run['user_role'])}) at {run['created_at']} UTC · "
               f"{run['model_info']} · current status: **{STATUS_LABELS[run['status']]}**")
    st.markdown(f"**Question:** {run['question']}")
    with st.container(border=True):
        st.markdown(run["answer"])
    g = run["guardrail"] or {}
    st.caption(f"Guardrails - input: `{g.get('input')}` · output: `{g.get('output')}`"
               + (" · ⚠ needs verification" if g.get("needs_verification") else ""))
    for e in g.get("events", []):
        st.markdown(f"- `{e['type']}` ({e['severity']}): {e['detail']}")
    with st.expander(f"Sources ({len(run['sources'] or [])})"):
        for s_ in run["sources"] or []:
            st.markdown(f"<div class='src'><b>[{s_['tag']}] {s_['document']}</b> › {s_['section']} "
                        f"<span style='color:#64748b'>· relevance {s_['score']} · <code>{s_['file']}</code></span><br>"
                        f"{s_['text'][:400]}</div>", unsafe_allow_html=True)
    with st.expander("Evidence: calculated analysis"):
        a = run["analysis"] or {}
        for line in a.get("risk_indicators", []) or ["None triggered."]:
            st.markdown(f"- {line}")
        st.json({k_: a.get(k_) for k_ in ("history", "deviations", "data_notes")}, expanded=False)

    st.markdown("#### Previous reviews")
    hist = store.reviews(run_id)
    if hist.empty:
        st.caption("No reviews yet.")
    else:
        hist["decision"] = hist["decision"].map(DECISIONS)
        st.dataframe(hist[["review_id", "timestamp", "reviewer", "decision", "comments"]], hide_index=True, width="stretch")

    st.markdown("#### Your review")
    if not roles.can(role, "review_ai_finding"):
        st.info(f"The **{roles.ROLES[role]}** role cannot review AI findings. Switch to **QA Reviewer** in the sidebar (demo).")
    else:
        st.caption(f"Reviewing as **{user}**. Comments are required to reject a finding or request more analysis. "
                   "You cannot review a run you asked yourself (four-eyes).")
        with st.form(f"review_{run_id}", clear_on_submit=True):
            comments = st.text_area("Comments", max_chars=2000, placeholder="Rationale, corrections, or what additional analysis is needed")
            b1, b2, b3 = st.columns(3)
            picked = next((d for d, btn in (("approved", b1.form_submit_button("✔ " + DECISIONS["approved"], type="primary", width="stretch")),
                                            ("rejected", b2.form_submit_button("✖ " + DECISIONS["rejected"], width="stretch")),
                                            ("more_analysis", b3.form_submit_button("↻ " + DECISIONS["more_analysis"], width="stretch")))
                           if btn), None)
        if picked:
            try:
                rid = store.add_review(run_id, user, role, picked, comments)
                # st.rerun() clears the page, so keep the confirmation in session state and show it after the refresh
                st.session_state["review_flash"] = (f"Recorded: **{DECISIONS[picked]}** for {run_id} (review #{rid}). This is an entry "
                                                    "about the AI finding only; no batch status was changed.")
                st.rerun()
            except (ReviewError, roles.PermissionDenied) as e:
                if isinstance(e, roles.PermissionDenied):
                    ui.log_security_event("PERMISSION_DENIED", "QA Review page", str(e))
                elif "Four-eyes" in str(e):
                    ui.log_security_event("FOUR_EYES_VIOLATION", "QA Review page", f"reviewer attempted to review own run {run_id}")
                st.error(str(e))

with tab_audit:
    st.caption("Append-only: reviews, runs and guardrail events cannot be edited or deleted.")
    rv = store.reviews()
    st.markdown("**Review decisions**")
    if rv.empty:
        st.caption("None yet.")
    else:
        rv["decision"] = rv["decision"].map(DECISIONS)
        st.dataframe(rv, hide_index=True, width="stretch")
        st.download_button("⬇ Reviews (CSV)", rv.to_csv(index=False).encode(), "qa_reviews.csv", "text/csv")
    st.markdown("**Guardrail events**")
    ev = store.events()
    st.dataframe(ev, hide_index=True, width="stretch") if not ev.empty else st.caption("None yet.")
