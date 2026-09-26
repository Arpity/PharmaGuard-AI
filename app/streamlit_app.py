"""PharmaGuard AI - home page. Run: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

# `streamlit run app/streamlit_app.py` puts only app/ on sys.path, so add the project root before importing the `app` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="PharmaGuard AI", page_icon="💊", layout="wide")
st.title("💊 PharmaGuard AI")
st.caption("Pharmaceutical batch-quality analytics")
st.markdown(
    """
Use the sidebar to open a page:

- **Data Quality** - score the raw batch data, run the controlled cleaning pipeline, inspect the audit log.
- **Analytics Dashboard** - KPIs, charts, filters and business insights.
- **Batch Investigation** - ask about a batch (deterministic analysis + QA procedures + optional LLM).
- **QA Review** - approve / reject / request more analysis on AI findings.
- **Evaluation Dashboard**, **AI Governance**, **Observability**, **Production Monitoring**.

The assistant supports investigations only: it never approves, rejects, releases or dispositions a batch.
Raw data in `data/raw/` is read-only by design; cleaned data is written to `data/processed/`.
Roles in the sidebar are a demo: there is no authentication.
"""
)
