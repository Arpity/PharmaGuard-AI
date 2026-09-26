"""PharmaGuard AI - home page. Run: streamlit run app/streamlit_app.py"""
import app.components  # noqa: F401  (adds project root to sys.path)
import streamlit as st

st.set_page_config(page_title="PharmaGuard AI", page_icon="💊", layout="wide")
st.title("💊 PharmaGuard AI")
st.caption("Pharmaceutical batch-quality analytics")
st.markdown(
    """
**Available now**
- **Data Quality** (sidebar): score the raw batch data, drill into each issue type, run the
  controlled cleaning pipeline and inspect the audit log.

- **Analytics Dashboard**: KPIs, interactive charts, filters (product, plant, date, status) and automatic business insights.

**Coming in later steps**: batch-failure prediction and deeper root-cause analysis.

Raw data in `data/raw/` is read-only by design; cleaned data is written to `data/processed/`.
"""
)
