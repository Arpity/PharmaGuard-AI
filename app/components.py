"""Shared Streamlit helpers (path bootstrap, cached loaders, chart styling)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import copy  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from config import load_config  # noqa: E402
from src.data_quality import checks  # noqa: E402

ACCENT, WARN, BAD, MUTED = "#2563eb", "#d97706", "#dc2626", "#94a3b8"


def get_config() -> dict:
    return load_config()


@st.cache_data(show_spinner=False)
def load_dataset(path: str, mtime: float) -> pd.DataFrame:
    """`mtime` is part of the cache key so the page refreshes when files are regenerated."""
    return checks.load_raw(path)


@st.cache_data(show_spinner="Running data-quality checks...")
def cached_report(path: str, mtime: float, iqr_multiplier: float):
    cfg = copy.deepcopy(load_config())
    cfg["outlier_detection"]["iqr_multiplier"] = iqr_multiplier
    return checks.run_all(load_dataset(path, mtime), cfg)


def score_color(score: float) -> str:
    return "#16a34a" if score >= 97 else WARN if score >= 93 else BAD


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str = ACCENT, suffix: str = "") -> go.Figure:
    fig = go.Figure(go.Bar(x=df[x], y=df[y], marker_color=color, text=df[x].map(lambda v: f"{v:g}{suffix}"),
                           textposition="outside", orientation="h", cliponaxis=False))
    fig.update_layout(title=title, height=max(260, 32 * len(df) + 90), margin=dict(l=10, r=30, t=50, b=10),
                      yaxis=dict(autorange="reversed"), xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,.2)"),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def gauge(score: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score, title={"text": title},
                                 number={"suffix": " / 100", "valueformat": ".1f"},
                                 gauge={"axis": {"range": [80, 100]}, "bar": {"color": score_color(score)},
                                        "steps": [{"range": [80, 93], "color": "rgba(220,38,38,.12)"},
                                                  {"range": [93, 97], "color": "rgba(217,119,6,.12)"},
                                                  {"range": [97, 100], "color": "rgba(22,163,74,.12)"}]}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ---- identity, roles and review store (demo) -----------------------------------------------
import os  # noqa: E402

from src.guardrails import roles  # noqa: E402
from src.review.store import ReviewStore  # noqa: E402


def get_store() -> ReviewStore:
    cfg = load_config()
    path = os.getenv("PHARMAGUARD_DB_PATH") or ROOT / cfg["review"]["db_path"]
    return ReviewStore(path, allow_self_review=cfg["guardrails"]["allow_self_review"])


def identity_sidebar() -> tuple[str, str]:
    """Demo identity picker shared by pages. Returns (name, role_key)."""
    ss = st.session_state
    ss.setdefault("user_name", "Demo Analyst")
    ss.setdefault("user_role", "quality_analyst")
    with st.sidebar:
        st.header("Signed in (demo)")
        name = st.text_input("Your name", value=ss["user_name"], max_chars=60, key="_w_user_name")
        keys = list(roles.ROLES)
        role = st.selectbox("Role", keys, index=keys.index(ss["user_role"]), format_func=roles.ROLES.get, key="_w_user_role")
        st.caption("Demo only - there is no authentication. In production, identity and role come from SSO/IAM.")
    ss["user_name"], ss["user_role"] = name.strip(), role
    return ss["user_name"], role


def get_recorder():
    from src.observability.recorder import get_recorder as _gr
    return _gr()


def log_security_event(event_type: str, source: str, detail: str, severity: str = "medium") -> None:
    """Record an access denial / policy violation for the monitoring dashboard. Never raises."""
    try:
        get_recorder().store.record_security_event(event_type, severity, st.session_state.get("user_name", ""),
                                                   st.session_state.get("user_role", ""), source, detail)
    except Exception:  # noqa: BLE001 - monitoring must not break the page
        pass
