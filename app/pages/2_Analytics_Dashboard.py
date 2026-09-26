"""Analytics dashboard: KPIs, interactive Plotly charts, filters and business insights."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from app import components as ui  # noqa: E402
from src.analytics import kpis  # noqa: E402

st.set_page_config(page_title="Analytics Dashboard | PharmaGuard AI", page_icon="📊", layout="wide")

STATUS_COLORS = {"Pass": "#16a34a", "Under Review": "#f59e0b", "Fail": "#dc2626", "Unknown": "#94a3b8"}
BLUE, TEMPLATE = "#2563eb", "plotly_white"

st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
[data-testid="stMetric"] {background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:14px 16px;
  box-shadow:0 1px 2px rgba(15,23,42,.05);}
[data-testid="stMetricLabel"] p {color:#64748b; font-size:.85rem;}
.insight {border-left:4px solid #94a3b8; background:#f8fafc; padding:10px 14px; margin:8px 0; border-radius:0 8px 8px 0;}
.insight.high {border-color:#dc2626; background:#fef2f2;} .insight.medium {border-color:#f59e0b; background:#fffbeb;}
.insight.low {border-color:#16a34a; background:#f0fdf4;} .insight.info {border-color:#2563eb; background:#eff6ff;}
.insight b {color:#0f172a;}
</style>""", unsafe_allow_html=True)

cfg = ui.get_config()
clean_path = ui.ROOT / cfg["processed"]["clean_path"]
st.title("📊 Analytics Dashboard")
st.caption("Batch quality performance from the cleaned dataset. All figures are calculated with deterministic pandas code.")

if not clean_path.exists():
    st.warning("Cleaned data not found. Run `python scripts/run_cleaning.py` or use the Data Quality page first.")
    st.stop()


@st.cache_data(show_spinner=False)
def _load(path: str, mtime: float) -> pd.DataFrame:
    return kpis.load_clean(path)


full = _load(str(clean_path), clean_path.stat().st_mtime)

# Filters ---------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    products = st.multiselect("Product", sorted(full["Product_Name"].unique()), placeholder="All products")
    plants = st.multiselect("Plant", sorted(full["Plant"].unique()), placeholder="All plants")
    statuses = st.multiselect("Quality status", [s for s in kpis.STATUS_ORDER if s in set(full["Quality_Status"])],
                              placeholder="All statuses")
    dmin, dmax = full["Manufacturing_Date"].min().date(), full["Manufacturing_Date"].max().date()
    drange = st.date_input("Manufacturing date", (dmin, dmax), min_value=dmin, max_value=dmax)
    if len(drange) != 2:
        drange = (dmin, dmax)
    st.caption("Empty selection = all values.")

df = kpis.apply_filters(full, products or None, plants or None, statuses or None, drange,
                        include_undated=(drange == (dmin, dmax)))
st.caption(f"Showing **{len(df):,}** of {len(full):,} batches")
if df.empty:
    st.info("No batches match the selected filters.")
    st.stop()

# KPI cards -------------------------------------------------------------------------------------
k = kpis.compute_kpis(df)
r1 = st.columns(4)
r1[0].metric("Total Batches", f"{k['total_batches']:,}")
r1[1].metric("Pass / Risk Batches", f"{k['pass_batches']:,} / {k['risk_batches']:,}",
             f"{k['review_batches']:,} under review · {k['fail_batches']:,} failed", delta_color="off")
r1[2].metric("Quality Failure Rate", f"{k['failure_rate']:.1f}%", f"Risk rate {k['risk_rate']:.1f}%", delta_color="off",
             help="Failed batches ÷ batches with a known status. Risk = Under Review + Fail.")
r1[3].metric("Deviations", f"{k['total_deviations']:,}", f"{k['avg_deviations']:.2f} per batch", delta_color="off")
r2 = st.columns(4)
r2[0].metric("Average Dissolution", f"{k['avg_dissolution']:.1f}%")
r2[1].metric("Average Assay", f"{k['avg_assay']:.1f}%")
r2[2].metric("Average Impurity", f"{k['avg_impurity']:.2f}%")
r2[3].metric("Average Yield", f"{k['avg_yield']:.1f}%")
if k["unknown_status"]:
    st.caption(f"{k['unknown_status']} batches have no recorded status and are excluded from rates. "
               "Averages ignore missing test results.")


def style(fig: go.Figure, h: int = 360) -> go.Figure:
    fig.update_layout(template=TEMPLATE, height=h, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.2), title_font_size=15)
    return fig


def show(fig: go.Figure, h: int = 360) -> None:
    st.plotly_chart(style(fig, h), use_container_width=True)


def status_stack(summary: pd.DataFrame, by: str, title: str) -> go.Figure:
    long = summary.melt(id_vars=[by], value_vars=["Pass", "Under Review", "Fail"], var_name="Status", value_name="Batches")
    fig = px.bar(long, x=by, y="Batches", color="Status", color_discrete_map=STATUS_COLORS, title=title,
                 category_orders={"Status": ["Pass", "Under Review", "Fail"]})
    return fig


def rate_bar(summary: pd.DataFrame, by: str, title: str) -> go.Figure:
    s = summary.sort_values("fail_rate", ascending=True)
    fig = px.bar(s, x="fail_rate", y=by, orientation="h", title=title, text=s["fail_rate"].map(lambda v: f"{v:.1f}%"),
                 labels={"fail_rate": "Failure rate (%)"}, color_discrete_sequence=[STATUS_COLORS["Fail"]])
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


st.divider()
tabs = st.tabs(["Overview", "Product & Plant", "Quality attributes", "Deviations", "Trends", "Anomalies"])

# Overview --------------------------------------------------------------------------------------
with tabs[0]:
    a, b = st.columns([1, 2])
    counts = df["Quality_Status"].value_counts().reindex(kpis.STATUS_ORDER).dropna().reset_index()
    counts.columns = ["Status", "Batches"]
    fig = px.pie(counts, names="Status", values="Batches", hole=0.55, color="Status",
                 color_discrete_map=STATUS_COLORS, title="Quality status mix")
    fig.update_traces(sort=False, textinfo="percent+label")
    with a:
        show(fig.update_layout(showlegend=False))
    with b:
        sc = kpis.spec_compliance(df, cfg["specs"])
        fig = px.bar(sc.sort_values("in_spec_pct"), x="in_spec_pct", y="attribute", orientation="h",
                     text=sc.sort_values("in_spec_pct")["in_spec_pct"].map(lambda v: f"{v:.1f}%"),
                     title="Share of tested batches within specification", labels={"in_spec_pct": "In spec (%)"},
                     color_discrete_sequence=[BLUE])
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(range=[0, 108])
        show(fig)

# Product & plant -------------------------------------------------------------------------------
with tabs[1]:
    ps, ls = kpis.group_summary(df, "Product_Name"), kpis.group_summary(df, "Plant")
    a, b = st.columns(2)
    with a:
        show(status_stack(ps.sort_values("batches", ascending=False), "Product_Name", "Batches by product and status"))
    with b:
        show(rate_bar(ps, "Product_Name", "Failure rate by product"))
    a, b = st.columns(2)
    with a:
        show(status_stack(ls, "Plant", "Batches by plant and status"))
    with b:
        heat = df[df["Operator_Shift"] != "Unknown"].groupby(["Plant", "Operator_Shift"])["Quality_Status"] \
            .apply(lambda x: (x == "Fail").sum() / max((x != "Unknown").sum(), 1) * 100).unstack()
        fig = px.imshow(heat.reindex(columns=[c for c in ["Morning", "Evening", "Night"] if c in heat.columns]),
                        text_auto=".1f", color_continuous_scale="Reds", aspect="auto",
                        title="Failure rate (%) by plant and shift", labels={"color": "Fail %"})
        show(fig)
    with st.expander("Summary table"):
        t = ps.rename(columns={"Product_Name": "Product"}).round(2)
        st.dataframe(t, hide_index=True, use_container_width=True)

# Quality attributes ----------------------------------------------------------------------------
with tabs[2]:
    specs = cfg["specs"]
    attrs = [("Dissolution", "Dissolution (%)"), ("Assay", "Assay (% label claim)"),
             ("Impurity_Level", "Impurity level (%)"), ("Yield_Percentage", "Yield (%)")]
    for row in (attrs[:2], attrs[2:]):
        cols = st.columns(2)
        for c, (col, label) in zip(cols, row):
            with c:
                d = df.dropna(subset=[col])
                fig = px.histogram(d, x=col, nbins=40, color_discrete_sequence=[BLUE], title=f"{label} distribution",
                                   labels={col: label})
                lo, hi = specs[col]
                for v, nm in ((lo, "Lower spec"), (hi, "Upper spec")):
                    fig.add_vline(x=v, line_dash="dash", line_color="#dc2626", annotation_text=nm,
                                  annotation_position="top")
                show(fig, 320)
    a = st.selectbox("Compare by", ["Plant", "Product_Name", "Operator_Shift"], key="cmp",
                     format_func=lambda v: v.replace("_", " "))
    m = st.selectbox("Attribute", [x[0] for x in attrs], key="cmpm", format_func=lambda v: v.replace("_", " "))
    d = df.dropna(subset=[m])
    fig = px.box(d, x=a, y=m, color_discrete_sequence=[BLUE], points=False, title=f"{m.replace('_', ' ')} by {a.replace('_', ' ')}")
    show(fig, 380)

# Deviations ------------------------------------------------------------------------------------
with tabs[3]:
    a, b = st.columns(2)
    d = df.dropna(subset=["Deviation_Count"])
    with a:
        dist = d["Deviation_Count"].astype(int).clip(upper=8).value_counts().sort_index().reset_index()
        dist.columns = ["Deviations", "Batches"]
        dist["Deviations"] = dist["Deviations"].map(lambda v: "8+" if v == 8 else str(v))
        show(px.bar(dist, x="Deviations", y="Batches", title="Batches by deviation count", color_discrete_sequence=[BLUE]))
    with b:
        by = d.groupby("Quality_Status")["Deviation_Count"].mean().reindex(kpis.STATUS_ORDER).dropna().reset_index()
        fig = px.bar(by, x="Quality_Status", y="Deviation_Count", color="Quality_Status", color_discrete_map=STATUS_COLORS,
                     title="Average deviations per batch by quality status", labels={"Deviation_Count": "Avg deviations"},
                     text=by["Deviation_Count"].map(lambda v: f"{v:.2f}"))
        show(fig.update_layout(showlegend=False))
    a, b = st.columns(2)
    for c, dim in ((a, "Plant"), (b, "Product_Name")):
        with c:
            g = kpis.group_summary(df, dim).sort_values("avg_Deviation_Count", ascending=False)
            fig = px.bar(g, x=dim, y="avg_Deviation_Count", title=f"Average deviations by {dim.replace('_', ' ').lower()}",
                         labels={"avg_Deviation_Count": "Avg deviations"}, color_discrete_sequence=[BLUE])
            show(fig)

# Trends ----------------------------------------------------------------------------------------
with tabs[4]:
    tr = kpis.monthly_trend(df)
    if len(tr) < 2:
        st.info("Not enough months in the current selection to show a trend.")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=tr["month"], y=tr["batches"], name="Batches", marker_color="#bfdbfe", secondary_y=False)
        fig.add_scatter(x=tr["month"], y=tr["fail_rate"], name="Failure rate (%)", mode="lines+markers",
                        line=dict(color=STATUS_COLORS["Fail"]), secondary_y=True)
        fig.add_scatter(x=tr["month"], y=tr["risk_rate"], name="Risk rate (%)", mode="lines",
                        line=dict(color=STATUS_COLORS["Under Review"], dash="dot"), secondary_y=True)
        fig.update_layout(title="Monthly batch volume and quality rates")
        fig.update_yaxes(title_text="Batches", secondary_y=False)
        fig.update_yaxes(title_text="Rate (%)", secondary_y=True)
        show(fig, 400)
        metric = st.selectbox("Monthly average of", ["Assay", "Dissolution", "Impurity_Level", "Yield_Percentage",
                                                     "Deviation_Count"], format_func=lambda v: v.replace("_", " "))
        col = f"avg_{metric}"
        fig = px.line(tr, x="month", y=col, markers=True, title=f"Monthly average {metric.replace('_', ' ').lower()}",
                      labels={col: "Average", "month": "Month"}, color_discrete_sequence=[BLUE])
        if metric in specs:
            lo, hi = specs[metric]
            fig.add_hrect(y0=lo, y1=hi, fillcolor="#16a34a", opacity=0.08, line_width=0, annotation_text="Specification",
                          annotation_position="top left")
        show(fig, 360)

# Anomalies -------------------------------------------------------------------------------------
with tabs[5]:
    st.caption("Anomalies are batches with statistically extreme readings (IQR method from the Data Quality step). "
               "They are retained for review, not removed.")
    an = kpis.anomaly_table(df)
    a, b = st.columns(2)
    a.metric("Anomalous batches", f"{len(an):,}", f"{len(an) / len(df) * 100:.1f}% of selection", delta_color="off")
    b.metric("Anomalous batches that failed", f"{int((an['Quality_Status'] == 'Fail').sum()):,}")
    d = df.dropna(subset=["Assay", "Impurity_Level"])
    fig = px.scatter(d, x="Assay", y="Impurity_Level", color=d["Outlier_Flag"].map({True: "Anomaly", False: "Normal"}),
                     color_discrete_map={"Anomaly": "#dc2626", "Normal": "#93c5fd"}, opacity=0.7,
                     hover_data=["Batch_ID", "Product_Name", "Plant", "Quality_Status"],
                     title="Assay vs impurity - anomalies highlighted", labels={"color": ""})
    show(fig, 420)
    if len(an):
        st.dataframe(an, hide_index=True, use_container_width=True)
        st.download_button("⬇ Anomalous batches (CSV)", an.to_csv(index=False).encode(), "anomalous_batches.csv", "text/csv")

# Insights --------------------------------------------------------------------------------------
st.divider()
st.subheader("Key insights")
st.caption("Generated automatically from the filtered data with fixed rules; every number is calculated, not estimated.")
for ins in kpis.generate_insights(df, cfg):
    st.markdown(f"<div class='insight {ins['level']}'><b>{ins['title']}</b><br>{ins['text']}</div>", unsafe_allow_html=True)
