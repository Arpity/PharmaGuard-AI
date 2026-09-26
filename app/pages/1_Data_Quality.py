"""Data Quality page: score, per-issue drill-downs, cleaning pipeline and audit log."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from app import components as ui  # noqa: E402
from src.cleaning.pipeline import run_cleaning  # noqa: E402
from src.data_quality import checks  # noqa: E402

st.set_page_config(page_title="Data Quality | PharmaGuard AI", page_icon="🧪", layout="wide")
cfg = ui.get_config()
raw_path = ui.ROOT / cfg["dataset"]["raw_path"]
clean_path = ui.ROOT / cfg["processed"]["clean_path"]
audit_path = ui.ROOT / cfg["processed"]["audit_path"]
summary_path = ui.ROOT / cfg["processed"]["summary_path"]

st.title("🧪 Data Quality")
st.caption("Profile the batch manufacturing data, then clean it with documented business rules.")

if not raw_path.exists():
    st.error(f"Raw dataset not found at `{raw_path.relative_to(ui.ROOT)}`. Run `python scripts/generate_data.py`.")
    st.stop()

# Sidebar ---------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    options = ["Raw data"] + (["Cleaned data"] if clean_path.exists() else [])
    source = st.radio("Dataset to inspect", options, help="Raw data is never modified.")
    mult = st.slider("Outlier fence (IQR multiplier)", 1.5, 5.0, float(cfg["outlier_detection"]["iqr_multiplier"]),
                     0.5, help="Higher = only more extreme values are flagged.")
    with st.expander("Score weights"):
        st.table(pd.Series(cfg["score_weights"], name="weight").to_frame())
        st.caption("Grades: Excellent ≥ 99 · Good ≥ 97 · Fair ≥ 93 · Poor < 93")

path = raw_path if source == "Raw data" else clean_path
mtime = path.stat().st_mtime
df = ui.load_dataset(str(path), mtime)
rep = ui.cached_report(str(path), mtime, mult)
score = rep.score

# Header KPIs -----------------------------------------------------------------------------------
left, right = st.columns([1, 2])
left.plotly_chart(ui.gauge(score["overall"], f"Data Quality Score - {score['grade']}"), use_container_width=True)
with right:
    dims = pd.DataFrame({"dimension": [k.replace("_", " ").title() for k in score["dimensions"]],
                         "score": list(score["dimensions"].values())})
    st.plotly_chart(ui.bar_chart(dims, "score", "dimension", "Score by dimension (100 = no issues)"), use_container_width=True)

ic = score["issue_counts"]
cols = st.columns(6)
for c, (label, key) in zip(cols, [("Rows", None), ("Missing cells", "missing_cells"), ("Duplicate rows", "duplicate_rows"),
                                  ("Invalid values", "invalid_values"), ("Outliers", "outliers"),
                                  ("Type / category issues", None)]):
    if label == "Rows":
        c.metric(label, f"{rep.rows:,}", f"{rep.columns} columns", delta_color="off")
    elif key:
        c.metric(label, f"{ic[key]:,}")
    else:
        c.metric(label, f"{ic['type_issues']:,} / {ic['category_inconsistencies']:,}")

tabs = st.tabs(["Missing", "Duplicates", "Outliers", "Invalid ranges", "Data types", "Categories", "Cleaning & audit"])

# Missing ---------------------------------------------------------------------------------------
with tabs[0]:
    m = rep.missing
    st.caption("Counts NaN, blanks and text placeholders (N/A, NA, null, -, Unknown).")
    st.plotly_chart(ui.bar_chart(m, "missing_pct", "column", "Missing values by column (%)", ui.WARN, "%"), use_container_width=True)
    st.dataframe(m, hide_index=True, width="stretch")

# Duplicates ------------------------------------------------------------------------------------
with tabs[1]:
    d = rep.duplicates
    a, b, c = st.columns(3)
    a.metric("Exact duplicate rows", d["exact_duplicate_rows"])
    b.metric("Extra rows sharing a Batch_ID", d["duplicate_batch_id_rows"], f"{d['duplicate_pct']}% of rows", delta_color="off")
    c.metric("Near-duplicates (values differ)", d["near_duplicate_rows"])
    ids = df["Batch_ID"].astype(str).str.strip().str.upper()
    dup_ids = ids[ids.duplicated(keep=False)]
    if len(dup_ids):
        st.caption("All rows whose Batch_ID appears more than once, grouped by Batch_ID.")
        st.dataframe(df.loc[dup_ids.index].assign(_id=dup_ids).sort_values(["_id"]).drop(columns="_id"),
                     width="stretch")
    else:
        st.success("No duplicate Batch_IDs.")

# Outliers --------------------------------------------------------------------------------------
with tabs[2]:
    st.caption("Values that are possible but statistically extreme (beyond the IQR fences). Invalid-range values are "
               "excluded here and reported separately. Cleaning keeps and flags these.")
    st.dataframe(rep.outliers, hide_index=True, width="stretch")
    col = st.selectbox("Distribution for column", checks.numeric_values(df).columns, key="out_col")
    vals = checks.numeric_values(df)[col]
    lo, hi = cfg["valid_ranges"][col]
    vals = vals[(vals >= lo) & (vals <= hi)]
    st.plotly_chart(px.box(vals.rename(col), x=col, points="outliers", height=260,
                           color_discrete_sequence=[ui.ACCENT]), use_container_width=True)

# Invalid ranges --------------------------------------------------------------------------------
with tabs[3]:
    st.caption("Values outside physically / logically possible limits (configured in `config/config.yaml`).")
    st.dataframe(rep.invalid, hide_index=True, width="stretch")

# Types -----------------------------------------------------------------------------------------
with tabs[4]:
    st.caption("Numbers stored as text (units, separators, words) and dates not in YYYY-MM-DD.")
    st.dataframe(rep.types, hide_index=True, width="stretch")

# Categories ------------------------------------------------------------------------------------
with tabs[5]:
    st.caption("Category values that differ from the canonical master-data spelling.")
    st.dataframe(rep.categories, hide_index=True, width="stretch")
    cat_col = st.selectbox("Show value mapping for", rep.categories["column"].tolist(), key="cat_col")
    st.dataframe(checks.category_variants(df, cat_col), hide_index=True, width="stretch")

# Cleaning --------------------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Cleaning business rules")
    st.markdown(
        """
1. **Placeholders → null**: `N/A`, `null`, `-` … become real missing values.
2. **Types**: numbers with units / commas / words are converted; dates parsed to `YYYY-MM-DD`.
3. **Categories** mapped to canonical master data (`plant a` → `Plant_A`, `PASSED` → `Pass`).
4. **Duplicates**: one record per Batch_ID - the most complete row is kept.
5. **Invalid ranges** (pH 15, humidity 130 %…) → set to missing; never guessed.
6. **Outliers** are **kept and flagged** (may be genuine process excursions).
7. **Imputation** only for process parameters (product median). **Critical quality attributes
   (assay, dissolution, impurity, yield, deviations) are never imputed.**
8. Dosage form derived from product, plant from equipment ID; remaining gaps → `Unknown`.
"""
    )
    st.info("Raw data is read-only. Output: `data/processed/pharma_batch_clean.csv` and `cleaning_audit_log.csv`.")
    if st.button("▶ Run cleaning pipeline", type="primary"):
        with st.spinner("Cleaning..."):
            run_cleaning(cfg)
        st.cache_data.clear()
        st.success("Cleaning complete. Cleaned data is now selectable in the sidebar.")
        st.rerun() if source != "Raw data" else None

    if summary_path.exists() and audit_path.exists():
        import json
        summ = json.loads(summary_path.read_text())
        b, a = summ["score_before"], summ["score_after"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Score before → after", f"{a['overall']:.1f}", f"{a['overall'] - b['overall']:+.1f}")
        k2.metric("Rows", f"{summ['rows_clean']:,}", f"{summ['rows_clean'] - summ['rows_raw']:+,} duplicates removed",
                  delta_color="off")
        k3.metric("Rows flagged as outlier", f"{summ['outlier_rows_flagged']:,}")
        k4.metric("Rows missing a critical result", f"{summ['rows_missing_cqa']:,}")

        cmp_df = pd.DataFrame({"dimension": [k.replace("_", " ").title() for k in b["dimensions"]] * 2,
                               "score": list(b["dimensions"].values()) + list(a["dimensions"].values()),
                               "stage": ["Before"] * len(b["dimensions"]) + ["After"] * len(a["dimensions"])})
        fig = px.bar(cmp_df, x="dimension", y="score", color="stage", barmode="group", height=320,
                     color_discrete_map={"Before": ui.MUTED, "After": ui.ACCENT})
        fig.update_yaxes(range=[80, 100])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Remaining gaps after cleaning are deliberate: critical results are not imputed and outliers are retained.")

        st.subheader("Cleaning audit log")
        audit = pd.read_csv(audit_path, dtype=str, keep_default_na=False)
        f1, f2, f3, f4 = st.columns(4)
        steps = f1.multiselect("Step", sorted(audit["step"].unique()))
        issues = f2.multiselect("Issue", sorted(audit["issue"].unique()))
        columns = f3.multiselect("Column", sorted(audit["column"].unique()))
        batch = f4.text_input("Batch_ID contains")
        view = audit
        if steps: view = view[view["step"].isin(steps)]
        if issues: view = view[view["issue"].isin(issues)]
        if columns: view = view[view["column"].isin(columns)]
        if batch: view = view[view["batch_id"].str.contains(batch, case=False)]
        st.caption(f"{len(view):,} of {len(audit):,} audit entries")
        st.dataframe(view.head(2000), hide_index=True, width="stretch")
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Audit log (CSV)", audit_path.read_bytes(), "cleaning_audit_log.csv", "text/csv")
        d2.download_button("⬇ Cleaned data (CSV)", clean_path.read_bytes(), "pharma_batch_clean.csv", "text/csv")
    else:
        st.warning("Cleaning has not been run yet.")
