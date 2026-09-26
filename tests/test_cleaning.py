import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import PROJECT_ROOT
from src.cleaning import pipeline
from src.cleaning.audit import AUDIT_COLUMNS
from src.data_quality import checks


def _clean(df, cfg):
    return pipeline.clean(df, cfg)


def test_input_frame_not_mutated(mk, cfg):
    make_df, row = mk
    df = make_df([row(Plant="plant a", Temperature="22C")])
    snapshot = df.copy()
    _clean(df, cfg)
    pd.testing.assert_frame_equal(df, snapshot)


def test_types_and_categories_standardised_and_logged(mk, cfg):
    make_df, row = mk
    df = make_df([row(Batch_ID="B00001", Plant="plant a", Temperature="22.4C", Quality_Status="PASSED",
                      Manufacturing_Date="05/03/2024", Deviation_Count="two")])
    out, audit = _clean(df, cfg)
    r = out.iloc[0]
    assert (r["Plant"], r["Temperature"], r["Quality_Status"], r["Manufacturing_Date"], r["Deviation_Count"]) == \
           ("Plant_A", 22.4, "Pass", "2024-03-05", 2)
    plant = audit[audit["column"] == "Plant"].iloc[0]
    assert plant["original_value"] == "plant a" and plant["new_value"] == "Plant_A"
    assert list(audit.columns) == AUDIT_COLUMNS
    assert audit["timestamp"].str.len().gt(0).all() and audit["reason"].str.len().gt(0).all()


def test_duplicates_keep_most_complete_row(mk, cfg):
    make_df, row = mk
    a = row(Batch_ID="B00001", Assay="N/A")
    b = row(Batch_ID="B00001", Assay="99.5")           # more complete -> kept
    c = row(Batch_ID="B00002")
    out, audit = _clean(make_df([a, b, dict(c), dict(c)]), cfg)
    assert sorted(out["Batch_ID"]) == ["B00001", "B00002"] and out["Batch_ID"].is_unique
    assert out.loc[out["Batch_ID"] == "B00001", "Assay"].iloc[0] == 99.5
    issues = audit[audit["step"] == "4_duplicates"]["issue"].tolist()
    assert sorted(issues) == ["Duplicate Batch_ID with conflicting values", "Exact duplicate row"]


def test_invalid_values_become_missing_and_critical_attributes_not_imputed(mk, cfg):
    make_df, row = mk
    rows = [row(Batch_ID=f"B{i:05d}") for i in range(1, 6)]
    rows[0].update(pH="15", Assay="180")
    out, audit = _clean(make_df(rows), cfg)
    bad = out[out["Batch_ID"] == "B00001"].iloc[0]
    assert bad["pH"] == pytest.approx(6.5)          # process parameter -> product median
    assert np.isnan(bad["Assay"])                    # critical quality attribute -> stays missing
    assert bad["Has_Missing_CQA"] and "pH" in bad["Imputed_Columns"]
    inv = audit[audit["issue"] == "Invalid range value"]
    assert set(inv["column"]) == {"pH", "Assay"} and set(inv["original_value"]) == {"15", "180"}


def test_outliers_are_flagged_not_changed(mk, cfg):
    make_df, row = mk
    rows = [row(Batch_ID=f"B{i:05d}", Temperature=str(22 + (i % 5) * 0.5)) for i in range(1, 41)]
    rows.append(row(Batch_ID="B00041", Temperature="60"))
    out, audit = _clean(make_df(rows), cfg)
    o = out[out["Batch_ID"] == "B00041"].iloc[0]
    assert o["Temperature"] == 60 and o["Outlier_Flag"] and o["Outlier_Columns"] == "Temperature"
    assert out["Outlier_Flag"].sum() == 1


def test_missing_categoricals_derived_or_unknown(mk, cfg):
    make_df, row = mk
    df = make_df([row(Batch_ID="B00001", Dosage_Form="", Plant="N/A", Equipment_ID="EQ-C05"),
                  row(Batch_ID="B00002", Product_Name="", Operator_Shift="null")])
    out = _clean(df, cfg)[0].set_index("Batch_ID")
    assert out.loc["B00001", "Dosage_Form"] == "Tablet" and out.loc["B00001", "Plant"] == "Plant_C"
    assert out.loc["B00002", "Product_Name"] == "Unknown" and out.loc["B00002", "Operator_Shift"] == "Unknown"


def test_cleaned_output_passes_quality_checks(mk, cfg):
    make_df, row = mk
    rows = [row(Batch_ID=f"B{i:05d}") for i in range(1, 31)]
    rows[0].update(Plant="PLANT-B", Humidity="130", Temperature="22C")
    rows[1].update(Manufacturing_Date="March 05, 2024", Quality_Status="failed")
    rows.append(dict(rows[2]))
    out, _ = _clean(make_df(rows), cfg)
    s = checks.quality_score(out.astype(object), cfg)
    assert s["issue_counts"]["duplicate_rows"] == 0
    assert s["issue_counts"]["invalid_values"] == 0
    assert s["issue_counts"]["type_issues"] == 0
    assert s["issue_counts"]["category_inconsistencies"] == 0


def test_output_guard_refuses_raw_dir():
    with pytest.raises(PermissionError):
        pipeline._guard_output(PROJECT_ROOT / "data" / "raw" / "x.csv")
    pipeline._guard_output(PROJECT_ROOT / "data" / "processed" / "x.csv")


def test_full_run_never_modifies_raw_and_writes_processed(cfg, tmp_path, monkeypatch):
    raw = PROJECT_ROOT / cfg["dataset"]["raw_path"]
    if not raw.exists():
        pytest.skip("raw dataset not generated")
    digest = lambda: hashlib.md5(raw.read_bytes()).hexdigest()
    before = digest()
    cfg = {**cfg, "processed": {"clean_path": "data/processed/_t_clean.csv", "audit_path": "data/processed/_t_audit.csv",
                                "summary_path": "data/processed/_t_summary.json"}}
    try:
        summary = pipeline.run_cleaning(cfg)
    finally:
        for f in ("_t_clean.csv", "_t_audit.csv", "_t_summary.json"):
            p = PROJECT_ROOT / "data" / "processed" / f
            if p.exists() and f != "":
                clean_ok = p.stat().st_size > 0
                p.unlink()
    assert digest() == before
    assert summary["rows_raw"] == 3000 and summary["rows_clean"] < 3000 and clean_ok
    assert summary["score_after"]["overall"] > summary["score_before"]["overall"]


def test_cleaning_is_deterministic(mk, cfg):
    make_df, row = mk
    rows = [row(Batch_ID=f"B{i:05d}", Plant="plant a" if i % 2 else "Plant_B", pH="N/A" if i % 5 == 0 else "6.5") for i in range(1, 31)]
    a, la = _clean(make_df(rows), cfg)
    b, lb = _clean(make_df(rows), cfg)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(la.drop(columns="timestamp"), lb.drop(columns="timestamp"))


def test_audit_log_reconciles_with_row_and_cell_changes(mk, cfg):
    make_df, row = mk
    a = row(Batch_ID="B00001")
    rows = [a, dict(a), row(Batch_ID="B00002", Plant="PLANT-B", Temperature="22.4C", Humidity="999")] + \
           [row(Batch_ID=f"B{i:05d}") for i in range(3, 20)]
    raw = make_df(rows)
    out, audit = _clean(raw, cfg)
    assert len(raw) - len(out) == (audit["action"] == "Row removed").sum() == 1
    b2 = audit[(audit["batch_id"] == "B00002") & audit["step"].isin(["2_data_types", "3_categories"])].set_index("column")
    assert b2.loc["Plant", "original_value"] == "PLANT-B" and b2.loc["Temperature", "original_value"] == "22.4C"
    assert set(audit[(audit["batch_id"] == "B00002") & (audit["column"] == "Humidity")]["action"]) == {
        "Set to missing", "Imputed with product median"}
    assert set(audit["step"]) <= {f"{i}_{n}" for i, n in enumerate(
        ["", "missing_tokens", "data_types", "categories", "duplicates", "invalid_ranges", "outliers", "imputation", "categorical_fill"])}


def test_second_cleaning_pass_finds_nothing_left_to_fix(mk, cfg):
    make_df, row = mk
    rows = [row(Batch_ID=f"B{i:05d}", Plant="plant a", Humidity="130", Manufacturing_Date="05/03/2024") for i in range(1, 25)]
    once, _ = _clean(make_df(rows), cfg)
    again, audit2 = _clean(once.drop(columns=["Source_Row", "Outlier_Flag", "Outlier_Columns", "Imputed_Columns", "Has_Missing_CQA"])
                           .astype(str).replace({"nan": "", "<NA>": ""}), cfg)
    issues = set(audit2["issue"])
    assert not issues & {"Inconsistent category", "Invalid range value", "Inconsistent date format", "Exact duplicate row"}
