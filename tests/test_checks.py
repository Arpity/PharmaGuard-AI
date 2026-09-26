import numpy as np
import pandas as pd

from src.data_quality import checks


def test_missing_report_counts_and_percent(mk):
    make_df, row = mk
    df = make_df([row(Batch_ID="B00001", pH="N/A"), row(Batch_ID="B00002", pH=""),
                  row(Batch_ID="B00003", pH=np.nan), row(Batch_ID="B00004")])
    rep = checks.missing_report(df).set_index("column")
    assert rep.loc["pH", "missing_count"] == 3
    assert rep.loc["pH", "missing_pct"] == 75.0
    assert rep.loc["Assay", "missing_count"] == 0


def test_duplicate_report_separates_exact_and_near(mk):
    make_df, row = mk
    a = row(Batch_ID="B00001")
    df = make_df([a, dict(a), row(Batch_ID="B00001", Temperature="23.0"), row(Batch_ID="B00002")])
    d = checks.duplicate_report(df)
    assert d["exact_duplicate_rows"] == 1
    assert d["duplicate_batch_id_rows"] == 2
    assert d["near_duplicate_rows"] == 1


def test_invalid_range_report(mk, cfg):
    make_df, row = mk
    df = make_df([row(Batch_ID="B00001", pH="15"), row(Batch_ID="B00002", Humidity="130%"),
                  row(Batch_ID="B00003", Impurity_Level="-1"), row(Batch_ID="B00004")])
    rep = checks.invalid_range_report(df, cfg).set_index("column")
    assert rep.loc["pH", "invalid_count"] == 1
    assert rep.loc["Humidity", "invalid_count"] == 1
    assert rep.loc["Impurity_Level", "invalid_count"] == 1
    assert rep.loc["Temperature", "invalid_count"] == 0


def test_outliers_exclude_invalid_values_and_flag_extremes(mk, cfg):
    make_df, row = mk
    rows = [row(Batch_ID=f"B{i:05d}", Temperature=str(22 + (i % 5) * 0.5)) for i in range(1, 41)]
    rows.append(row(Batch_ID="B00041", Temperature="60"))    # extreme but possible -> outlier
    rows.append(row(Batch_ID="B00042", Temperature="999"))   # impossible -> invalid, not outlier
    df = make_df(rows)
    out = checks.outlier_report(df, cfg).set_index("column")
    assert out.loc["Temperature", "outlier_count"] == 1
    assert checks.invalid_range_report(df, cfg).set_index("column").loc["Temperature", "invalid_count"] == 1


def test_type_report_flags_text_numbers_and_dates(mk):
    make_df, row = mk
    df = make_df([row(Batch_ID="B00001", Temperature="22.4C", Manufacturing_Date="05/03/2024"),
                  row(Batch_ID="B00002", Deviation_Count="two"), row(Batch_ID="B00003")])
    t = checks.type_report(df).set_index("column")
    assert t.loc["Temperature", "non_conforming"] == 1
    assert t.loc["Manufacturing_Date", "non_conforming"] == 1
    assert t.loc["Deviation_Count", "non_conforming"] == 1
    assert t.loc["pH", "non_conforming"] == 0


def test_category_report_and_variants(mk):
    make_df, row = mk
    df = make_df([row(Batch_ID="B00001", Plant="plant a"), row(Batch_ID="B00002", Plant="Plant_A"),
                  row(Batch_ID="B00003", Plant="Nowhere"), row(Batch_ID="B00004", Quality_Status="PASSED")])
    c = checks.category_report(df).set_index("column")
    assert c.loc["Plant", "non_canonical"] == 2 and c.loc["Plant", "unmapped"] == 1
    assert c.loc["Quality_Status", "non_canonical"] == 1
    v = checks.category_variants(df, "Plant")
    assert set(v["raw_value"]) == {"plant a", "Plant_A", "Nowhere"}


def test_score_perfect_data_is_100_and_dirty_data_is_lower(mk, cfg):
    make_df, row = mk
    clean = make_df([row(Batch_ID=f"B{i:05d}") for i in range(1, 21)])
    assert checks.quality_score(clean, cfg)["overall"] == 100.0
    dirty = clean.copy()
    dirty.loc[0, "pH"], dirty.loc[1, "Plant"], dirty.loc[2, "Temperature"] = "N/A", "plant a", "22.5C"
    dirty.loc[3, "Batch_ID"] = "B00001"
    s = checks.quality_score(dirty, cfg)
    assert s["overall"] < 100
    assert all(0 <= v <= 100 for v in s["dimensions"].values())
    assert s["issue_counts"] == {"missing_cells": 1, "duplicate_rows": 1, "invalid_values": 0,
                                 "outliers": 0, "type_issues": 1, "category_inconsistencies": 1}


def test_outlier_fences_do_not_collapse_when_iqr_is_zero(mk, cfg):
    make_df, row = mk
    temps = ["22.5"] * 20 + ["22.0"] * 4 + ["23.0"] * 4        # >50% identical -> IQR == 0
    rows = [row(Batch_ID=f"B{i:05d}", Temperature=t) for i, t in enumerate(temps, 1)]
    out = checks.outlier_report(make_df(rows), cfg).set_index("column")
    assert out.loc["Temperature", "outlier_count"] == 0


def test_score_weights_sum_to_one(cfg):
    assert abs(sum(cfg["score_weights"].values()) - 1.0) < 1e-9


def test_checks_handle_empty_frame(cfg):
    empty = pd.DataFrame(columns=["Batch_ID", "Product_Name", "Temperature", "Plant"])
    rep = checks.run_all(empty, cfg)
    assert rep.rows == 0 and rep.duplicates["exact_duplicate_rows"] == 0
    assert 0 <= rep.score["overall"] <= 100
