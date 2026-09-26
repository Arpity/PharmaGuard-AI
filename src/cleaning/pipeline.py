"""Controlled cleaning pipeline.

Business rules (each is logged to the audit trail):
 1. Text placeholders ('N/A', 'null', '-', ...) become true missing values.
 2. Numbers stored as text are converted (units / thousands separators / number words stripped);
    dates are parsed from any known format to ISO YYYY-MM-DD. Uninterpretable values -> missing.
 3. Category variants are mapped to the canonical master-data value; unmappable -> missing.
 4. Duplicate Batch_IDs are collapsed to ONE record: the most complete row (ties: earliest).
 5. Invalid values (physically impossible) are set to missing - never "fixed" by guessing.
 6. Outliers (possible but extreme) are KEPT and FLAGGED: they may be genuine process
    excursions, which is exactly what a quality team must see.
 7. Process parameters are imputed with the product median; critical quality attributes
    (assay, dissolution, impurity, yield, deviations) are NEVER imputed (no fabricated results).
 8. Dosage_Form is derived from Product_Name, Plant from Equipment_ID; anything still missing
    in a categorical column becomes 'Unknown'.
The raw file is only ever read; writing anywhere under data/raw/ is refused.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROJECT_ROOT, load_config
from src.data_quality import checks
from src.data_quality.rules import (CANONICALISERS, CATEGORICAL_COLS, DATA_COLS, DATE_COL, ID_COL,
                                    INTEGER_COLS, NUMERIC_COLS, PRODUCT_TO_FORM, UNKNOWN, canon_batch_id,
                                    canonicalise, is_missing, parse_dates, parse_numeric)

from .audit import AuditLog

FLAG_COLS = ["Outlier_Flag", "Outlier_Columns", "Imputed_Columns", "Has_Missing_CQA"]


def normalize_missing(df: pd.DataFrame, audit: AuditLog) -> pd.DataFrame:
    step = "1_missing_tokens"
    for col in DATA_COLS:
        miss = is_missing(df[col])
        explicit = miss & df[col].notna() & (df[col].astype(str).str.strip() != "")
        audit.log(df, explicit, col, df[col], np.nan, step, "Missing value stored as text",
                  "Converted to null", "Placeholder text is not a real value; standardised to null")
        df[col] = df[col].where(~miss, np.nan)
    return df


def fix_types(df: pd.DataFrame, audit: AuditLog) -> pd.DataFrame:
    step = "2_data_types"
    for col in NUMERIC_COLS:
        p = parse_numeric(df[col], integer=col in INTEGER_COLS)
        conv, bad = p["status"] == "converted", p["status"] == "unparseable"
        audit.log(df, conv, col, df[col], p["value"], step, "Number stored as text", "Converted to number",
                  "Units, separators, number words or decimal-point counts normalised")
        audit.log(df, bad, col, df[col], np.nan, step, "Unparseable numeric value", "Set to missing",
                  "Value cannot be interpreted as a number")
        df[col] = p["value"]
    p = parse_dates(df[DATE_COL])
    audit.log(df, p["status"] == "converted", DATE_COL, df[DATE_COL], p["value"], step,
              "Inconsistent date format", "Parsed to ISO date", "Standardised to YYYY-MM-DD")
    audit.log(df, p["status"] == "unparseable", DATE_COL, df[DATE_COL], np.nan, step,
              "Unparseable date", "Set to missing", "Date does not match any known format")
    df[DATE_COL] = p["value"]
    return df


def standardize_categories(df: pd.DataFrame, audit: AuditLog) -> pd.DataFrame:
    step = "3_categories"
    bid = canonicalise(df[ID_COL], canon_batch_id)
    audit.log(df, bid.notna() & (bid != df[ID_COL]), ID_COL, df[ID_COL], bid, step,
              "Inconsistent Batch_ID format", "Standardised", "Batch_ID upper-cased / trimmed")
    audit.log(df, bid.isna() & df[ID_COL].notna(), ID_COL, df[ID_COL], np.nan, step,
              "Invalid Batch_ID", "Set to missing", "Does not match B##### pattern")
    df[ID_COL] = bid
    for col in CATEGORICAL_COLS:
        canon = canonicalise(df[col], CANONICALISERS[col])
        present = df[col].notna()
        audit.log(df, present & canon.notna() & (canon != df[col]), col, df[col], canon, step,
                  "Inconsistent category", "Mapped to canonical value",
                  "Case / spelling / abbreviation variants unified using master data")
        audit.log(df, present & canon.isna(), col, df[col], np.nan, step, "Unrecognised category",
                  "Set to missing", "Value does not map to any master-data entry")
        df[col] = canon
    return df


def remove_duplicates(df: pd.DataFrame, audit: AuditLog) -> pd.DataFrame:
    step = "4_duplicates"
    has_id = df[ID_COL].notna()
    order = pd.DataFrame({"n_missing": df[DATA_COLS].isna().sum(axis=1), "row": df.index},
                         index=df.index).sort_values(["n_missing", "row"])
    keeper_idx = order.index[has_id.reindex(order.index)]
    keepers = df.loc[keeper_idx].drop_duplicates(subset=ID_COL, keep="first")
    keeper_of = pd.Series(keepers.index, index=keepers[ID_COL])
    dropped = df.index[has_id & ~df.index.isin(keepers.index)]
    if len(dropped):
        dropped_df = df.loc[dropped]
        k = keeper_of.loc[dropped_df[ID_COL]].to_numpy()
        a, b = dropped_df[DATA_COLS].reset_index(drop=True), df.loc[k, DATA_COLS].reset_index(drop=True)
        identical = pd.Series(((a == b) | (a.isna() & b.isna())).all(axis=1).to_numpy(), index=dropped)
        kept_txt = pd.Series([f"kept source_row {x}" for x in k], index=dropped)
        marker = pd.Series("row removed", index=dropped)
        audit.log(dropped_df, identical, "*", marker, kept_txt, step, "Exact duplicate row", "Row removed",
                  "Identical to another record with the same Batch_ID")
        audit.log(dropped_df, ~identical, "*", marker, kept_txt, step,
                  "Duplicate Batch_ID with conflicting values", "Row removed (most complete record kept)",
                  "One record per batch; the row with fewest missing fields is retained")
    return df.drop(index=dropped)


def null_invalid_ranges(df: pd.DataFrame, audit: AuditLog, cfg: dict) -> pd.DataFrame:
    step = "5_invalid_ranges"
    bad = checks.invalid_mask(df[NUMERIC_COLS], cfg["valid_ranges"])
    for col in NUMERIC_COLS:
        lo, hi = cfg["valid_ranges"][col]
        audit.log(df, bad[col], col, df[col], np.nan, step, "Invalid range value",
                  "Set to missing", f"Outside valid range {lo}..{hi}; impossible value, not guessed")
        df.loc[bad[col], col] = np.nan
    return df


def flag_outliers(df: pd.DataFrame, audit: AuditLog, cfg: dict) -> pd.DataFrame:
    step = "6_outliers"
    mask = checks.outlier_mask(df[NUMERIC_COLS], cfg, checks.product_groups(df))
    for col in NUMERIC_COLS:
        audit.log(df, mask[col], col, df[col], df[col], step, "Statistical outlier", "Flagged, value retained",
                  "Extreme but possible; may be a genuine process excursion - review, do not delete")
    df["Outlier_Flag"] = mask.any(axis=1)
    df["Outlier_Columns"] = mask.apply(lambda r: ";".join(r.index[r]), axis=1)
    return df


def impute(df: pd.DataFrame, audit: AuditLog, cfg: dict) -> pd.DataFrame:
    step = "7_imputation"
    clean_cfg = cfg["cleaning"]
    group = df["Product_Name"].fillna(UNKNOWN)
    df["Imputed_Columns"] = ""
    for col in clean_cfg["impute_process_columns"]:
        miss = df[col].isna()
        med = df[col].groupby(group).transform("median").fillna(df[col].median())
        audit.log(df, miss, col, df[col], med, step, "Missing process parameter", "Imputed with product median",
                  "Process parameter; product-level median is a low-risk estimate")
        df.loc[miss, "Imputed_Columns"] += col + ";"
        df.loc[miss, col] = med[miss]
    df["Imputed_Columns"] = df["Imputed_Columns"].str.rstrip(";")
    for col in clean_cfg["critical_quality_columns"]:
        n = int(df[col].isna().sum())
        audit.note(step, "Missing critical quality attribute", col, f"{n} missing", f"{n} missing",
                   "Not imputed", "Critical quality results must never be fabricated; rows flagged instead")
    df["Has_Missing_CQA"] = df[clean_cfg["critical_quality_columns"]].isna().any(axis=1)

    step = "8_categorical_fill"
    form_from_product = df["Product_Name"].map(PRODUCT_TO_FORM)
    miss = df["Dosage_Form"].isna() & form_from_product.notna()
    audit.log(df, miss, "Dosage_Form", df["Dosage_Form"], form_from_product, step, "Missing dosage form",
              "Derived from Product_Name", "Each product has exactly one dosage form (master data)")
    df.loc[miss, "Dosage_Form"] = form_from_product[miss]
    plant_from_eq = "Plant_" + df["Equipment_ID"].str.extract(r"EQ-([A-D])")[0]
    miss = df["Plant"].isna() & plant_from_eq.notna()
    audit.log(df, miss, "Plant", df["Plant"], plant_from_eq, step, "Missing plant",
              "Derived from Equipment_ID", "Equipment IDs embed the plant letter")
    df.loc[miss, "Plant"] = plant_from_eq[miss]
    for col in CATEGORICAL_COLS:
        miss = df[col].isna()
        audit.log(df, miss, col, df[col], UNKNOWN, step, "Missing category", "Filled with 'Unknown'",
                  "Cannot be derived safely; explicit label keeps the row usable")
        df.loc[miss, col] = UNKNOWN
    return df


def clean(raw: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean a raw (all-text) frame. Returns (clean_df, audit_df). Does not touch the input."""
    cfg = cfg or load_config()
    audit = AuditLog()
    df = raw[DATA_COLS].copy().astype(object)
    df.index = raw.index
    for step in (normalize_missing, fix_types, standardize_categories, remove_duplicates):
        df = step(df, audit)
    df = null_invalid_ranges(df, audit, cfg)
    df = flag_outliers(df, audit, cfg)
    df = impute(df, audit, cfg)

    df.insert(0, "Source_Row", df.index)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.strftime("%Y-%m-%d")
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col]).astype("Int64") if col in INTEGER_COLS else pd.to_numeric(df[col])
    return df.sort_values(DATE_COL).reset_index(drop=True), audit.to_frame()


def _guard_output(path: Path) -> None:
    raw_dir = (PROJECT_ROOT / "data" / "raw").resolve()
    if raw_dir in path.resolve().parents or path.resolve() == raw_dir:
        raise PermissionError(f"Refusing to write inside data/raw/: {path}")


def run_cleaning(cfg: dict | None = None) -> dict:
    """Read raw CSV, clean, write data/processed/* and return a summary dict."""
    cfg = cfg or load_config()
    raw = checks.load_raw(PROJECT_ROOT / cfg["dataset"]["raw_path"])
    cleaned, audit = clean(raw, cfg)

    out = {k: PROJECT_ROOT / cfg["processed"][k] for k in ("clean_path", "audit_path", "summary_path")}
    for p in out.values():
        _guard_output(p)
        p.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out["clean_path"], index=False)
    audit.to_csv(out["audit_path"], index=False)

    before = checks.quality_score(raw, cfg)
    after = checks.quality_score(cleaned.astype(object), cfg)
    summary = {"rows_raw": len(raw), "rows_clean": len(cleaned), "audit_entries": len(audit),
               "score_before": before, "score_after": after,
               "outlier_rows_flagged": int(cleaned["Outlier_Flag"].sum()),
               "rows_missing_cqa": int(cleaned["Has_Missing_CQA"].sum()),
               "changes_by_issue": audit["issue"].value_counts().to_dict()}
    out["summary_path"].write_text(json.dumps(summary, indent=2))
    return summary
