"""Reusable data-quality checks. Every function takes a DataFrame (raw text or typed) and
returns a small DataFrame / dict, so results are easy to display, test and log."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .rules import (CANONICALISERS, CATEGORICAL_COLS, DATA_COLS, DATE_COL, ID_COL, INTEGER_COLS,
                    NUMERIC_COLS, canon_product, canonicalise, is_missing,
                    parse_dates, parse_numeric)


def _cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _examples(values: pd.Series, n: int = 4) -> str:
    return ", ".join(map(str, values.dropna().astype(str).unique()[:n]))


def numeric_values(df: pd.DataFrame) -> pd.DataFrame:
    """Leniently parsed numeric columns (units/separators stripped) as floats."""
    return pd.DataFrame({c: parse_numeric(df[c], integer=c in INTEGER_COLS)["value"]
                         for c in _cols(df, NUMERIC_COLS)}, index=df.index)


# 1. Missing ----------------------------------------------------------------------------
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    n = max(len(df), 1)
    rows = [{"column": c, "missing_count": int(is_missing(df[c]).sum())} for c in _cols(df, DATA_COLS)]
    out = pd.DataFrame(rows)
    out["missing_pct"] = (out["missing_count"] / n * 100).round(2)
    return out.sort_values("missing_count", ascending=False).reset_index(drop=True)


# 2. Duplicates -------------------------------------------------------------------------
def duplicate_report(df: pd.DataFrame) -> dict:
    data = df[_cols(df, DATA_COLS)].astype(str)
    exact = data.duplicated(keep="first")
    ids = df[ID_COL].astype(str).str.strip().str.upper() if ID_COL in df else pd.Series(dtype=str)
    dup_id = ids.duplicated(keep="first") & ~is_missing(df[ID_COL]) if ID_COL in df else exact & False
    return {
        "total_rows": len(df),
        "exact_duplicate_rows": int(exact.sum()),
        "duplicate_batch_id_rows": int(dup_id.sum()),
        "near_duplicate_rows": int((dup_id & ~exact).sum()),  # same Batch_ID, values differ
        "duplicate_pct": round(float(dup_id.sum()) / max(len(df), 1) * 100, 2),
        "exact_duplicate_mask": exact,
        "duplicate_batch_id_mask": dup_id,
    }


# 3. Invalid ranges ---------------------------------------------------------------------
def invalid_mask(values: pd.DataFrame, ranges: dict) -> pd.DataFrame:
    out = pd.DataFrame(False, index=values.index, columns=values.columns)
    for c in values.columns:
        lo, hi = ranges[c]
        out[c] = values[c].notna() & ((values[c] < lo) | (values[c] > hi))
    return out


def invalid_range_report(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    vals, ranges = numeric_values(df), cfg["valid_ranges"]
    bad = invalid_mask(vals, ranges)
    rows = [{"column": c, "valid_min": ranges[c][0], "valid_max": ranges[c][1],
             "invalid_count": int(bad[c].sum()),
             "invalid_pct": round(bad[c].sum() / max(len(df), 1) * 100, 2),
             "examples": _examples(vals.loc[bad[c], c])} for c in vals.columns]
    return pd.DataFrame(rows).sort_values("invalid_count", ascending=False).reset_index(drop=True)


# 4. Outliers ---------------------------------------------------------------------------
def product_groups(df: pd.DataFrame) -> pd.Series:
    if "Product_Name" not in df:
        return pd.Series(UNKNOWN_GROUP, index=df.index)
    return canonicalise(df["Product_Name"], canon_product).fillna(UNKNOWN_GROUP)


UNKNOWN_GROUP = "Unknown"


def outlier_fences(values: pd.DataFrame, cfg: dict, groups: Optional[pd.Series] = None):
    """Return (lower, upper) fence frames aligned to `values`. Invalid-range values are
    excluded when computing quartiles so data errors do not mask real outliers."""
    ranges, k = cfg["valid_ranges"], cfg["outlier_detection"]["iqr_multiplier"]
    grouped = set(cfg["outlier_detection"].get("group_by_product", []))
    valid = values.where(~invalid_mask(values, ranges))
    lower, upper = pd.DataFrame(index=values.index), pd.DataFrame(index=values.index)
    for c in values.columns:
        if c in grouped and groups is not None:
            g = valid[c].groupby(groups)
            q1, q3 = g.transform(lambda x: x.quantile(0.25)), g.transform(lambda x: x.quantile(0.75))
            fallback = g.transform("std").fillna(0)
        else:
            q1, q3 = valid[c].quantile(0.25), valid[c].quantile(0.75)
            fallback = valid[c].std() if valid[c].notna().sum() > 1 else 0.0
        iqr = q3 - q1
        iqr = iqr.where(iqr > 0, fallback) if isinstance(iqr, pd.Series) else (iqr if iqr > 0 else fallback)
        lower[c], upper[c] = q1 - k * iqr, q3 + k * iqr
    return lower, upper


def outlier_mask(values: pd.DataFrame, cfg: dict, groups: Optional[pd.Series] = None) -> pd.DataFrame:
    """True where a valid value lies beyond the IQR fences (possible but extreme)."""
    lower, upper = outlier_fences(values, cfg, groups)
    valid = ~invalid_mask(values, cfg["valid_ranges"]) & values.notna()
    return valid & ((values < lower) | (values > upper))


def outlier_report(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    vals = numeric_values(df)
    mask = outlier_mask(vals, cfg, product_groups(df))
    lower, upper = outlier_fences(vals, cfg, product_groups(df))
    grouped = set(cfg["outlier_detection"].get("group_by_product", []))
    rows = [{"column": c, "method": f"IQR x{cfg['outlier_detection']['iqr_multiplier']}",
             "lower_fence": "per product" if c in grouped else (f"{float(lower[c].iloc[0]):.3f}" if len(lower) else "n/a"),
             "upper_fence": "per product" if c in grouped else (f"{float(upper[c].iloc[0]):.3f}" if len(upper) else "n/a"),
             "outlier_count": int(mask[c].sum()),
             "outlier_pct": round(mask[c].sum() / max(len(df), 1) * 100, 2),
             "examples": _examples(vals.loc[mask[c], c])} for c in vals.columns]
    return pd.DataFrame(rows).sort_values("outlier_count", ascending=False).reset_index(drop=True)


# 5. Data types -------------------------------------------------------------------------
def type_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in _cols(df, NUMERIC_COLS + [DATE_COL]):
        p = parse_dates(df[c]) if c == DATE_COL else parse_numeric(df[c], integer=c in INTEGER_COLS)
        bad = p["status"].isin(["converted", "unparseable"])
        rows.append({"column": c, "expected_type": "date (YYYY-MM-DD)" if c == DATE_COL
                     else ("integer" if c in INTEGER_COLS else "number"),
                     "non_conforming": int(bad.sum()),
                     "unparseable": int((p["status"] == "unparseable").sum()),
                     "non_conforming_pct": round(bad.sum() / max(len(df), 1) * 100, 2),
                     "examples": _examples(df.loc[bad, c])})
    return pd.DataFrame(rows).sort_values("non_conforming", ascending=False).reset_index(drop=True)


# 6. Categorical consistency ------------------------------------------------------------
def category_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in _cols(df, CATEGORICAL_COLS):
        miss = is_missing(df[c])
        canon = canonicalise(df[c], CANONICALISERS[c])
        present = ~miss
        non_canon = present & canon.notna() & (canon != df[c])
        unmapped = present & canon.isna()
        rows.append({"column": c, "distinct_raw": int(df.loc[present, c].nunique()),
                     "distinct_canonical": int(canon.nunique()),
                     "non_canonical": int((non_canon | unmapped).sum()),
                     "unmapped": int(unmapped.sum()),
                     "non_canonical_pct": round((non_canon | unmapped).sum() / max(len(df), 1) * 100, 2),
                     "examples": _examples(df.loc[non_canon | unmapped, c])})
    return pd.DataFrame(rows).sort_values("non_canonical", ascending=False).reset_index(drop=True)


def category_variants(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Raw value -> canonical value -> count, for a drill-down table."""
    present = ~is_missing(df[column])
    canon = canonicalise(df[column], CANONICALISERS[column])
    t = pd.DataFrame({"raw_value": df.loc[present, column].astype(str),
                      "canonical": canon[present].fillna("(unmapped)")})
    return (t.groupby(["raw_value", "canonical"]).size().rename("count").reset_index()
             .sort_values(["canonical", "count"], ascending=[True, False]).reset_index(drop=True))


# 7. Score ------------------------------------------------------------------------------
def quality_score(df: pd.DataFrame, cfg: dict) -> dict:
    """Weighted 0-100 score. Each dimension = 100 * (1 - share of affected cells/rows)."""
    n = max(len(df), 1)
    cols = _cols(df, DATA_COLS)
    miss = missing_report(df)["missing_count"].sum()
    dup = duplicate_report(df)["duplicate_batch_id_rows"]
    inv = invalid_range_report(df, cfg)["invalid_count"].sum()
    outl = outlier_report(df, cfg)["outlier_count"].sum()
    typ = type_report(df)["non_conforming"].sum()
    cat = category_report(df)["non_canonical"].sum()

    num_cells = n * len(_cols(df, NUMERIC_COLS))
    typed_cells = n * len(_cols(df, NUMERIC_COLS + [DATE_COL]))
    cat_cells = n * len(_cols(df, CATEGORICAL_COLS))
    dims = {
        "completeness": 1 - miss / (n * len(cols)),
        "validity": 1 - inv / max(num_cells, 1),
        "uniqueness": 1 - dup / n,
        "consistency": 1 - cat / max(cat_cells, 1),
        "type_conformity": 1 - typ / max(typed_cells, 1),
        "outliers": 1 - outl / max(num_cells, 1),
    }
    dims = {k: float(round(max(v, 0) * 100, 2)) for k, v in dims.items()}
    w = cfg["score_weights"]
    overall = float(round(sum(dims[k] * w[k] for k in dims) / sum(w.values()), 2))
    grade = "Excellent" if overall >= 99 else "Good" if overall >= 97 else "Fair" if overall >= 93 else "Poor"
    return {"overall": overall, "grade": grade, "dimensions": dims,
            "issue_counts": {"missing_cells": int(miss), "duplicate_rows": int(dup),
                             "invalid_values": int(inv), "outliers": int(outl),
                             "type_issues": int(typ), "category_inconsistencies": int(cat)}}


@dataclass
class DataQualityReport:
    rows: int
    columns: int
    missing: pd.DataFrame
    duplicates: dict
    outliers: pd.DataFrame
    invalid: pd.DataFrame
    types: pd.DataFrame
    categories: pd.DataFrame
    score: dict = field(default_factory=dict)


def run_all(df: pd.DataFrame, cfg: dict) -> DataQualityReport:
    return DataQualityReport(rows=len(df), columns=len(df.columns), missing=missing_report(df),
                             duplicates=duplicate_report(df), outliers=outlier_report(df, cfg),
                             invalid=invalid_range_report(df, cfg), types=type_report(df),
                             categories=category_report(df), score=quality_score(df, cfg))


def load_raw(path) -> pd.DataFrame:
    """Read a CSV without any type inference or NA conversion (keeps the data untouched)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)
