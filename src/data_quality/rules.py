"""Shared business rules: schema, master data, and value parsers/canonicalisers.

Both the data-quality checks and the cleaning pipeline use these, so "what counts as a
problem" and "how it is fixed" can never drift apart.
"""
from __future__ import annotations

import datetime
import re
from typing import Callable, Optional

import numpy as np
import pandas as pd

DATE_COL = "Manufacturing_Date"
NUMERIC_COLS = ["Batch_Size_Kg", "Temperature", "Humidity", "pH", "Dissolution", "Assay",
                "Impurity_Level", "Yield_Percentage", "Deviation_Count", "Cycle_Time_Hrs"]
INTEGER_COLS = ["Deviation_Count"]
CATEGORICAL_COLS = ["Product_Name", "Dosage_Form", "Plant", "Equipment_ID", "Operator_Shift",
                    "Quality_Status"]
ID_COL = "Batch_ID"
DATA_COLS = [ID_COL, "Product_Name", "Dosage_Form", DATE_COL, "Plant", "Batch_Size_Kg", "Temperature",
             "Humidity", "pH", "Dissolution", "Assay", "Impurity_Level", "Yield_Percentage",
             "Deviation_Count", "Equipment_ID", "Operator_Shift", "Cycle_Time_Hrs", "Quality_Status"]

# Master data ---------------------------------------------------------------
PRODUCT_TO_FORM = {
    "Paracetamol 500mg": "Tablet", "Amoxicillin 250mg": "Capsule", "Metformin 850mg": "Tablet",
    "Atorvastatin 20mg": "Tablet", "Ibuprofen 400mg": "Tablet", "Cetirizine 10mg": "Tablet",
    "Omeprazole 20mg": "Capsule", "Azithromycin Susp": "Suspension",
    "Insulin Glargine": "Injectable", "Ceftriaxone 1g": "Injectable",
}
DOSAGE_FORMS = sorted(set(PRODUCT_TO_FORM.values()))
PLANTS = ["Plant_A", "Plant_B", "Plant_C", "Plant_D"]
SHIFTS = ["Morning", "Evening", "Night"]
STATUSES = ["Pass", "Under Review", "Fail"]
UNKNOWN = "Unknown"

MISSING_TOKENS = {"", "n/a", "na", "null", "-", "--", "nan", "none", "unknown"}
NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d %b %Y", "%Y/%m/%d", "%B %d, %Y"]

_NUM_RE = re.compile(r"^\s*([-+]?(?:\d[\d,]*\.?\d*|\.\d+))\s*[a-zA-Z%°]*\s*$")
_INT_RE = re.compile(r"^[-+]?\d+$")
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def is_missing(s: pd.Series) -> pd.Series:
    """True for NaN/None and for text placeholders such as '', 'N/A', 'null', '-'."""
    return s.isna() | s.astype(str).str.strip().str.lower().isin(MISSING_TOKENS)


# Parsers --------------------------------------------------------------------
def _to_number(text: str) -> float:
    m = _NUM_RE.match(text)
    if m:
        return float(m.group(1).replace(",", ""))
    if text.lower() in NUMBER_WORDS:
        return float(NUMBER_WORDS[text.lower()])
    return np.nan


def parse_numeric(s: pd.Series, integer: bool = False) -> pd.DataFrame:
    """Parse a column to floats. Returns columns ``value`` and ``status``.

    status: missing | ok (clean numeric text) | converted (units / separators / words /
    '2.0' for an integer column had to be stripped) | unparseable.
    """
    miss = is_missing(s)
    text = s.where(~miss).map(lambda v: str(v).strip() if pd.notna(v) else None)
    direct = pd.to_numeric(text, errors="coerce")
    direct = direct.where(np.isfinite(direct))
    value, status = direct.copy(), pd.Series("ok", index=s.index, dtype=object)
    status[miss] = "missing"

    todo = ~miss & direct.isna()
    for i in s.index[todo]:
        value[i] = _to_number(text[i])
    status[todo] = np.where(value[todo].notna(), "converted", "unparseable")

    if integer:
        ok = status.isin(["ok", "converted"]) & value.notna()
        frac = ok & (value != value.round())
        value[frac], status[frac] = np.nan, "unparseable"
        is_text = s.map(lambda v: isinstance(v, str))          # a typed float such as 2.0 is not a text problem
        not_int_text = (status == "ok") & is_text & ~text.fillna("").str.match(_INT_RE)
        status[not_int_text] = "converted"
    return pd.DataFrame({"value": value, "status": status})


def parse_dates(s: pd.Series) -> pd.DataFrame:
    """Parse mixed-format dates. status: missing | ok (ISO) | converted | unparseable."""
    miss = is_missing(s)
    text = s.where(~miss).map(lambda v: str(v).strip() if pd.notna(v) else None)
    value = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    typed = ~miss & s.map(lambda v: isinstance(v, (pd.Timestamp, datetime.date, np.datetime64)))   # already real dates
    if typed.any():
        value[typed] = pd.to_datetime(s[typed])
    for fmt in DATE_FORMATS:
        todo = ~miss & value.isna() & ~typed
        if not todo.any():
            break
        value[todo] = pd.to_datetime(text[todo], format=fmt, errors="coerce")
    status = pd.Series("ok", index=s.index, dtype=object)
    status[miss] = "missing"
    status[~miss & ~typed & ~text.fillna("").str.match(_ISO_RE)] = "converted"
    status[~miss & value.isna()] = "unparseable"
    return pd.DataFrame({"value": value, "status": status})


# Canonicalisers: raw value -> canonical value, or None if it cannot be mapped ----
def _key(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def canon_batch_id(v) -> Optional[str]:
    t = str(v).strip().upper()
    return t if re.fullmatch(r"B\d{5}", t) else None


def canon_plant(v) -> Optional[str]:
    m = re.fullmatch(r"(?i)(?:plant[\s_-]*)?([a-d])", str(v).strip())
    return f"Plant_{m.group(1).upper()}" if m else None


def canon_status(v) -> Optional[str]:
    return {"pass": "Pass", "passed": "Pass", "fail": "Fail", "failed": "Fail", "review": "Under Review",
            "underreview": "Under Review"}.get(_key(v))


def canon_shift(v) -> Optional[str]:
    return {"m": "Morning", "morning": "Morning", "e": "Evening", "evening": "Evening",
            "n": "Night", "night": "Night"}.get(_key(v))


def canon_product(v) -> Optional[str]:
    return {_key(p): p for p in PRODUCT_TO_FORM}.get(_key(v))


def canon_dosage(v) -> Optional[str]:
    lookup = {f.lower(): f for f in DOSAGE_FORMS}
    k = str(v).strip().lower()
    return lookup.get(k) or (lookup.get(k[:-1]) if k.endswith("s") else None)


def canon_equipment(v) -> Optional[str]:
    t = str(v).strip().upper()
    return t if re.fullmatch(r"EQ-[A-D]\d{2}", t) else None


CANONICALISERS: dict[str, Callable] = {
    "Product_Name": canon_product, "Dosage_Form": canon_dosage, "Plant": canon_plant,
    "Equipment_ID": canon_equipment, "Operator_Shift": canon_shift, "Quality_Status": canon_status,
}


def canonicalise(s: pd.Series, fn: Callable) -> pd.Series:
    """Apply a canonicaliser; missing placeholders stay NaN, unmappable values become NaN too."""
    miss = is_missing(s)
    out = s.map(lambda v: fn(v) if pd.notna(v) else None).astype(object)
    out[miss] = None
    return out.where(out.notna(), np.nan)
