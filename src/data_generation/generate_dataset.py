"""Generate a reproducible, intentionally messy pharma batch manufacturing dataset.

A clean dataset is simulated first (with Quality_Status derived from spec limits),
then realistic data-quality problems are injected. The result is saved untouched
to data/raw/. A JSON log records how many issues of each kind were injected.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import PROJECT_ROOT, load_config

PRODUCTS = {  # name: (dosage form, assay mean shift, typical batch kg)
    "Paracetamol 500mg": ("Tablet", 0.0, 400),
    "Amoxicillin 250mg": ("Capsule", -0.3, 250),
    "Metformin 850mg": ("Tablet", 0.1, 500),
    "Atorvastatin 20mg": ("Tablet", 0.0, 300),
    "Ibuprofen 400mg": ("Tablet", -0.1, 450),
    "Cetirizine 10mg": ("Tablet", 0.2, 200),
    "Omeprazole 20mg": ("Capsule", -0.4, 220),
    "Azithromycin Susp": ("Suspension", -0.2, 150),
    "Insulin Glargine": ("Injectable", -0.5, 60),
    "Ceftriaxone 1g": ("Injectable", -0.3, 90),
}
PLANTS = ["Plant_A", "Plant_B", "Plant_C", "Plant_D"]
PLANT_EFFECT = {"Plant_A": 0.0, "Plant_B": 0.15, "Plant_C": 0.35, "Plant_D": 0.1}
SHIFTS = ["Morning", "Evening", "Night"]
SHIFT_EFFECT = {"Morning": 0.0, "Evening": 0.1, "Night": 0.3}

NUMERIC = ["Batch_Size_Kg", "Temperature", "Humidity", "pH", "Dissolution", "Assay",
           "Impurity_Level", "Yield_Percentage", "Deviation_Count", "Cycle_Time_Hrs"]
COLUMNS = ["Batch_ID", "Product_Name", "Dosage_Form", "Manufacturing_Date", "Plant",
           "Batch_Size_Kg", "Temperature", "Humidity", "pH", "Dissolution", "Assay",
           "Impurity_Level", "Yield_Percentage", "Deviation_Count", "Equipment_ID",
           "Operator_Shift", "Cycle_Time_Hrs", "Quality_Status"]


def _simulate_clean(n: int, rng: np.random.Generator, cfg: dict) -> pd.DataFrame:
    names = rng.choice(list(PRODUCTS), n)
    plant = rng.choice(PLANTS, n, p=[0.35, 0.25, 0.25, 0.15])
    shift = rng.choice(SHIFTS, n, p=[0.45, 0.35, 0.20])
    start, end = pd.Timestamp(cfg["dataset"]["start_date"]), pd.Timestamp(cfg["dataset"]["end_date"])
    days = rng.integers(0, (end - start).days + 1, n)
    dates = start + pd.to_timedelta(days, unit="D")

    risk = (np.array([PLANT_EFFECT[p] for p in plant])
            + np.array([SHIFT_EFFECT[s] for s in shift]))  # latent process-stress term
    form = [PRODUCTS[p][0] for p in names]
    size = np.array([PRODUCTS[p][2] for p in names]) * rng.uniform(0.8, 1.2, n)

    temp = rng.normal(22.5 + risk, 1.6)
    hum = rng.normal(45 + 4 * risk, 5.5)
    ph = rng.normal(6.5, 0.3 + 0.1 * risk)
    assay = rng.normal(100 + np.array([PRODUCTS[p][1] for p in names]) - 0.8 * risk, 1.5)
    diss = rng.normal(92 - 2.5 * risk, 4.0)
    impurity = rng.gamma(2.2, 0.16 * (1 + risk))
    yld = rng.normal(96.5 - 1.5 * risk, 2.2)
    dev = rng.poisson(0.6 + 1.8 * risk)
    cycle = rng.normal(8 + size / 200, 1.0).clip(2)

    plant_letter = [p[-1] for p in plant]
    equip = [f"EQ-{l}{rng.integers(1, 13):02d}" for l in plant_letter]

    df = pd.DataFrame({
        "Batch_ID": [f"B{i:05d}" for i in range(1, n + 1)],
        "Product_Name": names, "Dosage_Form": form, "Manufacturing_Date": dates,
        "Plant": plant, "Batch_Size_Kg": size.round(1),
        "Temperature": temp.round(2), "Humidity": hum.round(1), "pH": ph.round(2),
        "Dissolution": diss.round(1), "Assay": assay.round(2),
        "Impurity_Level": impurity.round(3), "Yield_Percentage": yld.round(1),
        "Deviation_Count": dev, "Equipment_ID": equip, "Operator_Shift": shift,
        "Cycle_Time_Hrs": cycle.round(1),
    })

    specs = cfg["specs"]
    fails = np.zeros(n, dtype=int)
    for col, (lo, hi) in specs.items():
        fails += ((df[col] < lo) | (df[col] > hi)).to_numpy().astype(int)
    critical = ((df["Assay"] < 94) | (df["Impurity_Level"] > 1.3) | (df["Dissolution"] < 75)
                | (df["Deviation_Count"] >= 4)).to_numpy()
    status = np.where(critical | (fails >= 2), "Fail",
                      np.where((fails >= 1) | (dev >= 3), "Under Review", "Pass"))
    df["Quality_Status"] = status
    return df[COLUMNS]


def _inject_issues(df: pd.DataFrame, rng: np.random.Generator, cfg: dict) -> tuple[pd.DataFrame, dict]:
    q, n, log = cfg["quality_issues"], len(df), {}
    df = df.astype(object)  # allow mixed types in the same column

    def pick(rate: float, size: int = n) -> np.ndarray:
        return np.flatnonzero(rng.random(size) < rate)

    # Inconsistent categories
    variants = {
        "Plant": lambda v: rng.choice([v.lower(), v.replace("_", " "), v.replace("_", "-").upper(), f" {v} ", v[-1]]),
        "Quality_Status": lambda v: rng.choice([v.upper(), v.lower(), {"Pass": "Passed", "Fail": "Failed", "Under Review": "Review"}[v]]),
        "Operator_Shift": lambda v: rng.choice([v.lower(), v[0], v.upper(), f"{v} "]),
        "Product_Name": lambda v: rng.choice([v.upper(), v.lower(), v.replace(" ", "")]),
        "Dosage_Form": lambda v: rng.choice([v.lower(), v.upper(), v + "s"]),
    }
    for col, fn in variants.items():
        idx = pick(q["inconsistent_category_rate"])
        for i in idx:
            df.iat[i, df.columns.get_loc(col)] = fn(df.iat[i, df.columns.get_loc(col)])
        log[f"inconsistent_category:{col}"] = int(len(idx))

    # Mixed date formats
    fmts = ["%d/%m/%Y", "%m-%d-%Y", "%d %b %Y", "%Y/%m/%d", "%B %d, %Y"]
    c = df.columns.get_loc("Manufacturing_Date")
    for i in range(n):
        d = df.iat[i, c]
        df.iat[i, c] = d.strftime(rng.choice(fmts)) if rng.random() < q["date_format_issue_rate"] else d.strftime("%Y-%m-%d")
    log["date_format_variants"] = int(sum(1 for v in df["Manufacturing_Date"] if "-" not in v[:5] or v[4] != "-"))

    # Outliers (extreme but numerically plausible) and invalid ranges (impossible values)
    outlier = {"Temperature": (48, 65), "Humidity": (85, 95), "pH": (9.5, 11), "Dissolution": (30, 55),
               "Assay": (120, 140), "Impurity_Level": (4, 9), "Yield_Percentage": (60, 75),
               "Batch_Size_Kg": (2500, 6000), "Cycle_Time_Hrs": (40, 90), "Deviation_Count": (15, 40)}
    invalid = {"Temperature": [-40, 999], "Humidity": [-10, 130, 250], "pH": [-1, 15, 0],
               "Dissolution": [-5, 140], "Assay": [-10, 180], "Impurity_Level": [-0.5, -2, 105],
               "Yield_Percentage": [-20, 115, 150], "Batch_Size_Kg": [-100, 0],
               "Cycle_Time_Hrs": [-5, 0], "Deviation_Count": [-1, -3]}
    for col in NUMERIC:
        c = df.columns.get_loc(col)
        oi = pick(q["outlier_rate"])
        for i in oi:
            lo, hi = outlier[col]
            v = rng.uniform(lo, hi)
            df.iat[i, c] = int(v) if col == "Deviation_Count" else round(v, 2)
        ii = pick(q["invalid_range_rate"])
        for i in ii:
            df.iat[i, c] = rng.choice(invalid[col])
        log[f"outliers:{col}"], log[f"invalid_range:{col}"] = int(len(oi)), int(len(ii))

    # Data-type issues: numbers stored as text (units, thousands separators, words, floats)
    unit = {"Temperature": "C", "Humidity": "%", "Dissolution": "%", "Assay": "%",
            "Yield_Percentage": "%", "Impurity_Level": "%", "Cycle_Time_Hrs": "h", "Batch_Size_Kg": "kg"}
    words = {0: "zero", 1: "one", 2: "two", 3: "three"}
    tcount = 0
    for col in NUMERIC:
        c = df.columns.get_loc(col)
        for i in pick(q["type_issue_rate"]):
            v = df.iat[i, c]
            if col == "Deviation_Count":
                df.iat[i, c] = words.get(int(v), f"{float(v)}") if rng.random() < 0.5 else f"{float(v)}"
            elif col == "Batch_Size_Kg" and float(v) >= 1000:
                df.iat[i, c] = f"{float(v):,.1f}"
            elif col in unit:
                df.iat[i, c] = f"{v}{unit[col]}" if rng.random() < 0.7 else f"{v} {unit[col]}"
            else:
                df.iat[i, c] = str(v)
            tcount += 1
    log["type_issues"] = tcount

    # Missing values (several encodings)
    tokens = [np.nan, np.nan, np.nan, "", "N/A", "NA", "null", "-"]
    mr = q["missing_rate"]
    total_missing = 0
    for col in COLUMNS:
        if col in ("Batch_ID", "Quality_Status"):
            rate = 0.005 if col == "Quality_Status" else 0.0
        else:
            rate = mr.get(col, mr["default"])
        c = df.columns.get_loc(col)
        for i in pick(rate):
            df.iat[i, c] = tokens[rng.integers(len(tokens))]
            total_missing += 1
    log["missing_cells_injected"] = total_missing
    return df, log


def _add_duplicates(df: pd.DataFrame, rng: np.random.Generator, n_total: int) -> tuple[pd.DataFrame, dict]:
    n_dup = n_total - len(df)
    src = rng.choice(len(df), n_dup, replace=False)
    dups = df.iloc[src].copy()
    near = rng.random(n_dup) < 0.35  # near-duplicates: same Batch_ID, slightly re-entered values
    for j in np.flatnonzero(near):
        dups.iat[j, dups.columns.get_loc("Temperature")] = _perturb(dups.iat[j, dups.columns.get_loc("Temperature")], rng)
    out = pd.concat([df, dups], ignore_index=True)
    out = out.sample(frac=1, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)
    return out, {"duplicate_rows_added": int(n_dup), "near_duplicates": int(near.sum())}


def _perturb(v, rng):
    try:
        return round(float(v) + rng.normal(0, 0.3), 2)
    except (TypeError, ValueError):
        return v


def generate(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    ds = cfg["dataset"]
    rng = np.random.default_rng(ds["seed"])
    n_total = ds["n_rows"]
    n_unique = n_total - int(round(n_total * cfg["quality_issues"]["duplicate_row_rate"]))

    clean = _simulate_clean(n_unique, rng, cfg)
    messy, log = _inject_issues(clean, rng, cfg)
    final, dup_log = _add_duplicates(messy, rng, n_total)
    log.update(dup_log)
    log.update({"rows": len(final), "columns": len(final.columns), "seed": ds["seed"],
                "clean_quality_distribution": clean["Quality_Status"].value_counts().to_dict()})

    raw_path = PROJECT_ROOT / ds["raw_path"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(raw_path, index=False)
    log_path = PROJECT_ROOT / ds["log_path"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))
    print(f"Saved {len(final):,} rows x {len(final.columns)} cols -> {raw_path}")
    return final


if __name__ == "__main__":
    generate()
