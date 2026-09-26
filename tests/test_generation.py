import copy
import hashlib

import pandas as pd
import pytest

from src.data_generation.generate_dataset import COLUMNS, generate
from src.data_quality import checks
from src.data_quality.rules import DATA_COLS


def _gen(cfg, tmp_path, seed=42, n=3000):
    c = copy.deepcopy(cfg)
    c["dataset"].update(seed=seed, n_rows=n, raw_path=str(tmp_path / "raw.csv"), log_path=str(tmp_path / "log.json"))
    return c, generate(c)


def test_generated_shape_and_schema(cfg, tmp_path):
    c, df = _gen(cfg, tmp_path)
    assert df.shape == (3000, 18) and list(df.columns) == COLUMNS == DATA_COLS
    assert (tmp_path / "raw.csv").exists() and (tmp_path / "log.json").exists()


def test_generation_is_reproducible_and_seed_sensitive(cfg, tmp_path):
    md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir(), (tmp_path / "c").mkdir()
    _gen(cfg, tmp_path / "a", 5, 500)
    _gen(cfg, tmp_path / "b", 5, 500)
    _gen(cfg, tmp_path / "c", 6, 500)
    assert md5(tmp_path / "a" / "raw.csv") == md5(tmp_path / "b" / "raw.csv") != md5(tmp_path / "c" / "raw.csv")


def test_all_intended_data_issues_are_present_and_detected(cfg, tmp_path):
    c, _ = _gen(cfg, tmp_path)
    raw = checks.load_raw(tmp_path / "raw.csv")
    r = checks.run_all(raw, c)
    assert r.missing["missing_count"].sum() > 500
    assert r.duplicates["exact_duplicate_rows"] > 0 and r.duplicates["near_duplicate_rows"] > 0
    assert r.invalid["invalid_count"].sum() > 100 and r.outliers["outlier_count"].sum() > 100
    assert r.types["non_conforming"].sum() > 300 and r.categories["non_canonical"].sum() > 500
    assert r.score["overall"] < 99


def test_dataset_supports_the_three_quality_classes(cfg, tmp_path):
    _, df = _gen(cfg, tmp_path)
    status = df["Quality_Status"].astype(str).str.lower()
    assert status.str.contains("pass").sum() > 0 and status.str.contains("fail").sum() > 0 and status.str.contains("review").sum() > 0
