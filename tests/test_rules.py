import numpy as np
import pandas as pd

from src.data_quality import rules as r


def test_is_missing_detects_placeholders():
    s = pd.Series(["x", "", "N/A", " null ", "-", np.nan, "Unknown", "0"])
    assert r.is_missing(s).tolist() == [False, True, True, True, True, True, True, False]


def test_parse_numeric_statuses_and_values():
    out = r.parse_numeric(pd.Series(["22.5", "22.4C", "99.7 %", "1,234.5", "-5", "abc", "", "N/A"]))
    assert out["status"].tolist() == ["ok", "converted", "converted", "converted", "ok",
                                      "unparseable", "missing", "missing"]
    assert out["value"].iloc[:5].tolist() == [22.5, 22.4, 99.7, 1234.5, -5.0]
    assert out["value"].iloc[5:].isna().all()


def test_parse_numeric_integer_words_and_decimals():
    out = r.parse_numeric(pd.Series(["2", "2.0", "two", "2.5"]), integer=True)
    assert out["status"].tolist() == ["ok", "converted", "converted", "unparseable"]
    assert out["value"].iloc[:3].tolist() == [2.0, 2.0, 2.0]


def test_parse_numeric_rejects_infinity():
    assert r.parse_numeric(pd.Series(["inf"]))["status"].iloc[0] == "unparseable"


def test_parse_dates_mixed_formats():
    s = pd.Series(["2024-03-05", "05/03/2024", "03-05-2024", "05 Mar 2024", "2024/03/05", "March 05, 2024", "garbage", ""])
    out = r.parse_dates(s)
    assert out["status"].tolist() == ["ok"] + ["converted"] * 5 + ["unparseable", "missing"]
    assert (out["value"].iloc[:6] == pd.Timestamp("2024-03-05")).all()
    assert out["value"].iloc[6:].isna().all()


def test_canonicalisers():
    assert [r.canon_plant(v) for v in ["Plant_A", "plant a", "PLANT-B", " Plant_C ", "D", "Plant_Z"]] == \
           ["Plant_A", "Plant_A", "Plant_B", "Plant_C", "Plant_D", None]
    assert [r.canon_status(v) for v in ["PASS", "Passed", "failed", "Under Review", "review", "??"]] == \
           ["Pass", "Pass", "Fail", "Under Review", "Under Review", None]
    assert [r.canon_shift(v) for v in ["morning ", "E", "NIGHT", "x"]] == ["Morning", "Evening", "Night", None]
    assert r.canon_product("AZITHROMYCINSUSP") == "Azithromycin Susp"
    assert r.canon_product("ibuprofen400mg") == "Ibuprofen 400mg"
    assert r.canon_dosage("tablets") == "Tablet" and r.canon_dosage("INJECTABLE") == "Injectable"
    assert r.canon_equipment(" eq-b07") == "EQ-B07" and r.canon_equipment("EQ-Z99") is None


def test_canonicalise_keeps_missing_missing():
    out = r.canonicalise(pd.Series(["plant a", "N/A", np.nan, "zzz"]), r.canon_plant)
    assert out.iloc[0] == "Plant_A" and out.iloc[1:].isna().all()


def test_typed_values_are_not_reported_as_text_problems():
    import datetime
    ints = r.parse_numeric(pd.Series([2.0, 3, np.nan], dtype=object), integer=True)
    assert ints["status"].tolist() == ["ok", "ok", "missing"]
    dates = r.parse_dates(pd.Series([pd.Timestamp("2024-03-05"), datetime.date(2024, 3, 6), "05/03/2024", None], dtype=object))
    assert dates["status"].tolist() == ["ok", "ok", "converted", "missing"]
    assert dates["value"].iloc[0] == pd.Timestamp("2024-03-05") and dates["value"].iloc[2] == pd.Timestamp("2024-03-05")
