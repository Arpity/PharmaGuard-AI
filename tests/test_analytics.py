import numpy as np
import pandas as pd
import pytest

from src.analytics import kpis


def frame(rows):
    base = {"Batch_ID": "B1", "Manufacturing_Date": "2024-01-10", "Product_Name": "P1", "Plant": "Plant_A",
            "Operator_Shift": "Morning", "Dissolution": 90.0, "Assay": 100.0, "Impurity_Level": 0.5,
            "Yield_Percentage": 95.0, "Deviation_Count": 1, "Quality_Status": "Pass", "Outlier_Flag": False,
            "Outlier_Columns": "", "Has_Missing_CQA": False, "Temperature": 22.0, "Humidity": 45.0, "pH": 6.5}
    df = pd.DataFrame([{**base, "Batch_ID": f"B{i}", **r} for i, r in enumerate(rows)])
    df["Manufacturing_Date"] = pd.to_datetime(df["Manufacturing_Date"])
    return df


def test_kpis_exact_values():
    df = frame([{"Quality_Status": "Pass", "Assay": 100, "Deviation_Count": 0},
                {"Quality_Status": "Pass", "Assay": 98, "Deviation_Count": 2},
                {"Quality_Status": "Under Review", "Assay": 96, "Deviation_Count": 3},
                {"Quality_Status": "Fail", "Assay": np.nan, "Deviation_Count": 5},
                {"Quality_Status": "Unknown", "Assay": 102, "Deviation_Count": 0}])
    k = kpis.compute_kpis(df)
    assert (k["total_batches"], k["pass_batches"], k["risk_batches"], k["fail_batches"]) == (5, 2, 2, 1)
    assert k["unknown_status"] == 1
    assert k["failure_rate"] == pytest.approx(25.0)          # 1 fail / 4 known
    assert k["risk_rate"] == pytest.approx(50.0)
    assert k["avg_assay"] == pytest.approx((100 + 98 + 96 + 102) / 4)   # NaN ignored
    assert k["total_deviations"] == 10 and k["batches_with_deviation"] == 3


def test_kpis_all_missing_metric_is_nan_not_error():
    df = frame([{"Impurity_Level": np.nan}])
    assert np.isnan(kpis.compute_kpis(df)["avg_impurity"])


def test_filters_combine_and_handle_dates():
    df = frame([{"Product_Name": "P1", "Plant": "Plant_A", "Manufacturing_Date": "2024-01-10"},
                {"Product_Name": "P2", "Plant": "Plant_A", "Manufacturing_Date": "2024-03-10"},
                {"Product_Name": "P2", "Plant": "Plant_B", "Manufacturing_Date": "2024-03-11", "Quality_Status": "Fail"},
                {"Product_Name": "P2", "Plant": "Plant_B", "Manufacturing_Date": None}])
    assert len(kpis.apply_filters(df)) == 4
    assert len(kpis.apply_filters(df, products=["P2"], plants=["Plant_B"])) == 2
    assert len(kpis.apply_filters(df, statuses=["Fail"])) == 1
    r = ("2024-03-01", "2024-03-31")
    assert len(kpis.apply_filters(df, date_range=r, include_undated=False)) == 2
    assert len(kpis.apply_filters(df, date_range=r, include_undated=True)) == 3
    assert kpis.apply_filters(df, products=[]).empty


def test_group_summary_rates():
    df = frame([{"Plant": "Plant_A", "Quality_Status": "Pass"}, {"Plant": "Plant_A", "Quality_Status": "Fail"},
                {"Plant": "Plant_B", "Quality_Status": "Pass"}, {"Plant": "Plant_B", "Quality_Status": "Under Review"}])
    g = kpis.group_summary(df, "Plant").set_index("Plant")
    assert g.loc["Plant_A", "fail_rate"] == 50.0 and g.loc["Plant_B", "fail_rate"] == 0.0
    assert g.loc["Plant_B", "risk_rate"] == 50.0 and g.loc["Plant_A", "batches"] == 2


def test_monthly_trend_and_spec_compliance(cfg):
    df = frame([{"Manufacturing_Date": "2024-01-05"}, {"Manufacturing_Date": "2024-01-20", "Quality_Status": "Fail"},
                {"Manufacturing_Date": "2024-02-02"}, {"Manufacturing_Date": None}])
    tr = kpis.monthly_trend(df)
    assert tr["batches"].tolist() == [2, 1] and tr["fail_rate"].tolist() == [50.0, 0.0]
    df.loc[0, "Impurity_Level"] = 2.0                                   # above the 1.0 limit
    sc = kpis.spec_compliance(df, cfg["specs"]).set_index("attribute")
    assert sc.loc["Impurity_Level", "in_spec_pct"] == pytest.approx(75.0)


def test_insights_are_grounded_in_data(cfg):
    rows = [{"Plant": "Plant_A", "Quality_Status": "Pass"}] * 40 + \
           [{"Plant": "Plant_B", "Quality_Status": "Pass"}] * 30 + [{"Plant": "Plant_B", "Quality_Status": "Fail"}] * 10
    ins = kpis.generate_insights(frame(rows), cfg)
    plant = next(i for i in ins if i["title"] == "Highest-risk plant")
    assert "Plant_B" in plant["text"] and "25.0%" in plant["text"]      # 10 / 40
    overall = next(i for i in ins if i["title"] == "Overall quality")
    assert "80 batches" in overall["text"] and "12.5%" in overall["text"]


def test_insights_empty_frame_is_safe(cfg):
    assert kpis.generate_insights(frame([{}]).iloc[0:0], cfg)[0]["title"] == "No data"


def test_anomaly_table_only_flagged():
    df = frame([{"Outlier_Flag": True, "Outlier_Columns": "Assay"}, {}, {"Outlier_Flag": True}])
    assert len(kpis.anomaly_table(df)) == 2


def test_dashboard_page_renders_with_filters():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app/pages/2_Analytics_Dashboard.py", default_timeout=120).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert labels[:4] == ["Total Batches", "Pass / Risk Batches", "Quality Failure Rate", "Deviations"]
    assert {"Average Dissolution", "Average Assay", "Average Impurity", "Average Yield"} <= set(labels)
    at.multiselect[1].set_value(["Plant_C"]).run()                     # Plant filter
    assert not at.exception
    assert int(at.metric[0].value.replace(",", "")) < 2910
