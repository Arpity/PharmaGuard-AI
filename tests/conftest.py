import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_collection_modifyitems(config, items):
    """Split the suite for CI: `integration` = cross-module / end-to-end, `ui` = Streamlit AppTest page tests,
    everything else is a fast unit test.  Unit job: -m "not integration and not e2e and not ui"."""
    import inspect
    for item in items:
        if item.path.name == "test_integration.py":
            item.add_marker(pytest.mark.integration)
        try:
            if "AppTest" in inspect.getsource(item.function):
                item.add_marker(pytest.mark.ui)
        except (OSError, TypeError):
            pass

from config import load_config  # noqa: E402
from src.data_quality.rules import DATA_COLS  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def make_row(**over) -> dict:
    row = {"Batch_ID": "B00001", "Product_Name": "Paracetamol 500mg", "Dosage_Form": "Tablet",
           "Manufacturing_Date": "2024-01-15", "Plant": "Plant_A", "Batch_Size_Kg": "400",
           "Temperature": "22.5", "Humidity": "45", "pH": "6.5", "Dissolution": "92", "Assay": "100",
           "Impurity_Level": "0.2", "Yield_Percentage": "96", "Deviation_Count": "1",
           "Equipment_ID": "EQ-A01", "Operator_Shift": "Morning", "Cycle_Time_Hrs": "9",
           "Quality_Status": "Pass"}
    row.update(over)
    return row


def make_df(rows: list[dict]) -> pd.DataFrame:
    """Small all-text frame shaped like the raw CSV."""
    return pd.DataFrame(rows, columns=DATA_COLS).astype(object)


@pytest.fixture
def mk():
    return make_df, make_row


@pytest.fixture(autouse=True)
def _isolated_review_db(tmp_path, monkeypatch):
    """Tests must never touch the real demo database."""
    monkeypatch.setenv("PHARMAGUARD_DB_PATH", str(tmp_path / "test_review.db"))
    monkeypatch.setenv("PHARMAGUARD_OBS_DB", str(tmp_path / "test_obs.db"))


@pytest.fixture(autouse=True)
def _isolated_log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PHARMAGUARD_LOG_DIR", str(tmp_path / "logs"))
