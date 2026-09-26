"""Entry point: python scripts/run_cleaning.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cleaning.pipeline import run_cleaning  # noqa: E402

if __name__ == "__main__":
    s = run_cleaning()
    print(f"rows {s['rows_raw']} -> {s['rows_clean']} | audit entries {s['audit_entries']}")
    print(f"score {s['score_before']['overall']} -> {s['score_after']['overall']}")
    print(s["score_after"]["dimensions"])
