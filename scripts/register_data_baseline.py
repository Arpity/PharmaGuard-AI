"""Entry point: python scripts/register_data_baseline.py - record the expected raw-data schema (columns and inferred types)
that the monitoring dashboard compares against to detect schema changes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT, load_config  # noqa: E402
from src.data_quality import checks  # noqa: E402
from src.monitoring.drift import register_schema_baseline  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    b = register_schema_baseline(checks.load_raw(PROJECT_ROOT / cfg["dataset"]["raw_path"]), PROJECT_ROOT)
    print(f"registered {len(b['columns'])} columns: {b['columns']}")
