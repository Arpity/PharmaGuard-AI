"""Entry point: python scripts/build_golden.py  (regenerates data/evaluation/golden_cases.json)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT, load_config  # noqa: E402
from src.analytics import kpis  # noqa: E402
from src.evaluation.golden import build_golden, save_golden  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    g = build_golden(kpis.load_clean(PROJECT_ROOT / cfg["processed"]["clean_path"]), cfg)
    save_golden(g, PROJECT_ROOT / cfg["evaluation"]["golden_path"])
    print(f"{g['n_cases']} golden cases; archetype batches: {g['archetype_batches']}")
