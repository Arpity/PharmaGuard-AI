"""Entry point: python scripts/run_evaluation.py [--live]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT, load_config  # noqa: E402
from src.analytics import kpis  # noqa: E402
from src.assistant.knowledge import KnowledgeBase  # noqa: E402
from src.evaluation.golden import load_golden  # noqa: E402
from src.evaluation.runner import run_evaluation  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    df = kpis.load_clean(PROJECT_ROOT / cfg["processed"]["clean_path"])
    rep = run_evaluation(load_golden(PROJECT_ROOT / cfg["evaluation"]["golden_path"]), df,
                         KnowledgeBase.from_directory(PROJECT_ROOT / "knowledge"), cfg, live="--live" in sys.argv,
                         save_to=PROJECT_ROOT / cfg["evaluation"]["results_path"])
    s = rep["summary"]
    print(f"mode: {rep['mode']} | cases {s['cases_passed']}/{s['cases']} passed | overall {s['overall_score']:.3f} | targets met: {s['all_targets_met']}")
    for k, v in s["metrics"].items():
        flag = "" if k not in s["target_status"] else ("  OK" if s["target_status"][k] else "  <-- below target")
        print(f"  {k:26s} {v if v is None else round(v, 3)}{flag}")
    for r in rep["cases"]:
        if not r["passed"]:
            print("FAIL", r["case_id"], r["category"], r["archetype"], r["failures"])
