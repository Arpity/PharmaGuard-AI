"""CI evaluation gate.  python scripts/evaluation_gate.py [--run] [--report PATH] [--gate PATH]

  --run     run the golden evaluation first (demo mode, no network) and write the report
  exit code 0 = gate passed, 1 = a blocking check failed, 2 = configuration / input error
Thresholds: config/evaluation_gate.yaml, overridable with EVAL_GATE_* environment variables."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT, load_config  # noqa: E402
from src.evaluation.gate import evaluate_gate, load_gate_config, load_report, render_markdown  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report")
    ap.add_argument("--gate", default=str(PROJECT_ROOT / "config" / "evaluation_gate.yaml"))
    a = ap.parse_args(argv)
    cfg = load_config()
    report_path = Path(a.report or PROJECT_ROOT / cfg["evaluation"]["results_path"])
    golden_path = PROJECT_ROOT / cfg["evaluation"]["golden_path"]
    try:
        if a.run:
            from src.analytics import kpis
            from src.assistant.knowledge import KnowledgeBase
            from src.evaluation.golden import load_golden
            from src.evaluation.runner import run_evaluation
            golden = load_golden(golden_path)
            run_evaluation(golden, kpis.load_clean(PROJECT_ROOT / cfg["processed"]["clean_path"]),
                           KnowledgeBase.from_directory(PROJECT_ROOT / "knowledge"), cfg, save_to=report_path)
        gate = load_gate_config(a.gate)
        report = load_report(report_path)
        current = json.loads(golden_path.read_text()).get("dataset_md5") if golden_path.exists() else None
    except (OSError, ValueError, KeyError) as e:
        print(f"evaluation gate could not run: {e}", file=sys.stderr)
        return 2
    res = evaluate_gate(report, gate, current)
    md = render_markdown(res, report)
    print(md)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(md + "\n")
    if not res.passed:
        print("\nEVALUATION GATE FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
