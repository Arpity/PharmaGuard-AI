"""Entry point: python scripts/generate_demo_traces.py [N]   - synthetic traffic through the real pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT, load_config  # noqa: E402
from src.analytics import kpis  # noqa: E402
from src.assistant.knowledge import KnowledgeBase  # noqa: E402
from src.observability.demo import generate_demo_traffic  # noqa: E402
from src.observability.recorder import get_recorder  # noqa: E402

if __name__ == "__main__":
    cfg, n = load_config(), int(sys.argv[1]) if len(sys.argv) > 1 else 150
    df = kpis.load_clean(PROJECT_ROOT / cfg["processed"]["clean_path"])
    print(generate_demo_traffic(n, df, KnowledgeBase.from_directory(PROJECT_ROOT / "knowledge"), cfg, get_recorder()))
