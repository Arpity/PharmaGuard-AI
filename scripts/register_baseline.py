"""Entry point: python scripts/register_baseline.py  - records approved fingerprints of prompt / knowledge / config / golden set."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.governance.metadata import load_governance, register_baseline  # noqa: E402

if __name__ == "__main__":
    print(register_baseline(load_governance()["system"]["version"]))
