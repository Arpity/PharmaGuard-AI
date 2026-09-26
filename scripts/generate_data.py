"""Entry point: python scripts/generate_data.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_generation.generate_dataset import generate  # noqa: E402

if __name__ == "__main__":
    generate()
