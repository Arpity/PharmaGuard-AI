"""Absolute paths for Streamlit AppTest (newer Streamlit resolves relative script paths from the calling test file)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def page(relative: str) -> str:
    return str(ROOT / relative)
