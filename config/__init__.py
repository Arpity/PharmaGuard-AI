"""Project configuration loader."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def load_config() -> dict:
    """Load config/config.yaml, letting .env values override dataset basics."""
    with open(PROJECT_ROOT / "config" / "config.yaml", "r") as fh:
        cfg = yaml.safe_load(fh)
    ds = cfg["dataset"]
    ds["seed"] = int(os.getenv("RANDOM_SEED", ds["seed"]))
    ds["n_rows"] = int(os.getenv("N_ROWS", ds["n_rows"]))
    ds["raw_path"] = os.getenv("RAW_DATA_PATH", ds["raw_path"])
    return cfg
