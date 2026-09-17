"""Chemins et parametres partages par tout le projet."""

from pathlib import Path
import os
import yaml

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

for _p in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{ROOT / 'mlruns'}")


def load_params(section: str | None = None) -> dict:
    """Charge params.yaml, ou une seule de ses sections."""
    with open(ROOT / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params[section] if section else params
