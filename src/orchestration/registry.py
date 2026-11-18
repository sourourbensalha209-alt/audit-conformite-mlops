"""Registre du modèle en production : quel modèle le service d'audit charge.

Ordre de priorité :
  1. variable d'environnement AUDIT_MODEL (ex. AUDIT_MODEL=best_model_v5), pour forcer un modèle ;
  2. models/production.json, écrit par la promotion (src/orchestration/promote.py) ;
  3. best_model_v4, le modèle validé en phase 2.

Changer de modèle ne demande donc aucune modification du code : la promotion met à jour production.json,
et le service charge le nouveau modèle à son prochain démarrage.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
PRODUCTION_FILE = MODELS / "production.json"
DEFAULT_MODEL = "best_model_v4"


def production_model_name(production_file: Path | None = None) -> str:
    env = os.getenv("AUDIT_MODEL")
    if env:
        return env
    path = production_file or PRODUCTION_FILE
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))["modele"]
        except (json.JSONDecodeError, KeyError):
            pass
    return DEFAULT_MODEL


def production_model_dir(production_file: Path | None = None) -> Path:
    return MODELS / production_model_name(production_file)
