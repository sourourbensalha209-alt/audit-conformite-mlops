"""Data lake du projet : chaque audit et chaque correction y est versé, puis Spark le transforme.

Organisation en trois zones (architecture « médaillon ») :

    data/lake/
      bronze/   événements BRUTS, tels qu'ils arrivent, jamais modifiés (un fichier JSON par événement)
        audits/jour=2026-11-19/<job_id>.json
        corrections/jour=2026-11-19/<uuid>.json
      silver/   données NETTOYÉES et typées par Spark (Parquet, doublons retirés, une ligne par clause attendue)
      gold/     AGRÉGATS prêts à lire (Parquet + resume.json) : catégories souvent manquantes, volume par jour...

Pourquoi un fichier par événement : écrire n'écrase jamais rien (plusieurs workers en parallèle, sans verrou),
et le partitionnement par jour permet à Spark de ne relire que les jours utiles.
Ce module n'utilise que la bibliothèque standard : le service d'audit l'appelle sans dépendance supplémentaire.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAKE = Path(os.getenv("AUDIT_LAKE", ROOT / "data" / "lake"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def audit_event(job_id: str, document: str, result: dict, duree_s: float | None, modele: str,
                source: str = "api") -> dict:
    """Résultat d'un audit -> événement compact pour le data lake (sans le texte du contrat)."""
    mon = result.get("monitoring", {})
    return {
        "type": "audit",
        "job_id": job_id,
        "date": _now().isoformat(timespec="seconds"),
        "document": document,
        "source": source,                      # "api" (interface) ou "kafka" (ingestion en masse)
        "modele": modele,
        "duree_s": duree_s,
        "paragraphes": mon.get("paragraphes", result.get("stats", {}).get("segments")),
        "confiance_moyenne": mon.get("confiance_moyenne"),
        "part_abstention": mon.get("part_abstention"),
        "part_autre": mon.get("part_autre"),
        "rgpd": result.get("stats", {}).get("rgpd", 0),
        "attendues": [{"categorie": a["categorie"], "statut": a["statut"]} for a in result.get("attendues", [])],
        "clauses": [{"categorie": c["categorie"], "confiance": c["confiance"]} for c in result.get("clauses", [])],
    }


def correction_event(correction: dict) -> dict:
    """Correction d'un juriste -> événement (le passage du contrat n'est pas recopié, seulement sa présence)."""
    return {
        "type": "correction",
        "job_id": correction.get("job_id"),
        "date": correction.get("date") or _now().isoformat(timespec="seconds"),
        "clause_ref": correction.get("clause_ref"),
        "verdict_corrige": correction.get("verdict_corrige"),
        "categorie_corrigee": correction.get("categorie_corrigee"),
        "avec_passage": bool(correction.get("texte")),
    }


def write_bronze(kind: str, event: dict, lake: Path | None = None) -> Path:
    """Ajoute un événement à la zone bronze : <lake>/bronze/<kind>/jour=AAAA-MM-JJ/<id>.json (une ligne JSON)."""
    lake = Path(lake or LAKE)
    jour = str(event.get("date", _now().isoformat()))[:10]
    folder = lake / "bronze" / kind / f"jour={jour}"
    folder.mkdir(parents=True, exist_ok=True)
    name = event.get("job_id") if kind == "audits" and event.get("job_id") else uuid.uuid4().hex
    path = folder / f"{name}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(event, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)                         # écriture atomique : Spark ne lit jamais un fichier à moitié écrit
    return path


def read_gold(lake: Path | None = None) -> dict | None:
    """Résumé produit par le job Spark (gold/resume.json), ou None si Spark n'a pas encore tourné."""
    path = Path(lake or LAKE) / "gold" / "resume.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
