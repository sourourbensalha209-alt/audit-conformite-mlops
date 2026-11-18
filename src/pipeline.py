"""Chaîne d'audit complète, utilisable par l'API comme par un script.

    from src.pipeline import audit_file
    result, html = audit_file("contrat.pdf")

Étapes : lecture (PDF / Word / texte) -> découpage en clauses -> classification (Legal-BERT v4)
-> clauses attendues manquantes -> rapport HTML. Mêmes fonctions que scripts/audit_contract.py.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from src.data.segment import segment
from src.judge.report import analyse, build_meta, render_html
from src.retrieval.rules import load_rules

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "best_model_v4"
RULES = ROOT / "data" / "rules" / "rules_contractuelles.json"


def read_document(path: str | Path) -> str:
    """Texte du document, avec une ligne vide entre paragraphes (ce qu'attend segment())."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        from src.data.paragraphs import pdf_to_contract_text
        return pdf_to_contract_text(path)
    if ext == ".docx":
        from src.data.paragraphs import docx_to_contract_text
        return docx_to_contract_text(path)
    if ext == ".txt":
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Format non pris en charge : {ext}")


@lru_cache(maxsize=1)
def _classifier():
    """Chargé une seule fois : le modèle met plusieurs secondes à charger."""
    from src.models.predict import ClauseClassifier
    return ClauseClassifier(MODEL_DIR, threshold=0.0)


@lru_cache(maxsize=1)
def _rules():
    return load_rules(RULES)


def _corpus_version() -> str:
    try:
        from src.retrieval.corpus import CorpusStore
        return CorpusStore(ROOT / "data" / "corpus").version()
    except Exception:  # noqa: BLE001
        return "inconnue"


def audit_text(text: str, title: str) -> tuple[dict, str]:
    """Texte d'un contrat -> (résultat de l'analyse, page HTML du rapport)."""
    segs = segment(text)
    if not segs:
        raise ValueError("Aucune clause trouvée : le document est vide ou illisible.")
    labels = json.loads((MODEL_DIR / "labels.json").read_text(encoding="utf-8"))
    probs = [p.probs for p in _classifier().predict([s.text for s in segs])]
    result = analyse(text, labels, probs, _rules())
    html = render_html(result, build_meta(title, MODEL_DIR.name, _corpus_version()))
    return result, html


def audit_file(path: str | Path, title: str | None = None) -> tuple[dict, str]:
    path = Path(path)
    return audit_text(read_document(path), title or path.stem)
