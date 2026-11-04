"""Lecture de la réponse du juge, et contrôle des sources citées.

Un petit modèle ouvert répond parfois avec du texte autour du JSON, des accents dans le verdict, ou des identifiants entre
crochets. On accepte ces écarts, mais on les compte : `valid` n'est vrai que pour un JSON propre.
"""

from __future__ import annotations

import json
import re
import unicodedata

from .prompt import VERDICTS

_ALIASES = {"conforme": "conforme", "compliant": "conforme", "non conforme": "non_conforme", "non-conforme": "non_conforme",
            "non_conforme": "non_conforme", "nonconforme": "non_conforme", "a verifier": "a_verifier",
            "a_verifier": "a_verifier", "a-verifier": "a_verifier", "averifier": "a_verifier"}


def _strip(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn").strip()


def normalize_verdict(v) -> str | None:
    return _ALIASES.get(_strip(str(v))) if v is not None else None


def parse_output(text: str) -> dict:
    """Retourne {verdict, justification, sources, valid, recovered, error}.
    valid     : JSON propre avec un verdict connu
    recovered : pas de JSON propre, mais le verdict a pu être lu dans le texte"""
    out = {"verdict": None, "justification": "", "sources": [], "valid": False, "recovered": False, "error": None}
    body = re.sub(r"```(?:json)?", "", text or "")
    start, end = body.find("{"), body.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(body[start:end + 1])
            v = normalize_verdict(data.get("verdict"))
            if v in VERDICTS:
                srcs = data.get("sources", [])
                srcs = [srcs] if isinstance(srcs, str) else list(srcs) if isinstance(srcs, (list, tuple)) else []
                out.update(verdict=v, justification=str(data.get("justification", "")).strip(),
                           sources=[str(s).strip() for s in srcs], valid=True)
                return out
            out["error"] = "verdict inconnu"
        except (json.JSONDecodeError, AttributeError):
            out["error"] = "JSON illisible"
    else:
        out["error"] = "pas de JSON"
    m = re.search(r'verdict["\']?\s*[:=]\s*["\']?(.{0,40})', body, re.S | re.I)
    if m:
        v = _find_verdict(m.group(1))
        if v:
            out.update(verdict=v, recovered=True)
    return out


def _find_verdict(fragment: str) -> str | None:
    """Premier verdict reconnaissable au début d'un texte libre (« conforme car la clause... », « non conforme », « à vérifier »)."""
    t = _strip(fragment)
    for pattern, verdict in ((r"^non[ _-]?conforme", "non_conforme"), (r"^a[ _-]?verifier", "a_verifier"), (r"^conforme", "conforme")):
        if re.search(pattern, t):
            return verdict
    return None


def clean_source(s: str) -> str:
    return re.sub(r"^[\[\(\s]+|[\]\)\s]+$", "", s.strip())


def check_citations(sources: list[str], allowed: list[str]) -> tuple[int, int]:
    """(nombre de sources citées, nombre de sources qui ne figuraient PAS dans le prompt)."""
    ok = {clean_source(a) for a in allowed}
    cited = [clean_source(s) for s in sources if s.strip()]
    return len(cited), sum(c not in ok for c in cited)
