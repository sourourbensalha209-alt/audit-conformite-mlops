"""Exemples de la classe « Autre » : les paragraphes d'un contrat qui ne sont AUCUNE des 41 catégories.

Pourquoi : Legal-BERT a appris sur des clauses déjà repérées, donc il ne connaît pas le "reste" du contrat
(définitions, formules de politesse, signatures...). Sur un vrai contrat, il annonçait quand même une catégorie
sur 100 % de ces paragraphes ("fantômes"). On lui apprend donc une 42e réponse : « Autre ».

Un paragraphe est "neutre" si aucun passage annoté par les juristes ne le recouvre, NI ne se trouve à moins de
MARGIN caractères (CUAD n'annote parfois qu'une partie d'une clause : le voisinage immédiat est écarté pour ne
pas apprendre « Autre » sur le bout d'une vraie clause).
"""

from __future__ import annotations

import math
import random
import re

from .segment import Segment, segment

NEUTRAL_LABEL = "Autre"
MARGIN = 150            # caractères autour d'un passage annoté, écartés
MIN_CHARS = 40
MIN_WORDS = 6


def neutral_segments(doc: dict) -> list[Segment]:
    spans = [(a, b) for _, a, b in doc["spans"] if b > a]
    out = []
    for s in segment(doc["text"]):
        if len(s.text) < MIN_CHARS or len(re.findall(r"\w+", s.text)) < MIN_WORDS:
            continue
        if any(s.start < b + MARGIN and s.end > a - MARGIN for a, b in spans):
            continue
        out.append(s)
    return out


def sample_autre(docs: list[dict], n: int, seed: int = 42, cap_factor: float = 2.5) -> tuple[list[dict], dict]:
    """Tire n paragraphes neutres, répartis sur les contrats (au plus cap_factor fois la part moyenne par contrat,
    pour que les longs contrats ne dominent pas). Retourne (lignes, statistiques)."""
    rng = random.Random(seed)
    cap = max(3, math.ceil(cap_factor * n / max(len(docs), 1)))
    pool, available = [], 0
    for d in docs:
        segs = neutral_segments(d)
        available += len(segs)
        rng.shuffle(segs)
        pool += [(d["title"], s) for s in segs[:cap]]
    rng.shuffle(pool)
    chosen = pool[:n]
    rows = [{"text": s.text, "label": NEUTRAL_LABEL, "source": title, "origin": "autre"} for title, s in chosen]
    stats = {"contrats": len(docs), "paragraphes neutres disponibles": available, "plafond par contrat": cap,
             "retenus": len(rows)}
    return rows, stats
