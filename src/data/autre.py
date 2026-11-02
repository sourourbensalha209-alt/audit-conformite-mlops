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

from .header import HEADER_CATEGORIES
from .segment import HARD_MAX, Segment, segment

NEUTRAL_LABEL = "Autre"
MARGIN = 150            # caractères autour d'un passage annoté, écartés
MIN_CHARS = 40
MIN_WORDS = 6


IGNORED = frozenset(HEADER_CATEGORIES)     # gérées par l'extracteur à règles (src/data/header.py), pas par le classifieur


def neutral_segments(doc: dict, ignore=IGNORED) -> list[Segment]:
    """Paragraphes sans AUCUNE des catégories du classifieur. Les 4 catégories d'identité (nom, parties, dates de signature
    et d'entrée en vigueur) sont ignorées : un bloc de signature ou un titre est « Autre » pour le classifieur, sans
    contradiction avec les blocs que CUAD annote comme « Parties »."""
    spans = [(a, b) for c, a, b in doc["spans"] if b > a and c not in ignore]
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


def decide_with_autre(probs: list[float], labels: list[str], a: float | None) -> tuple[str, float]:
    """Choisit l'étiquette d'un segment à partir des probabilités du modèle.

    a = None : la plus probable (comportement habituel).
    a = 0.9  : on ne répond « Autre » que si le modèle en est sûr à au moins 90 % ; sinon on retient la meilleure
               des 41 catégories, avec une confiance recalculée entre catégories (sans « Autre »).
    Plus a est haut, plus on rend de clauses (rappel) et plus on accepte de fantômes.
    """
    if a is None:
        i = max(range(len(probs)), key=probs.__getitem__)
        return labels[i], probs[i]
    i_autre = labels.index(NEUTRAL_LABEL)
    if probs[i_autre] >= a:
        return NEUTRAL_LABEL, probs[i_autre]
    best = max((i for i in range(len(probs)) if i != i_autre), key=probs.__getitem__)
    return labels[best], probs[best] / max(1.0 - probs[i_autre], 1e-9)


def positive_segments(doc: dict, ignore=IGNORED) -> list[dict]:
    """Clauses annotées prises SOUS LA FORME OÙ ON LES VERRA : le segment (paragraphe) qui contient le passage annoté.

    À l'inférence, Legal-BERT reçoit des segments de 471 caractères en médiane ; il avait appris sur des passages de 194.
    On ajoute donc, pour chaque passage annoté entièrement contenu dans un segment, ce segment avec l'étiquette de la catégorie.
    """
    segs = segment(doc["text"])
    rows, seen = [], set()
    for cat, a, b in doc["spans"]:
        if b <= a or cat in ignore or not segs:
            continue
        seg = max(segs, key=lambda s: min(b, s.end) - max(a, s.start))
        covered = (min(b, seg.end) - max(a, seg.start)) / (b - a)
        if covered < 0.9 or len(seg.text) < MIN_CHARS or len(seg.text) > HARD_MAX or (seg.text, cat) in seen:
            continue
        seen.add((seg.text, cat))
        rows.append({"text": seg.text, "label": cat, "source": doc["title"], "origin": "segment"})
    return rows
