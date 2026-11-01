"""Nettoyage du texte extrait d'un PDF de loi : retire en-têtes, pieds de page et numéros de page.

Règle de prudence : on ne retire JAMAIS une ligne du corps de la page. Seules les premières
(HEAD_LINES) et dernières (TAIL_LINES) lignes de chaque page peuvent être des en-têtes ou pieds de page,
et seulement si elles sont connues (motifs fixes) ou répétées sur au moins 40 % des pages.
Supprimer du texte de loi en silence serait pire que laisser un en-tête.
"""

from __future__ import annotations

import re

HEAD_LINES = 4
TAIL_LINES = 3
MIN_LEN = 12            # une ligne répétée plus courte ("a)", "1.") n'est jamais prise pour un en-tête
MIN_SHARE = 0.4         # part des pages sur lesquelles la ligne doit revenir

# Motifs fixes, valables n'importe où sur la page
NOISE = [
    re.compile(r"^\s*Page\s+\d+\s*(?:sur|/|of)\s*\d+\s*$", re.I),   # "Page 3 sur 18"
    re.compile(r"^\s*\d{1,3}\s*$"),                                   # numéro de page seul
    re.compile(r"Journal\s+officiel\s+de\s+l.Union\s+europ", re.I),  # en-tête du JO de l'UE (le RGPD le contient)
    re.compile(r"^\s*L\s+\d{1,3}/\d{1,3}\s*$"),                       # "L 119/17"
    re.compile(r"^\s*\d{1,2}\.\d{1,2}\.\d{4}\s*$"),                   # "4.5.2016"
    re.compile(r"^\s*FR\s*$"),
]


def _norm(line: str) -> str:
    """Forme comparable : chiffres remplacés, pour que "Page 3 sur 18" et "Page 4 sur 18" coïncident."""
    return re.sub(r"\d+", "#", " ".join(line.split()))


def strip_page_noise(pages: list[list[str]]) -> tuple[list[str], int, list[tuple[str, int]]]:
    """pages : lignes de chaque page. Retourne (lignes conservées, nb retirées, en-têtes répétés détectés)."""
    cleaned = [[ln.rstrip() for ln in page if ln.strip()] for page in pages]
    n_pages = len(cleaned)

    def edge_positions(n_lines: int) -> set[int]:
        return set(range(min(HEAD_LINES, n_lines))) | set(range(max(0, n_lines - TAIL_LINES), n_lines))

    presence: dict[str, int] = {}
    for lines in cleaned:
        norms = {_norm(lines[i]) for i in edge_positions(len(lines)) if len(lines[i].strip()) >= MIN_LEN}
        for n in norms:
            presence[n] = presence.get(n, 0) + 1
    threshold = max(2, int(MIN_SHARE * n_pages))        # au moins 2 pages : un PDF de loi peut être très court
    repeated = {n: c for n, c in presence.items() if c >= threshold}

    kept: list[str] = []
    dropped = 0
    for lines in cleaned:
        edges = edge_positions(len(lines))
        for i, line in enumerate(lines):
            if any(rx.search(line) for rx in NOISE) or (i in edges and _norm(line) in repeated):
                dropped += 1
            else:
                kept.append(line)

    found = sorted(((n, c) for n, c in repeated.items() if not any(rx.search(n) for rx in NOISE)),
                   key=lambda x: -x[1])
    return kept, dropped, found
