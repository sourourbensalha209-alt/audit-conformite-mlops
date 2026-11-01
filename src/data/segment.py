"""Découpage d'un contrat en clauses : étape 2 de la chaîne d'audit.

Entrée : le texte d'un contrat (sorti de extract.py pour un PDF, ou du jeu CUAD).
Sortie : une liste de segments (début, fin, texte), chacun étant une clause, ou un titre rattaché à sa clause.
`start` et `end` sont des positions dans le texte d'ORIGINE : c'est ce qui permet de mesurer le découpage
contre les passages annotés par les juristes (scripts/eval_segmentation.py).

Le contrat est écrit en paragraphes séparés par des lignes vides ; c'est la seule structure fiable
(le jeu CUAD n'a aucun retour à la ligne simple). L'algorithme :
  1. découpe aux lignes vides ;
  2. écarte le bruit : numéros de page, pieds de page "Source : ..." ;
  3. rattache un titre ("LIMITATION OF LIABILITY", "21. Law application") au paragraphe qui le suit ;
  4. recolle une phrase coupée par un saut de page ;
  5. coupe les paragraphes trop longs aux fins de phrase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_CHARS = 600           # taille visée : on coupe aux fins de phrase pour rester sous cette limite
HARD_MAX = 1000           # fenêtre de Legal-BERT (256 jetons, environ 1 000 caractères) : jamais au-delà
MIN_CHARS = 25            # en dessous, un segment n'est pas une clause

_BLANK = re.compile(r"\n\s*\n")
_TERMINAL = tuple(".;:!?)”\"'")
_PAGE_NUMBER = re.compile(
    r"^\s*(?:page\s+)?[-–—]?\s*(?:\d{1,3}|[ivxlc]{1,6})\s*[-–—]?\s*(?:of\s+\d{1,3})?\s*$", re.I)
_FOOTER = re.compile(r"^\s*(?:Source\s*:|CONFIDENTIAL)", re.I)
# Frontières où couper un texte trop long, de la plus forte à la plus faible.
_CUT_PATTERNS = [
    re.compile(r"(?<=[.;])\s+(?=[A-Z(\"“]|\d+(?:\.\d+)*\.?\s+[A-Z])"),   # fin de phrase, ou début d'une clause numérotée
    re.compile(r"(?<=[:;])\s+"),                                              # deux-points, point-virgule
    re.compile(r"(?<=,)\s+"),                                                 # virgule
    re.compile(r"\s+"),                                                       # dernier recours : un espace
]


@dataclass
class Segment:
    start: int
    end: int
    text: str


def _make(src: str, start: int, end: int) -> Segment:
    return Segment(start, end, " ".join(src[start:end].split()))


def _paragraph_spans(src: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for m in list(_BLANK.finditer(src)) + [None]:
        stop = m.start() if m else len(src)
        chunk = src[pos:stop]
        if chunk.strip():
            lead = len(chunk) - len(chunk.lstrip())
            spans.append((pos + lead, pos + len(chunk.rstrip())))
        if m:
            pos = m.end()
    return spans


def _is_noise(t: str) -> bool:
    return bool(_PAGE_NUMBER.match(t)) or (bool(_FOOTER.match(t)) and len(t) < 160)


def _is_heading(t: str) -> bool:
    """Court et sans ponctuation finale : un titre, ou un début de phrase coupé."""
    return len(t) < 100 and not t.endswith(_TERMINAL)


_FUNCTION_END = re.compile(r"\b(?:the|of|to|a|an|and|or|by|in|for|with|on|at|as|under|such|any|its|their|this|that|which|be|shall|may|from)$", re.I)


def _continues(cur: str, nxt: str) -> bool:
    """Le paragraphe suivant prolonge-t-il une phrase coupée par un saut de page ?
    Oui si le premier est long, ne finit pas sur une ponctuation, et que la suite commence par une minuscule,
    par un numéro de page collé au texte ("4 Agreement, which..."), ou que le premier finit sur un mot-outil."""
    if len(cur) < 100 or cur.endswith(_TERMINAL):
        return False
    return bool(re.match(r"^[a-z]", nxt) or re.match(r"^\d{1,3}\s+[A-Za-z]", nxt) or _FUNCTION_END.search(cur))


def segment(src: str, *, merge_headings: bool = True, join_page_breaks: bool = True,
            split_long: bool = True) -> list[Segment]:
    paras = [_make(src, a, b) for a, b in _paragraph_spans(src)]
    paras = [p for p in paras if not _is_noise(p.text)]

    merged: list[Segment] = []
    i = 0
    while i < len(paras):
        cur = paras[i]
        while (merge_headings and _is_heading(cur.text) and i + 1 < len(paras)
               and not _is_heading(paras[i + 1].text)):
            cur = _make(src, cur.start, paras[i + 1].end)
            i += 1
        while join_page_breaks and i + 1 < len(paras) and _continues(cur.text, paras[i + 1].text):
            cur = _make(src, cur.start, paras[i + 1].end)
            i += 1
        merged.append(cur)
        i += 1

    out: list[Segment] = []
    for s in merged:
        out.extend(_split_long(src, s) if split_long else [s])
    return [s for s in out if len(s.text) >= MIN_CHARS]


def _split_long(src: str, s: Segment) -> list[Segment]:
    """Coupe un segment de plus de MAX_CHARS. À chaque pas, on prend la frontière la plus forte disponible :
    fin de phrase sous MAX_CHARS ; sinon fin de phrase sous HARD_MAX (on garde la phrase entière plutôt que
    de la couper) ; sinon deux-points / point-virgule, virgule, espace, toujours sous HARD_MAX."""
    if len(s.text) <= MAX_CHARS:
        return [s]
    plan = [(_CUT_PATTERNS[0], MAX_CHARS), (_CUT_PATTERNS[0], HARD_MAX)] + [(p, HARD_MAX) for p in _CUT_PATTERNS[1:]]
    out, a = [], s.start
    while s.end - a > HARD_MAX or (s.end - a > MAX_CHARS and _has_cut(src, a, s.end)):
        cut = None
        for pat, limit in plan:
            cands = [m.end() for m in pat.finditer(src, a, s.end) if MIN_CHARS <= m.end() - a <= limit]
            if cands:
                cut = max(cands)
                break
        if cut is None:
            cut = a + HARD_MAX
        out.append(_make(src, a, cut))
        a = cut
    if s.end > a:
        out.append(_make(src, a, s.end))
    return out


def _has_cut(src: str, a: int, b: int) -> bool:
    """Reste-t-il une fin de phrase exploitable avant la fin ? Sinon on garde le reste d'un seul tenant (<= HARD_MAX)."""
    return any(MIN_CHARS <= m.end() - a <= MAX_CHARS for m in _CUT_PATTERNS[0].finditer(src, a, b))
