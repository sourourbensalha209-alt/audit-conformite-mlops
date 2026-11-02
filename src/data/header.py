"""Extraction par règles des informations d'IDENTITÉ d'un contrat : nom, parties, date de signature, date d'entrée en vigueur.

Pourquoi des règles plutôt que Legal-BERT : ces quatre catégories CUAD ne sont pas des clauses mais de petits passages
(16 à 25 caractères) dans l'en-tête ou le bloc de signature. Un classifieur paragraphe par paragraphe les confond avec du
texte neutre (mesuré : 14 % retrouvés sur 8 contrats). Des règles explicites sont plus fiables, vérifiables par le
juriste, et mesurables sur des contrats annotés (scripts/eval_header.py).

Chaque détection renvoie le PASSAGE retrouvé, pour que le rapport d'audit montre sa preuve ; None signifie "absent".
La date d'expiration n'est pas ici : c'est une vraie clause de durée, traitée par le classifieur.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HEADER_CATEGORIES = ("Document Name", "Parties", "Agreement Date", "Effective Date")

_MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
# "February 28, 2019" · "Feb. 5th, 2020" · "5 February 2020" · "12/09/2019" · "as of June , 2010" (jour laissé en blanc)
_DATE = re.compile(
    rf"(?:{_MONTHS}\s+(?:\d{{1,2}}\s?(?:st|nd|rd|th)?\s*,?\s*|_*\s*,?\s*)\d{{4}}"
    rf"|\d{{1,2}}\s?(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?{_MONTHS}\s*,?\s*\d{{4}}"
    rf"|\d{{1,2}}[/.-]\d{{1,2}}[/.-]\d{{2,4}}"
    rf"|{_MONTHS}\s*\d{{1,2}}\s?(?:st|nd|rd|th)?\s*,?)",
    re.I)
_DATE_CONTEXT = re.compile(r"(dated|made|entered\s+into|as\s+of|executed|effective|this\s+\w*\s*day|signed|date\s*:)", re.I)
_TITLE_WORDS = (r"AGREEMENT|CONTRACT|LICENSE|LICENCE|LEASE|AMENDMENT|ADDENDUM|MEMORANDUM|DEED|PLAN|POLICY|CHARTER|"
                r"UNDERTAKING|COVENANT|SETTLEMENT|WARRANT|NOTE|GUARANTY|GUARANTEE|SCHEDULE|TERMS|PROTOCOL|OFFER")
_TITLE = re.compile(rf"((?:[A-Z0-9][A-Za-z0-9&/’'\-]*\s+){{0,7}}(?:{_TITLE_WORDS})(?:\s+(?:OF|FOR|AND|TO|&)\s+[A-Z][A-Za-z]+){{0,3}}"
                    rf"(?:\s+\([A-Za-z ]+\))?)", re.I)
_PARTIES = re.compile(
    r"(between|by\s+and\s+among|among|by\s+and\s+between|entered\s+into\s+by|made\s+by|"
    r"(?:is|are)\s+made\s+and\s+entered\s+into\s+(?:as\s+of\s+[^,]{0,40},?\s*)?by|party\s+of\s+the\s+first\s+part)", re.I)
_EFFECTIVE = re.compile(
    r"(effective\s+date|effective\s+as\s+of|effective\s+on|effective\s+this|shall\s+(?:be\s+)?(?:come\s+)?effective|"
    r"becomes?\s+effective|date\s+last\s+(?:written|signed)|commenc(?:e|ing)\s+(?:on|as\s+of))", re.I)


@dataclass
class Found:
    category: str
    start: int
    end: int
    text: str


def _found(cat: str, src: str, a: int, b: int) -> Found:
    return Found(cat, a, b, " ".join(src[a:b].split()))


def _document_name(src: str) -> Found | None:
    head = src[:700]
    best = None
    for m in _TITLE.finditer(head):
        words = m.group(1).split()
        if len(words) < 2:
            continue
        # un titre est écrit en MAJUSCULES le plus souvent ; on préfère la première occurrence en capitales
        caps = sum(w.isupper() for w in words) / len(words)
        if best is None or (caps >= 0.6 and best[1] < 0.6):
            best = (m, caps)
        if caps >= 0.6:
            break
    if best is None:
        return None
    m = best[0]
    return _found("Document Name", src, m.start(1), m.end(1))


def _agreement_date(src: str) -> Found | None:
    """Première date, dans les 8 000 premiers caractères, introduite par un mot de contexte (dated, as of, entered into...)."""
    head = src[:8000]
    for m in _DATE.finditer(head):
        before = head[max(0, m.start() - 90):m.start()]
        if _DATE_CONTEXT.search(before):
            return _found("Agreement Date", src, m.start(), m.end())
    # signature : "IN WITNESS WHEREOF ... as of <date>"
    for m in re.finditer(r"IN\s+WITNESS\s+WHEREOF", src, re.I):
        tail = src[m.end():m.end() + 220]
        d = _DATE.search(tail)
        if d:
            return _found("Agreement Date", src, m.end() + d.start(), m.end() + d.end())
    return None


_COMPANY = re.compile(r"\b(?:Inc\.?|Incorporated|Corp(?:oration)?\.?|LLC|L\.L\.C\.|L\.P\.|LP|Ltd\.?|Limited|GmbH|S\.A\.|N\.V\.|"
                      r"Company|Co\.|Holdings|Partners|Bank|Trust|Fund|University|Association)\b")


def _parties(src: str) -> Found | None:
    """Passage qui désigne les parties : "between X and Y", "by and among...", sinon la première raison sociale de l'en-tête."""
    head = src[:2500]
    m = _PARTIES.search(head)
    if m:
        return _found("Parties", src, m.start(), min(len(head), m.end() + 260))
    c = _COMPANY.search(src[:1800])
    if c:
        return _found("Parties", src, max(0, c.start() - 120), min(len(src), c.end() + 160))
    return None


_DEFINED = re.compile(r"[\"“”'‘’]\s*Effective\s+Date\s*[\"“”'‘’]|Effective\s+Date\s*:|date\s+last\s+(?:written|signed)|date\s+first\s+(?:written|above)", re.I)
_EFFECTIVE_ON = re.compile(r"(?:made\s+|become(?:s)?\s+|shall\s+be\s+|is\s+|are\s+)?effective\s+(?:as\s+of|on|this|upon|from|date\s+of)", re.I)


def _nearest_date(window: str, anchor: int):
    best = None
    for d in _DATE.finditer(window):
        dist = min(abs(d.start() - anchor), abs(d.end() - anchor))
        if best is None or dist < best[0]:
            best = (dist, d)
    return best[1] if best else None


def _effective_date(src: str) -> Found | None:
    """Date d'entrée en vigueur, dans les 15 000 premiers caractères. On privilégie les formes qui DÉFINISSENT la date
    ("(the "Effective Date")", "Effective Date:", "date last written below"), puis "effective as of / on <date>".
    Si une date est écrite à côté, c'est elle qu'on renvoie ; sinon la formule."""
    head = src[:15000]
    for pattern in (_DEFINED, _EFFECTIVE_ON):
        for m in pattern.finditer(head):
            lo, hi = max(0, m.start() - 90), min(len(head), m.end() + 90)
            d = _nearest_date(head[lo:hi], m.start() - lo)
            if d:
                return _found("Effective Date", src, lo + d.start(), lo + d.end())
            if pattern is _DEFINED and re.search(r"date\s+(?:last|first)", m.group(0), re.I):
                return _found("Effective Date", src, m.start(), m.end())
    # repli : la date est définie sans être écrite ("(the "Effective Date")" avec une date laissée en blanc, etc.)
    m = _DEFINED.search(head)
    if m:
        return _found("Effective Date", src, max(0, m.start() - 60), min(len(src), m.end() + 20))
    m = _EFFECTIVE_ON.search(head)
    if m:
        return _found("Effective Date", src, m.start(), min(len(src), m.end() + 60))
    return None


def detect_header(src: str) -> dict[str, Found | None]:
    return {"Document Name": _document_name(src), "Parties": _parties(src),
            "Agreement Date": _agreement_date(src), "Effective Date": _effective_date(src)}


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def same_value(found_text: str, truth_texts: list[str]) -> bool:
    """Le texte trouvé contient un texte annoté (ou l'inverse) : même titre, même date, mêmes parties."""
    f = _norm(found_text)
    return any(len(_norm(t)) >= 4 and (_norm(t) in f or (len(f) >= 4 and f in _norm(t))) for t in truth_texts)


def score_header(docs: list[dict]) -> dict[str, tuple[int, int]]:
    """Par catégorie d'identité : (contrats où la bonne valeur est retrouvée, contrats où la catégorie est annotée)."""
    out = {c: [0, 0] for c in HEADER_CATEGORIES}
    for d in docs:
        found = detect_header(d["text"])
        for cat in HEADER_CATEGORIES:
            truth = [" ".join(d["text"][a:b].split()) for c, a, b in d["spans"] if c == cat and b > a]
            if truth:
                out[cat][1] += 1
                out[cat][0] += found[cat] is not None and same_value(found[cat].text, truth)
    return {c: (f, t) for c, (f, t) in out.items()}
