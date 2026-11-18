"""Lecture d'un contrat en PDF : reconstruit les paragraphes avant le découpage en clauses.

Problème : pdfplumber rend une ligne de texte par ligne de la page, sans ligne vide entre les paragraphes.
Or segment() (src/data/segment.py) découpe aux lignes vides. Sans cette étape, un contrat PDF entier
ressortirait en un seul bloc, coupé ensuite au hasard tous les 600 caractères.

Règle utilisée pour décider qu'une ligne commence un nouveau paragraphe :
  1. c'est un titre de clause ("1.", "2.3", "Article 5", "ARTICLE V", "Section 4", "(a)") et la ligne
     précédente finit par une ponctuation finale (sinon c'est un renvoi : "...conformément à / Section 8.") ;
  2. c'est un titre en majuscules ("LIMITATION OF LIABILITY") ;
  3. la ligne précédente finit par une ponctuation finale ET est nettement plus courte que la largeur
     habituelle des lignes (dernière ligne d'un paragraphe).
Sinon, la ligne prolonge le paragraphe en cours.

    from src.data.paragraphs import pdf_to_contract_text
    text = pdf_to_contract_text("mon_contrat.pdf")     # texte prêt pour segment()
"""

from __future__ import annotations

import re
from pathlib import Path

_NUMBERED = re.compile(
    r"^\s*(?:"
    r"\d{1,2}(?:\.\d{1,2})*\.\s+[A-Z(]"                    # "1. The" / "2.3. Each" (le point évite "30 days")
    r"|\d{1,2}(?:\.\d{1,2})+\s+[A-Z(]"                     # "2.3 Each"
    r"|(?:article|art\.|section|clause|annex|annexe|schedule|exhibit)\s+[\dIVXLC]+"
    r"|\(?[a-z]\)\s+\S"                                     # (a) / a)
    r"|\(?[ivx]{1,4}\)\s+\S"                                # (iv)
    r")",
    re.I,
)
_TERMINAL = tuple(".;:!?)”\"'")
SHORT_RATIO = 0.75     # une ligne plus courte que 75 % de la largeur habituelle peut finir un paragraphe


def _is_caps_heading(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return 3 <= len(letters) and len(line) < 80 and all(c.isupper() for c in letters)


def rebuild_paragraphs(lines: list[str]) -> str:
    """Lignes d'un PDF (bruit de page déjà retiré) -> texte avec une ligne vide entre paragraphes."""
    lines = [" ".join(l.split()) for l in lines if l.strip()]
    if not lines:
        return ""
    widths = sorted(len(l) for l in lines)
    typical = widths[int(len(widths) * 0.8)]           # largeur d'une ligne pleine (80e centile)

    paras: list[list[str]] = [[lines[0]]]
    for prev, line in zip(lines, lines[1:]):
        prev_closed = prev.endswith(_TERMINAL)
        prev_ends_para = prev_closed and len(prev) < SHORT_RATIO * typical
        # Un numéro en début de ligne n'ouvre une clause que si la phrase d'avant est finie :
        # "...in accordance with / Section 8." est un renvoi, pas une nouvelle clause.
        numbered = bool(_NUMBERED.match(line)) and prev_closed
        starts_para = numbered or _is_caps_heading(line) or _is_caps_heading(prev)
        if starts_para or prev_ends_para:
            paras.append([line])
        else:
            paras[-1].append(line)
    return "\n\n".join(_join(p) for p in paras)


def _join(lines: list[str]) -> str:
    """Recolle les lignes d'un paragraphe, en réparant les mots coupés par un trait d'union en fin de ligne."""
    out = lines[0]
    for l in lines[1:]:
        if out.endswith("-") and l[:1].islower():
            out = out[:-1] + l
        else:
            out += " " + l
    return out


def pdf_to_contract_text(path: str | Path, ocr_threshold: int = 100) -> str:
    """PDF -> texte prêt pour segment(). Bascule sur l'OCR si le PDF est scanné."""
    import pdfplumber

    from src.retrieval.textclean import strip_page_noise

    with pdfplumber.open(path) as pdf:
        pages = [(p.extract_text() or "").splitlines() for p in pdf.pages]
    if sum(len(l) for page in pages for l in page) < ocr_threshold:
        from src.data.extract import extract_pdf_ocr
        pages = [extract_pdf_ocr(Path(path)).splitlines()]
    kept, _, _ = strip_page_noise(pages)
    return rebuild_paragraphs(kept)


def docx_to_contract_text(path: str | Path) -> str:
    """Word -> texte prêt pour segment() : les paragraphes Word sont déjà séparés."""
    from docx import Document

    return "\n\n".join(p.text.strip() for p in Document(path).paragraphs if p.text.strip())
