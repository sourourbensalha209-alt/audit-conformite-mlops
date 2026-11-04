"""Verdict « manquante » : une clause attendue par la règle n'apparaît nulle part dans le contrat.

Aucun modèle de langage ici : on s'appuie sur le classifieur (catégories de clauses) et sur l'extracteur à règles
(identité du contrat). Une catégorie est déclarée PRÉSENTE si au moins un segment reçoit cette catégorie (réponse la plus
probable, confiance >= tau) ou si l'extracteur trouve l'information ; sinon elle est MANQUANTE.

La vérité de référence vient des juristes CUAD : la catégorie est annotée dans le contrat, ou elle ne l'est pas.
"""

from __future__ import annotations

from ..data.autre import NEUTRAL_LABEL
from ..data.header import detect_header

TAU = 0.5          # fixé d'avance : même seuil que l'essai de bout en bout
CRITERIA = {"recall_min": 60.0, "precision_min": 50.0}      # fixés avant la mesure (docs/criteres_phase5.md)


def present_categories(text: str, probs_by_segment: list[list[float]], labels: list[str], tau: float = TAU) -> set[str]:
    present = set()
    for p in probs_by_segment:
        i = max(range(len(p)), key=p.__getitem__)
        if p[i] >= tau and labels[i] != NEUTRAL_LABEL:
            present.add(labels[i])
    present |= {cat for cat, found in detect_header(text).items() if found is not None}
    return present


def expected_categories(book) -> list[str]:
    """Catégories dont l'absence est à signaler (champ `absence_a_signaler` des règles)."""
    return [r["category"] for r in book.rules if r["absence_a_signaler"]]


def score_missing(docs: list[dict], present_by_doc: list[set[str]], expected: list[str]) -> dict:
    rows = []
    tot = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for cat in expected:
        c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for d, present in zip(docs, present_by_doc):
            truly_absent = not any(k == cat and b > a for k, a, b in d["spans"])
            says_missing = cat not in present
            key = ("tp" if truly_absent else "fp") if says_missing else ("fn" if truly_absent else "tn")
            c[key] += 1
        for k in tot:
            tot[k] += c[k]
        rows.append({"catégorie": cat, **c, "absentes_en_vrai": c["tp"] + c["fn"], "dit_manquante": c["tp"] + c["fp"]})

    def pr(c):
        return (round(100 * c["tp"] / (c["tp"] + c["fp"]), 1) if c["tp"] + c["fp"] else None,
                round(100 * c["tp"] / (c["tp"] + c["fn"]), 1) if c["tp"] + c["fn"] else None)

    precision, recall = pr(tot)
    n = sum(tot.values())
    base = round(100 * (tot["tp"] + tot["fn"]) / n, 1) if n else 0.0
    ok = (precision or 0) >= CRITERIA["precision_min"] and (recall or 0) >= CRITERIA["recall_min"]
    return {"par_categorie": rows, "total": tot, "precision_manquante_pct": precision, "rappel_manquante_pct": recall,
            "taux_de_base_pct": base, "couples": n, "critere_atteint": ok}
