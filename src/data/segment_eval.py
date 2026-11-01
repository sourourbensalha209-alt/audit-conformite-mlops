"""Mesures du découpage en clauses, contre les passages annotés par les juristes (jeu CUAD).

Deux niveaux :
  evaluate()        le découpage seul : chaque passage annoté reste-t-il entier dans un segment ?
  score_pipeline()  découpage + classifieur : retrouve-t-on la bonne catégorie, et combien de
                    "clauses fantômes" (catégorie annoncée sur un paragraphe que les juristes n'ont pas annoté) ?
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from .segment import HARD_MAX, Segment, segment


def load_docs(path: Path | str) -> list[dict]:
    """Lit un fichier au format de data/raw/cuad/test.jsonl : un dict par contrat (texte + passages annotés)."""
    docs: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = docs.setdefault(r["title"], {"title": r["title"], "text": r["context"], "spans": []})
            cat = r["id"].split("__")[-1]
            for t, s in zip(r["answers"]["text"], r["answers"]["answer_start"]):
                d["spans"].append((cat, s, s + len(t)))
    return list(docs.values())


def _overlap(a: int, b: int, seg: Segment) -> int:
    return min(b, seg.end) - max(a, seg.start)


def evaluate(docs: list[dict], **opts) -> dict:
    n = covered = tight = cut = 0
    lens, over, counts = [], 0, []
    per_cat: dict[str, list[int]] = {}
    for d in docs:
        segs = segment(d["text"], **opts)
        counts.append(len(segs))
        lens += [len(s.text) for s in segs]
        over += sum(len(s.text) > HARD_MAX for s in segs)
        for cat, a, b in d["spans"]:
            if b <= a or not segs:
                continue
            n += 1
            seg = max(segs, key=lambda s: _overlap(a, b, s))
            ok = _overlap(a, b, seg) / (b - a) >= 0.9
            good = ok and (seg.end - seg.start) <= 2.5 * (b - a) + 60
            covered += ok
            tight += good
            cut += not ok
            per_cat.setdefault(cat, []).append(int(good))
    return {"passages": n, "entiers %": round(100 * covered / n, 1), "nets %": round(100 * tight / n, 1),
            "coupés %": round(100 * cut / n, 1), "segments/contrat (médiane)": int(statistics.median(counts)),
            "longueur médiane": int(statistics.median(lens)), f"au-delà de {HARD_MAX} car. %": round(100 * over / len(lens), 1),
            "_per_cat": {k: round(100 * sum(v) / len(v)) for k, v in per_cat.items()}}


def annotate_segments(doc: dict, segs: list[Segment]) -> list[set[str]]:
    """Pour chaque segment, l'ensemble des catégories dont un passage annoté le recouvre."""
    out: list[set[str]] = []
    for s in segs:
        cats = set()
        for cat, a, b in doc["spans"]:
            if b > a and _overlap(a, b, s) > 0.5 * min(b - a, s.end - s.start):
                cats.add(cat)
        out.append(cats)
    return out


def score_pipeline(docs: list[dict], segs_by_doc: list[list[Segment]], preds_by_doc: list[list[tuple[str, float]]],
                   thresholds=(0.0, 0.5, 0.7, 0.9)) -> list[dict]:
    """preds_by_doc[i][j] = (catégorie prédite, confiance) pour le segment j du contrat i.

    Pour chaque seuil de confiance :
      rappel %      parmi les (contrat, catégorie) annotés par les juristes, part retrouvée sur un segment bon
      fantômes %    parmi les segments SANS annotation, part qui reçoit quand même une catégorie
                    (c'est ce que le seuil, ou une classe "autre", doit faire baisser)
      abstention %  part de tous les segments laissés "à vérifier" faute de confiance
    """
    rows = []
    for thr in thresholds:
        wanted = found = 0
        neutral = ghosts = 0
        total = abstained = 0
        for doc, segs, preds in zip(docs, segs_by_doc, preds_by_doc):
            ann = annotate_segments(doc, segs)
            hit: set[str] = set()
            for cats, (label, conf) in zip(ann, preds):
                total += 1
                if conf < thr:
                    abstained += 1
                    if not cats:
                        neutral += 1
                    continue
                if not cats:
                    neutral += 1
                    ghosts += 1
                elif label in cats:
                    hit.add(label)
            expected = {c for c, a, b in doc["spans"] if b > a}
            wanted += len(expected)
            found += len(expected & hit)
        rows.append({"seuil": thr, "rappel %": round(100 * found / max(wanted, 1), 1),
                     "fantômes %": round(100 * ghosts / max(neutral, 1), 1),
                     "abstention %": round(100 * abstained / max(total, 1), 1),
                     "contrat-catégories annotées": wanted})
    return rows
