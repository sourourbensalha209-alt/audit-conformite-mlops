"""Mesures du découpage en clauses, contre les passages annotés par les juristes (jeu CUAD).

Deux niveaux :
  evaluate()        le découpage seul : chaque passage annoté reste-t-il entier dans un segment ?
  score_pipeline()  découpage + classifieur : retrouve-t-on la bonne catégorie, et combien de
                    "clauses fantômes" (catégorie annoncée sur un paragraphe que les juristes n'ont pas annoté) ?
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from .autre import NEUTRAL_LABEL, decide_with_autre
from .segment import HARD_MAX, Segment, segment


def load_docs(path: Path | str) -> list[dict]:
    """Lit un fichier au format de data/raw/cuad/test.jsonl : un dict par contrat (texte + passages annotés)."""
    docs: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = docs.setdefault(r["title"], {"title": r["title"], "text": r["context"], "spans": []})
            cat = re.sub(r"_\d+$", "", r["id"].split("__")[-1])      # le jeu d'entraînement ajoute _0, _1... à la catégorie
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
                    (c'est ce que le seuil, ou la classe "Autre", doit faire baisser)
      abstention %  part de tous les segments laissés "à vérifier par un humain" faute de confiance
      Autre %       part de tous les segments que le modèle classe « Autre » (aucune catégorie annoncée)
    """
    rows = []
    for thr in thresholds:
        wanted = found = 0
        neutral = ghosts = 0
        total = abstained = as_autre = 0
        for doc, segs, preds in zip(docs, segs_by_doc, preds_by_doc):
            ann = annotate_segments(doc, segs)
            hit: set[str] = set()
            for cats, (label, conf) in zip(ann, preds):
                total += 1
                neutral += not cats
                if label == NEUTRAL_LABEL:                     # le modèle dit : "ce n'est aucune catégorie"
                    as_autre += 1
                    continue
                if conf < thr:
                    abstained += 1
                    continue
                if not cats:
                    ghosts += 1
                elif label in cats:
                    hit.add(label)
            expected = {c for c, a, b in doc["spans"] if b > a}
            wanted += len(expected)
            found += len(expected & hit)
        rows.append({"seuil": thr, "rappel %": round(100 * found / max(wanted, 1), 1),
                     "fantômes %": round(100 * ghosts / max(neutral, 1), 1),
                     "abstention %": round(100 * abstained / max(total, 1), 1),
                     "Autre %": round(100 * as_autre / max(total, 1), 1),
                     "contrat-catégories annotées": wanted, "trouvés": found})
    return rows


CRITERE = {"rappel_min": 61.0, "fantomes_max": 28.0, "seuil": 0.5}   # fixé AVANT de voir les résultats de l'itération 3


def sweep_autre(docs, segs_by_doc, probs_by_doc, labels, rules=(None, 0.7, 0.9, 0.97, 0.99), thresholds=(0.0, 0.5)):
    """Pour chaque règle de décision sur « Autre », le tableau rappel / fantômes. Retourne une liste de lignes."""
    rows = []
    for a in rules:
        preds = [[decide_with_autre(p, labels, a) for p in probs] for probs in probs_by_doc]
        for r in score_pipeline(docs, segs_by_doc, preds, thresholds=thresholds):
            rows.append({"règle": "argmax" if a is None else f"{a:g}", **r})
    return rows


def critere_atteint(rows: list[dict]) -> list[str]:
    """Règles qui remplissent le critère (au seuil fixé) : rappel assez haut ET fantômes assez bas."""
    return [r["règle"] for r in rows
            if r["seuil"] == CRITERE["seuil"] and r["rappel %"] >= CRITERE["rappel_min"]
            and r["fantômes %"] <= CRITERE["fantomes_max"]]


METADATA = {"Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date"}   # petits passages dans un en-tête
BUCKETS = [(0, 250), (250, 500), (500, 800), (800, 10 ** 9)]


def diagnose(docs: list[dict], segs_by_doc: list[list[Segment]], preds_by_doc: list[list[tuple[str, float]]]) -> dict:
    """Où les clauses sont-elles perdues ? (prédictions au plus probable, sans seuil)

    par_longueur : pour les segments qui contiennent une clause annotée, selon leur longueur : combien portent la bonne
                   étiquette, combien sont classés « Autre ». Si le rappel s'effondre sur les longs segments, le modèle
                   confond "long" et "Autre".
    par_categorie : rappel par (contrat, catégorie) ; "en-têtes" regroupe les 5 catégories qui sont de petits passages
                    (nom, parties, dates) noyés dans un paragraphe d'en-tête.
    """
    bstats = {b: [0, 0, 0] for b in BUCKETS}
    cats: dict[str, list[int]] = {}
    for doc, segs, preds in zip(docs, segs_by_doc, preds_by_doc):
        hit: set[str] = set()
        for ann, seg, (label, _) in zip(annotate_segments(doc, segs), segs, preds):
            if not ann:
                continue
            b = next(b for b in BUCKETS if b[0] <= len(seg.text) < b[1])
            bstats[b][0] += 1
            bstats[b][1] += label in ann
            bstats[b][2] += label == NEUTRAL_LABEL
            if label in ann:
                hit.add(label)
        for c in {c for c, a, b in doc["spans"] if b > a}:
            cats.setdefault(c, [0, 0])
            cats[c][1] += 1
            cats[c][0] += c in hit

    def rate(items):
        found = sum(cats[c][0] for c in items)
        total = sum(cats[c][1] for c in items)
        return {"trouvés": found, "total": total, "rappel %": round(100 * found / max(total, 1), 1)}

    return {
        "par_longueur": [{"longueur": f"{a}-{b if b < 10 ** 9 else '+'}", "segments avec clause": n,
                          "bonne étiquette %": round(100 * ok / max(n, 1), 1), "classés Autre %": round(100 * au / max(n, 1), 1)}
                         for (a, b), (n, ok, au) in bstats.items()],
        "en_tetes": rate([c for c in cats if c in METADATA]),
        "en_tetes_categories": sorted(c for c in cats if c in METADATA),
        "autres_categories": rate([c for c in cats if c not in METADATA]),
        "pires": sorted(((c, f, t) for c, (f, t) in cats.items() if t >= 3), key=lambda x: x[1] / x[2])[:8],
    }


def pick_contracts(docs: list[dict], n: int, max_chars: int = 120000, exclude_titles=frozenset()) -> list[dict]:
    """n contrats répartis des plus courts aux plus longs (les très gros sont écartés : lents à classer).
    `exclude_titles` retire d'avance certains contrats, par exemple ceux qui ont servi à diagnostiquer le système."""
    pool = sorted((d for d in docs if len(d["text"]) <= max_chars and d["title"] not in exclude_titles),
                  key=lambda d: len(d["text"]))
    if n >= len(pool):
        return pool
    step = (len(pool) - 1) / max(n - 1, 1)
    return [pool[round(i * step)] for i in range(n)]
