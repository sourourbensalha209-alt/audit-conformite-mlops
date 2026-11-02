"""Mesure l'extracteur d'identité (nom, parties, date de signature, date d'entrée en vigueur) sur les contrats annotés.

    python scripts/eval_header.py                  # test (102 contrats) ET entraînement (408), pour comparer
    python scripts/eval_header.py --split test

Pour chaque catégorie, au niveau du CONTRAT :
  présent %    part des contrats où les juristes ont annoté cette catégorie (le taux de base)
  trouvé %     part des contrats où l'extracteur trouve quelque chose
  précision %  quand il trouve, la catégorie est-elle bien annotée dans ce contrat ?
  rappel %     quand elle est annotée, l'extracteur trouve-t-il quelque chose ?
  même endroit % quand elle est annotée ET trouvée, le passage trouvé recouvre-t-il un passage annoté ? (mesure stricte :
                 CUAD annote parfois la même information à plusieurs endroits, la bonne valeur peut être ailleurs)
  même valeur %  le texte trouvé contient le texte annoté, ou l'inverse (même titre, même date, mêmes parties)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.header import HEADER_CATEGORIES, detect_header, same_value as _same_value  # noqa: E402
from src.data.segment_eval import load_docs  # noqa: E402


def evaluate(docs: list[dict]) -> list[dict]:
    rows = []
    found = {d["title"]: detect_header(d["text"]) for d in docs}
    docs_by_title = {d["title"]: d for d in docs}
    for cat in HEADER_CATEGORIES:
        truth = {d["title"]: [(a, b) for c, a, b in d["spans"] if c == cat and b > a] for d in docs}
        n = len(docs)
        present = sum(bool(v) for v in truth.values())
        hit_any = sum(found[t][cat] is not None for t in truth)
        tp = sum(bool(truth[t]) and found[t][cat] is not None for t in truth)
        value = sum(
            bool(truth[t]) and found[t][cat] is not None
            and _same_value(found[t][cat].text, [" ".join(docs_by_title[t]["text"][a:b].split()) for a, b in truth[t]])
            for t in truth)
        evid = sum(
            bool(truth[t]) and found[t][cat] is not None
            and any(min(b, found[t][cat].end) - max(a, found[t][cat].start) > 0 for a, b in truth[t])
            for t in truth)
        rows.append({"catégorie": cat, "présent %": round(100 * present / n, 1), "trouvé %": round(100 * hit_any / n, 1),
                     "précision %": round(100 * tp / max(hit_any, 1), 1), "rappel %": round(100 * tp / max(present, 1), 1),
                     "même endroit %": round(100 * evid / max(tp, 1), 1),
                     "même valeur %": round(100 * value / max(tp, 1), 1)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--raw", default="data/raw/cuad")
    args = ap.parse_args()
    raw = Path(args.raw) if Path(args.raw).is_absolute() else ROOT / args.raw
    for split in (["test", "train"] if args.split == "both" else [args.split]):
        path = raw / f"{split}.jsonl"
        if not path.exists():
            print(f"{path} introuvable, passé.")
            continue
        docs = load_docs(path)
        print(f"\n=== {split} : {len(docs)} contrats ===")
        print(f"{'catégorie':16s} | {'présent %':>9} | {'trouvé %':>8} | {'précision %':>11} | {'rappel %':>8} | {'même endroit %':>14} | {'même valeur %':>13}")
        for r in evaluate(docs):
            print(f"{r['catégorie']:16s} | {r['présent %']:>9} | {r['trouvé %']:>8} | {r['précision %']:>11} | {r['rappel %']:>8} | {r['même endroit %']:>14} | {r['même valeur %']:>13}")


if __name__ == "__main__":
    main()
