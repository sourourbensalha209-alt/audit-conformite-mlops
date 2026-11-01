"""Mesure le découpage en clauses sur les contrats annotés par les juristes (jeu CUAD, test).

    python scripts/eval_segmentation.py                  # lit data/raw/cuad/test.jsonl
    python scripts/eval_segmentation.py --compare        # compare les variantes du découpage

Pour chaque passage annoté, on cherche le segment qui le recouvre le mieux :
  entier  : un seul segment contient au moins 90 % du passage (la clause n'est pas coupée en deux)
  net     : entier ET segment au plus 2,5 fois plus long que le passage (peu de texte autour)
  coupé   : aucun segment ne contient 90 % du passage
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.segment_eval import evaluate, load_docs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/raw/cuad/test.jsonl")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    path = Path(args.test)
    docs = load_docs(path if path.is_absolute() or path.exists() else ROOT / path)
    print(f"{len(docs)} contrats, {sum(len(d['spans']) for d in docs)} passages annotés\n")

    variants = {"version retenue": {}}
    if args.compare:
        variants = {
            "paragraphes bruts": dict(merge_headings=False, join_page_breaks=False, split_long=False),
            "+ titres rattachés": dict(join_page_breaks=False, split_long=False),
            "+ sauts de page recollés": dict(split_long=False),
            "+ coupe des textes longs (retenue)": {},
        }
    cats = {}
    for name, opts in variants.items():
        r = evaluate(docs, **opts)
        cats = r.pop("_per_cat")
        print(f"{name:36s}", r)
    print("\nCatégories les moins bien découpées (% de passages nets) :", sorted(cats.items(), key=lambda kv: kv[1])[:6])


if __name__ == "__main__":
    main()
