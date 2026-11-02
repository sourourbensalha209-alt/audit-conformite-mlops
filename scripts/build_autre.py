"""Construit les données de l'itération 4 de Legal-BERT.

    python scripts/build_autre.py

Lit data/raw/cuad/train.jsonl et test.jsonl, écrit dans data/processed :
  autre_train.jsonl / autre_test.jsonl        paragraphes neutres, étiquette « Autre »
  segments_train.jsonl / segments_test.jsonl  clauses annotées prises sous la forme du segment qui les contient

Deux principes :
  * le classifieur ne traite plus les 4 catégories d'identité (nom, parties, date de signature, date d'entrée en vigueur) :
    l'extracteur à règles s'en charge (src/data/header.py). Un bloc de signature est donc « Autre », sans contradiction.
  * l'entraînement et le test viennent de contrats DIFFÉRENTS (comme le reste du jeu) : aucune fuite.
À déposer sur Colab avec clauses_train.jsonl et clauses_test.jsonl (les 6 fichiers).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.autre import positive_segments, sample_autre  # noqa: E402
from src.data.segment_eval import load_docs  # noqa: E402


def write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def median_len(rows: list[dict]) -> int:
    lens = sorted(len(r["text"]) for r in rows)
    return lens[len(lens) // 2] if lens else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/cuad")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=1500)
    args = ap.parse_args()

    raw = Path(args.raw) if Path(args.raw).is_absolute() else ROOT / args.raw
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    for name, n in (("train", args.n_train), ("test", args.n_test)):
        docs = load_docs(raw / f"{name}.jsonl")
        autre, st = sample_autre(docs, n)
        positives = [r for d in docs for r in positive_segments(d)]
        write(out / f"autre_{name}.jsonl", autre)
        write(out / f"segments_{name}.jsonl", positives)
        print(f"{name:5s} : {len(autre):>5} « Autre » (sur {st['paragraphes neutres disponibles']} paragraphes neutres, "
              f"médiane {median_len(autre)} car.) | {len(positives):>5} clauses sous forme de segment "
              f"({len({r['label'] for r in positives})} catégories, médiane {median_len(positives)} car.)")
        if name == "train":
            print("  Trois exemples « Autre » tirés au hasard :")
            for r in autre[:3]:
                print("   -", r["text"][:130] + ("..." if len(r["text"]) > 130 else ""))


if __name__ == "__main__":
    main()
