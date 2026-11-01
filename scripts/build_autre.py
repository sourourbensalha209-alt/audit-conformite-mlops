"""Construit les exemples de la classe « Autre » pour l'itération 3 de Legal-BERT.

    python scripts/build_autre.py

Lit data/raw/cuad/train.jsonl et test.jsonl, écrit data/processed/autre_train.jsonl et autre_test.jsonl.
Le train et le test viennent de contrats DIFFÉRENTS (comme le reste du jeu) : aucune fuite.
Ces deux fichiers sont à déposer sur Colab avec clauses_train.jsonl et clauses_test.jsonl.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.autre import neutral_segments, sample_autre  # noqa: E402
from src.data.segment_eval import load_docs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/cuad")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=500)
    args = ap.parse_args()

    raw = Path(args.raw) if Path(args.raw).is_absolute() else ROOT / args.raw
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    for name, n in (("train", args.n_train), ("test", args.n_test)):
        docs = load_docs(raw / f"{name}.jsonl")
        rows, stats = sample_autre(docs, n)
        path = out / f"autre_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        lens = sorted(len(r["text"]) for r in rows)
        print(f"{name:5s} : {stats['retenus']} exemples « Autre » -> {path.name}  "
              f"(sur {stats['paragraphes neutres disponibles']} paragraphes neutres, {stats['contrats']} contrats, "
              f"longueur médiane {lens[len(lens) // 2]} car.)")
        if name == "train":
            print("  Trois exemples tirés au hasard :")
            for r in rows[:3]:
                print("   -", r["text"][:140] + ("..." if len(r["text"]) > 140 else ""))


if __name__ == "__main__":
    main()
