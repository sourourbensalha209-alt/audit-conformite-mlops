"""Pose une question au corpus de lois et affiche les articles les plus proches.

    python scripts/ask_corpus.py "durée de conservation des données personnelles"
    python scripts/ask_corpus.py "sous-traitant instructions documentées" --k 3
    python scripts/ask_corpus.py "retention of personal data after termination" --dense

Sert à vérifier que le corpus est bien chargé, et de démonstration : on voit la citation de l'article,
son état (non validé, à vérifier) et un extrait du texte. Par défaut : BM25 (rapide, sans modèle).
Avec --dense : BGE-M3, le modèle retenu (téléchargé la première fois, environ 2 Go ; sur un PC sans GPU
l'encodage du corpus prend quelques minutes, puis il est gardé en cache).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.corpus import CorpusStore  # noqa: E402
from src.retrieval.index import HybridIndex  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--corpus-dir", default="data/corpus")
    args = ap.parse_args()

    store = CorpusStore(ROOT / args.corpus_dir)
    articles = store.load()
    if not articles:
        sys.exit("Corpus vide. Lancez d'abord : python -m src.retrieval.law_watch")

    embedder = None
    if args.dense:
        from src.retrieval.index import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(args.model)
    index = HybridIndex(embedder, cache_path=store.root / "embeddings.npz" if embedder else None)
    index.ensure_fresh(store)

    laws = {a.source_id for a in articles.values() if a.source_id != "rules"}
    print(f"Corpus {index.version} : {len(articles)} articles ({len(laws)} textes de loi + règles)")
    hits = index.search(args.question, k=args.k, mode="hybrid" if args.dense else "bm25")
    if not hits:
        print("Aucun résultat : essayez avec d'autres mots, ou avec --dense.")
        return
    print(f"\nQuestion : {args.question}\n")
    for i, h in enumerate(hits, 1):
        a = h.article
        flags = [f for f, on in (("non validé", not a.validated), ("à vérifier", a.under_review)) if on]
        print(f"{i}. {a.citation()}" + (f"   [{', '.join(flags)}]" if flags else ""))
        print("   " + " ".join(a.text.split())[:220] + ("..." if len(a.text) > 220 else ""))


if __name__ == "__main__":
    main()
