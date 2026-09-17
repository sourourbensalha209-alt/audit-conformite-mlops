"""Construction de l'index de recherche sur le corpus reglementaire.

Produit un index BM25 et un index dense, tous deux utilises par la
recherche hybride.
"""

import json
import pickle
from pathlib import Path

from src.config import DATA_RAW, DATA_PROCESSED, load_params


def load_articles() -> list[dict]:
    """Charge les articles reglementaires depuis data/raw/reglementation."""
    articles = []
    folder = DATA_RAW / "reglementation"
    for path in sorted(folder.glob("*.json")):
        articles.append(json.loads(path.read_text(encoding="utf-8")))
    return articles


def main() -> None:
    p = load_params("retrieval")
    articles = load_articles()

    if not articles:
        raise SystemExit(
            f"Aucun article trouve dans {DATA_RAW / 'reglementation'}. "
            "Lance d'abord python -m src.data.download."
        )

    out = DATA_PROCESSED / "index"
    out.mkdir(parents=True, exist_ok=True)
    texts = [a["texte"] for a in articles]

    # --- Index lexical BM25 ---
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([t.lower().split() for t in texts])
    with open(out / "bm25.pkl", "wb") as f:
        pickle.dump({"index": bm25, "articles": articles}, f)

    # --- Index dense ---
    from sentence_transformers import SentenceTransformer
    import numpy as np

    encoder = SentenceTransformer(p["dense_model"])
    embeddings = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    np.save(out / "dense.npy", embeddings)

    print(f"{len(articles)} articles indexes -> {out}")


if __name__ == "__main__":
    main()
