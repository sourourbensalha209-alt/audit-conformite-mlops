"""Démonstration pour la soutenance : une nouvelle loi est publiée, le système la prend en compte seul.

    python scripts/demo_veille.py            # BM25 seul, aucun téléchargement
    python scripts/demo_veille.py --dense    # avec l'encodeur E5 multilingue (télécharge le modèle)

Les textes de loi sont FICTIFS (numéros en 9xxx) : ils illustrent la mécanique, pas le droit tunisien.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.corpus import CorpusStore  # noqa: E402
from src.retrieval.index import HybridIndex  # noqa: E402
from src.retrieval.law_watch import Source, Watcher, print_report  # noqa: E402

LOI_BASE = """Loi fictive n° 9001-01 relative à la conservation des données contractuelles
Article premier. - La présente loi fixe les règles de conservation des données échangées dans les contrats.
Art. 2. - Les données personnelles sont conservées cinq ans au plus après la fin du contrat.
Art. 3. - Le contrat désigne le sous-traitant chargé de la conservation et ses obligations.
"""

LOI_NOUVELLE = """# publication_date: 2026-10-15
Loi fictive n° 9002-05 portant modification de la loi n° 9001-01 relative à la conservation des données
Article premier. - Les données personnelles sont conservées trois ans au plus après la fin du contrat.
Art. 2. - Tout contrat de sous-traitance prévoit la notification des violations de données sous 48 heures.
"""

QUESTION = "durée maximale de conservation des données personnelles après la fin du contrat"


def show(index, label):
    print(f"\n  Question de l'audit : « {QUESTION} »  [{label}]")
    for h in index.search(QUESTION, k=3):
        flag = []
        if not h.article.validated:
            flag.append("non validé")
        if h.article.under_review:
            flag.append("à vérifier")
        extra = f"  ({', '.join(flag)})" if flag else ""
        print(f"    - {h.article.citation()}{extra}\n      {h.article.text[:95]}")


def main() -> None:
    dense = "--dense" in sys.argv
    embedder = None
    if dense:
        from src.retrieval.index import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "raw").mkdir()
        (root / "inbox").mkdir()
        (root / "raw" / "loi_base.txt").write_text(LOI_BASE, encoding="utf-8")
        sources = [
            Source(id="loi-9001-01", title="Loi fictive n° 9001-01", kind="file", path="raw/loi_base.txt",
                   jurisdiction="TN", aliases=["9001-01"]),
            Source(id="inbox", title="Textes déposés", kind="inbox", path="inbox", jurisdiction="TN",
                   auto_validate=False),
        ]
        store = CorpusStore(root / "corpus")
        index = HybridIndex(embedder, cache_path=root / "corpus" / "emb.npz")
        watcher = Watcher(sources, store, root=root, index=index)

        print("\n#### 1. Premier passage : le corpus est construit")
        print_report(watcher.run_once())
        show(index, f"corpus {index.version}")

        print("\n\n#### 2. Une nouvelle loi est publiée (modifie la loi n° 9001-01)")
        (root / "inbox" / "loi_9002_05.txt").write_text(LOI_NOUVELLE, encoding="utf-8")

        print("\n#### 3. Passage suivant de la veille : aucune intervention humaine, aucun réentraînement")
        print_report(watcher.run_once())
        show(index, f"corpus {index.version}")

        print("\n\n#### 4. Traçabilité : historique de l'article modifié")
        for art_id in store.load():
            for row in store.history(art_id):
                if row["change"] == "flagged":
                    print(f"  {art_id}: signalé à vérifier ({row['note']})")
                    break


if __name__ == "__main__":
    main()
