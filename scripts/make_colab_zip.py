"""Prépare colab_retrieval.zip : le strict nécessaire pour mesurer la recherche sur Colab.

    python scripts/make_colab_zip.py

Contenu : src/ (sans caches), data/rules/, data/processed/clauses_test.jsonl.
Le fichier est créé à la racine du projet ; l'envoyer ensuite dans le notebook Colab.
"""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "processed" / "clauses_test.jsonl"


def main() -> None:
    if not TEST.exists():
        sys.exit(f"{TEST} introuvable. Lancer d'abord : python -m src.data.prepare")
    out = ROOT / "colab_retrieval.zip"
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ("src", "data/rules"):
            for p in sorted((ROOT / folder).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                    z.write(p, p.relative_to(ROOT).as_posix())
                    n += 1
        z.write(TEST, TEST.relative_to(ROOT).as_posix())
        n += 1
    print(f"{n} fichiers -> {out} ({out.stat().st_size / 1024:.0f} Ko)")


if __name__ == "__main__":
    main()
