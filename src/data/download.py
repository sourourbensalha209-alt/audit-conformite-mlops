"""Etape 2 du projet : recuperation des deux corpus.

Usage :
    python -m src.data.download

Produit :
    data/raw/cuad/           contrats annotes clause par clause
    data/raw/reglementation/ texte reglementaire decoupe par article
"""

import json
import sys
from pathlib import Path

from src.config import DATA_RAW


def download_cuad() -> Path:
    """Telecharge CUAD depuis le Hub Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Installe d'abord les dependances : pip install -r requirements.txt")

    out = DATA_RAW / "cuad"
    out.mkdir(parents=True, exist_ok=True)

    print("Telechargement de CUAD depuis Hugging Face ...")
    ds = load_dataset("theatticusproject/cuad-qa", trust_remote_code=True)

    for split, rows in ds.items():
        path = out / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {split:10s} {len(rows):>7,} lignes  ->  {path.name}")

    return out


def check_cuad(folder: Path) -> None:
    """Controle de faisabilite : structure, volumetrie, categories."""
    train = folder / "train.jsonl"
    if not train.exists():
        print("Fichier train.jsonl absent, controle impossible.")
        return

    categories: dict[str, int] = {}
    n = 0
    with open(train, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n += 1
            # dans CUAD-QA, la categorie juridique est encodee dans le champ 'id'
            cat = row.get("id", "").split("__")[-1] or "inconnue"
            categories[cat] = categories.get(cat, 0) + 1

    print(f"\nControle CUAD")
    print(f"  exemples          : {n:,}")
    print(f"  categories        : {len(categories)}")
    print(f"  top 5 categories  :")
    for cat, count in sorted(categories.items(), key=lambda kv: -kv[1])[:5]:
        print(f"      {cat[:50]:50s} {count:>6,}")

    if len(categories) < 10:
        print("\n  ATTENTION : moins de 10 categories detectees.")
        print("  Verifie le champ utilise pour l'etiquette avant d'entrainer.")


def main() -> None:
    folder = download_cuad()
    check_cuad(folder)

    reg = DATA_RAW / "reglementation"
    reg.mkdir(parents=True, exist_ok=True)
    print(f"\nPlace le corpus reglementaire decoupe par article dans : {reg}")
    print("Format attendu : un fichier .json par article")
    print('  {"article": "13", "titre": "...", "texte": "..."}')


if __name__ == "__main__":
    main()
