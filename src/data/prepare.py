"""Construit le jeu de clauses etiquetees a partir de CUAD-QA.

Chaque question de CUAD porte sur une categorie juridique, encodee a la fin
du champ 'id'. Les extraits renvoyes dans 'answers.text' sont les clauses du
contrat qui relevent de cette categorie.

    id      : ..._AGREEMENT__Document Name_0   ->  categorie "Document Name"
    answers : {"text": ["DISTRIBUTOR AGREEMENT"], "answer_start": [44]}

On transforme donc chaque extrait annote en une ligne :
    {"text": "<la clause>", "label": "<la categorie>", "doc": "<le contrat>"}

Ecrit data/processed/clauses_train.jsonl et clauses_test.jsonl.

Usage :
    python -m src.data.prepare
"""

import json
import re
from collections import Counter
from pathlib import Path

from src.config import DATA_RAW, DATA_PROCESSED, load_params

CUAD = DATA_RAW / "cuad"

# La categorie est le dernier segment de l'id, suivi d'un index numerique.
CATEGORY_SUFFIX = re.compile(r"_\d+$")


def extract_category(row_id: str) -> str:
    """'..._AGREEMENT__Document Name_0' -> 'Document Name'"""
    tail = row_id.rsplit("__", 1)[-1]
    return CATEGORY_SUFFIX.sub("", tail).strip()


def clean(text: str) -> str:
    """Normalise les espaces et les sauts de ligne d'un extrait."""
    return re.sub(r"\s+", " ", text).strip()


def build_split(path: Path, min_chars: int, max_chars: int) -> list[dict]:
    """Parcourt un fichier CUAD-QA et en extrait les clauses etiquetees."""
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    n_lines = 0
    n_empty = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            row = json.loads(line)

            spans = row.get("answers", {}).get("text", [])
            if not spans:
                n_empty += 1
                continue

            label = extract_category(row["id"])

            for span in spans:
                text = clean(span)
                if not (min_chars <= len(text) <= max_chars):
                    continue
                key = (text, label)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "doc": row.get("title", ""),
                    "text": text,
                    "label": label,
                })

    print(f"  {path.name:14s} {n_lines:>7,} questions | "
          f"{n_empty:>7,} sans reponse | {len(rows):>7,} clauses retenues")
    return rows


def report(rows: list[dict], name: str) -> None:
    """Affiche la repartition par categorie."""
    counts = Counter(r["label"] for r in rows)
    lengths = [len(r["text"]) for r in rows]

    print(f"\n--- {name} ---")
    print(f"  clauses          : {len(rows):,}")
    print(f"  categories       : {len(counts)}")
    if lengths:
        lengths.sort()
        print(f"  longueur mediane : {lengths[len(lengths) // 2]:,} caracteres")
    print(f"  top 10 categories :")
    for label, count in counts.most_common(10):
        print(f"      {label[:45]:45s} {count:>6,}")

    rares = [lbl for lbl, c in counts.items() if c < 20]
    if rares:
        print(f"\n  {len(rares)} categories sous les 20 exemples : "
              f"{', '.join(rares[:5])}{' ...' if len(rares) > 5 else ''}")
        print("  -> candidates a l'augmentation synthetique (corpus v3)")


def main() -> None:
    p = load_params("prepare")
    min_chars = p["min_clause_chars"]
    max_chars = p["max_clause_chars"]

    if not (CUAD / "train.jsonl").exists():
        raise SystemExit(
            f"Fichiers CUAD absents de {CUAD}. "
            "Lance d'abord : python -m src.data.download"
        )

    print("Lecture de CUAD (le fichier train pese ~1,4 Go, compte 2 a 3 minutes)\n")

    splits = {
        "train": build_split(CUAD / "train.jsonl", min_chars, max_chars),
        "test": build_split(CUAD / "test.jsonl", min_chars, max_chars),
    }

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        out = DATA_PROCESSED / f"clauses_{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        report(rows, f"clauses_{name}.jsonl")
        print(f"  ecrit -> {out}")

    total = sum(len(r) for r in splits.values())
    print(f"\nTotal : {total:,} clauses etiquetees, pretes pour l'entrainement.")


if __name__ == "__main__":
    main()
