"""Etape 2 du pipeline : decoupage en clauses et split train/test.

Ecrit data/processed/clauses_train.jsonl et clauses_test.jsonl.
"""

import json
import re

from sklearn.model_selection import train_test_split

from src.config import DATA_INTERIM, DATA_PROCESSED, load_params

# Une clause commence typiquement par une numerotation : "3.", "3.1", "Article 4"
CLAUSE_START = re.compile(
    r"^\s*(?:(?:Article|ARTICLE)\s+\d+|\d+(?:\.\d+)*\.?)\s",
    re.MULTILINE,
)


def split_clauses(text: str, min_chars: int, max_chars: int) -> list[str]:
    """Decoupe un document en clauses."""
    positions = [m.start() for m in CLAUSE_START.finditer(text)]
    if not positions:
        positions = [0]

    chunks = []
    for start, end in zip(positions, positions[1:] + [len(text)]):
        chunk = text[start:end].strip()
        if len(chunk) < min_chars:
            continue
        # une clause trop longue est redecoupee par paragraphes
        while len(chunk) > max_chars:
            cut = chunk.rfind("\n\n", 0, max_chars)
            cut = cut if cut > min_chars else max_chars
            chunks.append(chunk[:cut].strip())
            chunk = chunk[cut:].strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)
    return chunks


def main() -> None:
    p = load_params("prepare")
    rows = []

    with open(DATA_INTERIM / "documents.jsonl", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            for i, clause in enumerate(
                split_clauses(doc["text"], p["min_clause_chars"], p["max_clause_chars"])
            ):
                rows.append({
                    "doc": doc["source"],
                    "clause_id": i,
                    "text": clause,
                    "label": None,   # a renseigner depuis les annotations CUAD
                })

    train, test = train_test_split(
        rows, test_size=p["test_size"], random_state=p["random_state"]
    )

    for name, subset in (("train", train), ("test", test)):
        path = DATA_PROCESSED / f"clauses_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in subset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name:6s} {len(subset):>7,} clauses -> {path.name}")


if __name__ == "__main__":
    main()
