"""Etape 1 du pipeline : extraction du texte, OCR compris.

Ecrit data/interim/documents.jsonl : une ligne par document.
"""

import json
from pathlib import Path

from src.config import DATA_RAW, DATA_INTERIM


def extract_pdf_native(path: Path) -> str:
    """Texte d'un PDF natif."""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def extract_pdf_ocr(path: Path) -> str:
    """Texte d'un PDF scanne, via OCR."""
    import pytesseract
    from pdf2image import convert_from_path

    pages = convert_from_path(path, dpi=300)
    return "\n".join(pytesseract.image_to_string(p, lang="fra+eng") for p in pages)


def extract(path: Path, ocr_threshold: int = 100) -> tuple[str, str]:
    """Retourne (texte, methode). Bascule sur l'OCR si le PDF est pauvre en texte."""
    text = extract_pdf_native(path)
    if len(text.strip()) < ocr_threshold:
        return extract_pdf_ocr(path), "ocr"
    return text, "native"


def main() -> None:
    out = DATA_INTERIM / "documents.jsonl"
    docs = list(DATA_RAW.rglob("*.pdf"))

    with open(out, "w", encoding="utf-8") as f:
        for path in docs:
            text, method = extract(path)
            f.write(json.dumps(
                {"source": path.name, "method": method, "text": text},
                ensure_ascii=False,
            ) + "\n")
            print(f"{path.name:50s} {method:8s} {len(text):>8,} caracteres")

    print(f"\n{len(docs)} documents -> {out}")


if __name__ == "__main__":
    main()
