"""Convertit le PDF d'une loi en texte propre, prêt pour la veille juridique.

    python scripts/pdf_to_txt.py data/corpus_raw/rgpd_fr.pdf data/corpus_raw/rgpd_fr.txt

Ce que fait le script :
  1. extrait le texte de chaque page (pdfplumber) ;
  2. retire le "bruit" qui s'insère au milieu des articles : numéros de page, "Page 3 sur 18",
     en-têtes du Journal officiel de l'UE, et tout en-tête ou pied de page qui se répète sur au moins
     40 % des pages (par exemple le bandeau du site qui publie le PDF). Seules les premières et dernières
     lignes de chaque page sont concernées : le corps du texte n'est jamais touché. Les lignes détectées
     automatiquement sont affichées pour que vous puissiez les vérifier ;
  3. vérifie le résultat : combien d'articles sont reconnus, et s'il en manque dans la numérotation.

Attention : une ligne qui ne contient qu'un nombre de 1 à 3 chiffres est considérée comme un numéro de
page et supprimée. Le contrôle de numérotation à la fin permet de s'en apercevoir si ce n'était pas le cas.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.corpus import parse_articles  # noqa: E402
from src.retrieval.textclean import strip_page_noise  # noqa: E402


def extract_clean(pdf_path: Path):
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        pages = [(page.extract_text() or "").splitlines() for page in pdf.pages]
    kept, dropped, found = strip_page_noise(pages)
    return "\n".join(kept), len(pages), dropped, found


def check_numbering(text: str) -> None:
    arts = parse_articles(text, "x", "x")
    nums = [int(a.number) for a in arts if a.number.isdigit()]
    print(f"  {len(arts)} articles reconnus", end="")
    if not nums:
        print("\n  ATTENTION : aucun article reconnu. Le texte ne contient pas de lignes du type "
              "'Article 5' ou 'Art. 5 :'. Ouvrez le .txt pour voir ce qui a été extrait.")
        return
    print(f" (de l'article {min(nums)} à l'article {max(nums)})")
    missing = sorted(set(range(1, max(nums) + 1)) - set(nums))
    dup = [a.number for a in arts if "#" in a.number]
    if missing:
        print(f"  ATTENTION : numéros absents : {missing[:30]}{' ...' if len(missing) > 30 else ''}")
        print("  -> soit l'extraction a mal lu ces articles, soit la loi les a abrogés. À vérifier dans le .txt.")
    if dup:
        print(f"  ATTENTION : numéros présents deux fois : {dup[:20]} (renvoi pris pour un titre, ou texte répété).")
    if not missing and not dup:
        print("  Numérotation complète et sans doublon.")


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("Usage : python scripts/pdf_to_txt.py entree.pdf sortie.txt")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if not src.exists():
        sys.exit(f"{src} introuvable.")
    text, n_pages, dropped, found = extract_clean(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    print(f"{src.name} : {n_pages} pages, {dropped} lignes de bruit retirées -> {dst}")
    for norm, count in found[:8]:
        print(f"  en-tête répété retiré ({count} pages) : {norm[:80]}")
    check_numbering(text)


if __name__ == "__main__":
    main()
