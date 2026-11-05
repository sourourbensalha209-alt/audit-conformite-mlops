"""Produit le rapport d'audit d'un contrat (page HTML autonome).

    python scripts/audit_contract.py --list                       # contrats déjà analysés (démonstration instantanée)
    python scripts/audit_contract.py --cuad TUNIU                 # un contrat CUAD choisi par un morceau de son titre
    python scripts/audit_contract.py --txt mon_contrat.txt        # un contrat en texte (paragraphes séparés par une ligne vide)

Pour les 30 contrats de confirmation, les probabilités du classifieur sont relues dans reports/_cache_pipeline_*_fresh.json
(créé par scripts/eval_pipeline.py) : aucun modèle à charger, le rapport sort en une seconde. Sinon, le classifieur tourne
(compter 0,2 seconde par paragraphe sans carte graphique).

La lecture d'un PDF n'est pas encore branchée : le texte extrait d'un PDF n'a pas de lignes vides entre les paragraphes.
"""

import argparse
import json
import re
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, ROOT.as_posix())

from src.data.segment import segment  # noqa: E402
from src.judge.report import analyse, build_meta, render_html  # noqa: E402
from src.retrieval.rules import load_rules  # noqa: E402


def load_cache(model_name: str) -> dict[str, list]:
    """title -> probabilités par segment, pour tous les contrats déjà classés par eval_pipeline.py."""
    out = {}
    for f in sorted((ROOT / "reports").glob(f"_cache_pipeline_{model_name}_*_fresh.json")):
        saved = json.loads(f.read_text(encoding="utf-8"))
        for title, n, probs in zip(saved["signature"]["titres"], saved["signature"]["segments"], saved["probs"]):
            out[title] = (n, probs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="liste les contrats déjà analysés")
    ap.add_argument("--cuad", help="morceau du titre d'un contrat du jeu de test CUAD")
    ap.add_argument("--txt", help="fichier texte d'un contrat")
    ap.add_argument("--model", default="models/best_model_v4")
    ap.add_argument("--test", default="data/raw/cuad/test.jsonl")
    ap.add_argument("--rules", default="data/rules/rules_contractuelles.json")
    ap.add_argument("--out", help="fichier HTML de sortie")
    ap.add_argument("--no-open", action="store_true", help="ne pas ouvrir le rapport dans le navigateur")
    args = ap.parse_args()

    model_dir = Path(args.model)
    cache = load_cache(model_dir.name)
    if args.list or not (args.cuad or args.txt):
        print(f"{len(cache)} contrats déjà analysés (rapport instantané) :")
        for t, (n, _) in cache.items():
            print(f"  {t[:75]:75s} {n:>4} paragraphes")
        if not cache:
            print("  (aucun : lancer d'abord scripts/eval_pipeline.py --model ... --contracts 30 --fresh)")
        print("\nExemple :  python scripts/audit_contract.py --cuad " + (next(iter(cache))[:12] if cache else "NOM"))
        return

    if args.txt:
        text, title = Path(args.txt).read_text(encoding="utf-8"), Path(args.txt).stem
    else:
        from src.data.segment_eval import load_docs

        docs = load_docs(ROOT / args.test)
        hits = [d for d in docs if args.cuad.lower() in d["title"].lower()]
        if not hits:
            sys.exit(f"Aucun contrat dont le titre contient « {args.cuad} ». Essayer --list.")
        if len(hits) > 1:
            print(f"{len(hits)} contrats correspondent, précisez :")
            for d in hits[:15]:
                print("  -", d["title"])
            sys.exit(1)
        text, title = hits[0]["text"], hits[0]["title"]

    labels = json.loads((ROOT / model_dir / "labels.json").read_text(encoding="utf-8"))
    n_seg = len(segment(text))
    if title in cache and cache[title][0] == n_seg:
        probs = cache[title][1]
        print(f"Probabilités déjà calculées : rapport instantané ({n_seg} paragraphes).")
    else:
        from src.models.predict import ClauseClassifier

        print(f"Classification de {n_seg} paragraphes (environ {n_seg * 0.2 / 60:.1f} minute sans carte graphique)...")
        clf = ClauseClassifier(ROOT / model_dir, threshold=0.0)
        probs = [p.probs for p in clf.predict([s.text for s in segment(text)])]

    book = load_rules(ROOT / args.rules)
    result = analyse(text, labels, probs, book)
    try:
        from src.retrieval.corpus import CorpusStore

        version = CorpusStore(ROOT / "data" / "corpus").version()
    except Exception:  # noqa: BLE001
        version = "inconnue"
    html_text = render_html(result, build_meta(title, model_dir.name, version))
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title)[:50].strip("_")
    out = ROOT / (args.out or f"reports/audit_{slug}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")

    st = {k: sum(1 for a in result["attendues"] if a["statut"] == k) for k in ("presente", "manquante", "a_verifier")}
    print(f"\nContrat : {title}")
    print(f"  {result['stats']['clauses']} clauses retrouvées dans {result['stats']['segments']} paragraphes, {result['stats']['categories']} catégories")
    print(f"  clauses attendues : {st['presente']} présentes, {st['manquante']} manquantes, {st['a_verifier']} à vérifier")
    print(f"Rapport écrit : {out.relative_to(ROOT)}")
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
