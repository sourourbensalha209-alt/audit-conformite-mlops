"""Construit l'interface de démonstration : une page HTML autonome où l'on choisit un contrat dans une liste.

    python scripts/build_interface.py

Utilise les probabilités déjà calculées pour les contrats de confirmation (reports/_cache_pipeline_*_fresh.json, créé par
scripts/eval_pipeline.py) : aucun modèle à charger, aucun serveur, aucun accès internet. Écrit reports/interface_audit.html.
"""

import argparse
import json
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, ROOT.as_posix())

from src.data.segment import segment  # noqa: E402
from src.data.segment_eval import load_docs  # noqa: E402
from src.judge.report import analyse, build_meta, render_interface, statut_counts  # noqa: E402
from src.retrieval.rules import load_rules  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/best_model_v4")
    ap.add_argument("--test", default="data/raw/cuad/test.jsonl")
    ap.add_argument("--rules", default="data/rules/rules_contractuelles.json")
    ap.add_argument("--out", default="reports/interface_audit.html")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model)
    labels = json.loads((ROOT / model_dir / "labels.json").read_text(encoding="utf-8"))
    cache = {}
    for f in sorted((ROOT / "reports").glob(f"_cache_pipeline_{model_dir.name}_*_fresh.json")):
        saved = json.loads(f.read_text(encoding="utf-8"))
        cache.update({t: (n, p) for t, n, p in zip(saved["signature"]["titres"], saved["signature"]["segments"], saved["probs"])})
    if not cache:
        sys.exit("Aucune probabilité enregistrée. Lancer d'abord : python scripts/eval_pipeline.py --model "
                 f"{args.model} --contracts 30 --fresh")

    docs = {d["title"]: d for d in load_docs(ROOT / args.test)}
    book = load_rules(ROOT / args.rules)
    items = []
    for title, (n, probs) in cache.items():
        doc = docs.get(title)
        if doc is None or len(segment(doc["text"])) != n:
            continue
        items.append({"titre": title, "analyse": analyse(doc["text"], labels, probs, book)})
    items.sort(key=lambda it: -it["analyse"]["stats"]["segments"])       # les plus riches d'abord : plus parlants pour une démonstration
    try:
        from src.retrieval.corpus import CorpusStore

        version = CorpusStore(ROOT / "data" / "corpus").version()
    except Exception:  # noqa: BLE001
        version = "inconnue"

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_interface(items, build_meta("", model_dir.name, version)), encoding="utf-8")
    tot = {k: sum(statut_counts(it["analyse"])[k] for it in items) for k in ("presente", "manquante", "a_verifier")}
    print(f"{len(items)} contrats dans l'interface | clauses attendues : {tot['presente']} présentes, {tot['manquante']} manquantes, {tot['a_verifier']} à vérifier")
    print(f"Interface écrite : {out.relative_to(ROOT)}")
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
