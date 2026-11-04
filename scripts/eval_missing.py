"""Mesure du verdict « manquante » sur les 30 contrats de confirmation (jamais utilisés pour le diagnostic).

    python scripts/eval_missing.py --model models/best_model_v4

Réutilise les probabilités déjà calculées par scripts/eval_pipeline.py (fichier reports/_cache_pipeline_*.json) : si elles
n'existent pas, lancer d'abord :  python scripts/eval_pipeline.py --model models/best_model_v4 --contracts 30 --fresh

Réglage et critère fixés d'avance (docs/criteres_phase5.md) : confiance >= 0,5 ; rappel de « manquante » >= 60 % ET
précision >= 50 %. Pour situer la précision : en disant toujours « manquante », elle serait égale au taux de base.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.segment import segment  # noqa: E402
from src.data.segment_eval import load_docs, pick_contracts  # noqa: E402
from src.judge.missing import CRITERIA, TAU, expected_categories, present_categories, score_missing  # noqa: E402
from src.retrieval.rules import load_rules  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/best_model_v4")
    ap.add_argument("--test", default="data/raw/cuad/test.jsonl")
    ap.add_argument("--rules", default="data/rules/rules_contractuelles.json")
    ap.add_argument("--contracts", type=int, default=30)
    ap.add_argument("--max-chars", type=int, default=120000)
    args = ap.parse_args()

    model_dir = Path(args.model)
    corpus = load_docs(ROOT / args.test)
    seen = {d["title"] for d in pick_contracts(corpus, 8, args.max_chars)}
    docs = pick_contracts(corpus, args.contracts, args.max_chars, seen)
    segs = [segment(d["text"]) for d in docs]

    cache = ROOT / "reports" / f"_cache_pipeline_{model_dir.name}_{len(docs)}_fresh.json"
    if not cache.exists():
        sys.exit(f"{cache.name} introuvable. Lancer d'abord :\n  python scripts/eval_pipeline.py --model {args.model} "
                 f"--contracts {args.contracts} --fresh")
    saved = json.loads(cache.read_text(encoding="utf-8"))
    sig = saved["signature"]
    if sig["titres"] != [d["title"] for d in docs] or sig["segments"] != [len(s) for s in segs]:
        sys.exit("Le fichier de probabilités ne correspond pas à ces contrats : relancer eval_pipeline.py avec --no-cache.")
    labels = sig["labels"]

    book = load_rules(ROOT / args.rules)
    expected = expected_categories(book)
    present = [present_categories(d["text"], probs, labels, TAU) for d, probs in zip(docs, saved["probs"])]
    res = score_missing(docs, present, expected)

    print(f"{len(docs)} contrats | {len(expected)} catégories dont l'absence est à signaler | confiance >= {TAU}\n")
    print(f"{'catégorie':20s} | {'absentes en vrai':>16} | {'dites manquantes':>16} | {'bien vues':>9} | {'à tort':>6} | {'ratées':>6}")
    for r in res["par_categorie"]:
        print(f"{r['catégorie']:20s} | {r['absentes_en_vrai']:>16} | {r['dit_manquante']:>16} | {r['tp']:>9} | {r['fp']:>6} | {r['fn']:>6}")
    print(f"\nTaux de base : {res['taux_de_base_pct']} % des {res['couples']} couples contrat-catégorie sont réellement absents.")
    print(f"Précision de « manquante » : {res['precision_manquante_pct']} %   (si on disait toujours « manquante » : {res['taux_de_base_pct']} %)")
    print(f"Rappel de « manquante »    : {res['rappel_manquante_pct']} %")
    print(f"\nCritère fixé d'avance : rappel >= {CRITERIA['recall_min']:.0f} % ET précision >= {CRITERIA['precision_min']:.0f} %  ->  "
          f"{'CRITÈRE ATTEINT' if res['critere_atteint'] else 'CRITÈRE NON ATTEINT'}")

    out = ROOT / "reports" / f"manquantes_{model_dir.name}_{len(docs)}contrats_frais.json"
    out.write_text(json.dumps({"modele": model_dir.name, "contrats": len(docs), "tau": TAU, "criteres": CRITERIA, **res},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Résultats enregistrés : {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
