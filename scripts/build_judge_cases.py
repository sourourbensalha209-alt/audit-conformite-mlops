"""Fabrique les cas d'évaluation du juge et la feuille Excel d'annotation.

    python scripts/build_judge_cases.py

Écrit :
  data/judge/cases.jsonl       tous les cas (clause, règle, sources, verdict attendu pour les paires sans rapport)
  data/judge/a_remplir.xlsx    la feuille que Sourour et Ghayth remplissent (verdict de référence)

Les clauses viennent des 64 contrats de test qui n'ont servi à AUCUNE mesure précédente (ni les 8 du diagnostic, ni les 30
de confirmation). Exception : les clauses sur les données personnelles, rares, sont aussi cherchées dans les contrats
d'entraînement (le juge n'est pas entraîné sur CUAD). Le tirage est reproductible (graine fixe).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.segment_eval import load_docs, pick_contracts  # noqa: E402
from src.judge.cases import build_cases  # noqa: E402
from src.judge.sheet import write_sheet  # noqa: E402
from src.retrieval.corpus import CorpusStore  # noqa: E402
from src.retrieval.rules import load_rules  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/raw/cuad/test.jsonl")
    ap.add_argument("--rules", default="data/rules/rules_contractuelles.json")
    ap.add_argument("--corpus-dir", default="data/corpus")
    ap.add_argument("--out", default="data/judge")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    corpus = load_docs(ROOT / args.test)
    used = {d["title"] for d in pick_contracts(corpus, 8)}
    used |= {d["title"] for d in pick_contracts(corpus, 30, exclude_titles=used)}
    docs = [d for d in corpus if d["title"] not in used]
    print(f"{len(corpus)} contrats de test, {len(used)} déjà utilisés pour des mesures, {len(docs)} disponibles pour le juge")

    articles = CorpusStore(ROOT / args.corpus_dir).load()
    n_rgpd = sum(a.article_id.startswith("rgpd:") for a in articles.values())
    if not n_rgpd:
        print("ATTENTION : aucun article du RGPD dans le corpus (lancer d'abord : python -m src.retrieval.law_watch). "
              "La famille « rgpd » sera vide.")

    train = ROOT / Path(args.test).parent / "train.jsonl"
    rgpd_docs = corpus + (load_docs(train) if train.exists() else [])   # les clauses sur les données personnelles sont rares
    cases, stats = build_cases(docs, load_rules(ROOT / args.rules), articles, rgpd_docs=rgpd_docs, seed=args.seed)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cases.jsonl", "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    who = write_sheet(out / "a_remplir.xlsx", cases)

    print(f"  famille « données personnelles » : {stats['rgpd']} clauses, {stats['rgpd_contrats']} contrats différents")
    print(f"\n{len(cases)} cas écrits dans {out.relative_to(ROOT)}/cases.jsonl")
    print("  par famille :", dict(Counter(c["family"] for c in cases)))
    print(f"  à annoter par l'équipe : {sum(c['a_annoter'] for c in cases)} cas "
          f"({dict(Counter(who.values()))}) -> {out.relative_to(ROOT)}/a_remplir.xlsx")
    print("  catégories couvertes (clauses réelles) :", len({c['category'] for c in cases if c['family'] == 'playbook'}))
    for fam in ("playbook", "rgpd", "mismatch", "neutral"):
        ex = next((c for c in cases if c["family"] == fam), None)
        if ex:
            print(f"\n  exemple [{fam}] règle {ex['rule']['id']} : « {ex['clause'][:110]}... »")


if __name__ == "__main__":
    main()
