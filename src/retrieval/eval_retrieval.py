"""Mesure de la recherche de règle : Recall@k et MRR, BM25 contre dense contre hybride.

La vérité terrain ne coûte rien : chaque clause de test porte déjà sa catégorie CUAD, donc la
règle attendue est celle de cette catégorie (R06 pour "Cap On Liability"...). On mesure combien
de fois la bonne règle apparaît dans les k premiers résultats.

Usage :
    python -m src.retrieval.eval_retrieval                              # BM25 seul
    python -m src.retrieval.eval_retrieval --dense                      # + E5 multilingue + hybride
    python -m src.retrieval.eval_retrieval --dense --model BAAI/bge-m3 --out retrieval_bge_m3.json

Sur Colab (pas de MLflow local), on calcule avec --no-mlflow puis, sur le PC :
    python -m src.retrieval.eval_retrieval --log-only reports/retrieval_e5.json

Résultats : reports/<out>, et un run MLflow par méthode dans l'expérience "02-recherche-regles".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .index import HybridIndex
from .rules import RULES_SOURCE_ID, RuleBook, load_rules, rules_to_articles


def evaluate(index: HybridIndex, clauses: list[str], expected_ids: list[str],
             mode: str, ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    hits_at = {k: 0 for k in ks}
    rr = 0.0
    top = max(ks)
    for text, truth in zip(clauses, expected_ids):
        ranked = [h.article.article_id for h in index.search(text, k=top, mode=mode)]
        for k in ks:
            hits_at[k] += truth in ranked[:k]
        if truth in ranked:
            rr += 1.0 / (ranked.index(truth) + 1)
    n = len(clauses)
    return {**{f"recall@{k}": hits_at[k] / n for k in ks}, "mrr": rr / n, "n": n}


def build_rule_index(book: RuleBook, embedder=None) -> HybridIndex:
    index = HybridIndex(embedder)
    articles = {a.article_id: a for a in rules_to_articles(book)}
    index.refresh(articles)
    return index


def load_clauses(path: Path, book: RuleBook) -> tuple[list[str], list[str]]:
    cat_to_rule = {r["category"]: f"{RULES_SOURCE_ID}:{r['id']}" for r in book.rules}
    texts, truth = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("label") in cat_to_rule:
                texts.append(row["text"])
                truth.append(cat_to_rule[row["label"]])
    return texts, truth


def log_to_mlflow(results: dict, encoder: str | None, n_rules: int, tracking_uri: str | None) -> None:
    try:
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("02-recherche-regles")
        for mode, r in results.items():
            with mlflow.start_run(run_name=f"regles-{mode}" + (f"-{encoder.split('/')[-1]}" if encoder and mode != "bm25" else "")):
                mlflow.set_tag("etage", "recherche")
                mlflow.log_params({"mode": mode, "encodeur": (encoder or "aucun") if mode != "bm25" else "aucun",
                                   "n_regles": n_rules})
                # MLflow refuse "@" dans un nom de métrique : recall@5 -> recall_at_5
                mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in r.items() if k != "n"})
        print("Résultats enregistrés dans MLflow (expérience 02-recherche-regles).")
    except Exception as e:  # noqa: BLE001
        print(f"(MLflow non renseigné : {e})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default="data/rules/rules_contractuelles.json")
    ap.add_argument("--test", default="data/processed/clauses_test.jsonl")
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--out", default="retrieval_eval.json", help="nom du fichier écrit dans reports/")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--log-only", metavar="FICHIER", help="n'évalue rien : envoie un fichier de résultats dans MLflow")
    args = ap.parse_args()

    try:
        from src.config import MLFLOW_TRACKING_URI, REPORTS, ROOT
    except Exception:  # config absente ou illisible (ex. Colab) : dossier courant
        ROOT, REPORTS, MLFLOW_TRACKING_URI = Path.cwd(), Path.cwd() / "reports", None

    if args.log_only:
        data = json.loads(Path(args.log_only).read_text(encoding="utf-8"))
        log_to_mlflow(data["resultats"], data.get("encodeur"), data.get("n_regles", 41), MLFLOW_TRACKING_URI)
        return

    book = load_rules(Path(ROOT) / args.rules)
    texts, truth = load_clauses(Path(ROOT) / args.test, book)
    print(f"{len(texts)} clauses de test, {len(book.rules)} règles")

    embedder = None
    modes = ["bm25"]
    if args.dense:
        from .index import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(args.model)
        modes += ["dense", "hybrid"]
    index = build_rule_index(book, embedder)

    results = {m: evaluate(index, texts, truth, m) for m in modes}
    for m, r in results.items():
        print(f"  {m:7s} Recall@1 {r['recall@1']:.3f}  Recall@3 {r['recall@3']:.3f}  "
              f"Recall@5 {r['recall@5']:.3f}  MRR {r['mrr']:.3f}")

    encoder = args.model if args.dense else None
    out = Path(REPORTS) / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"encodeur": encoder, "n_regles": len(book.rules), "resultats": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Résultats écrits dans {out}")

    if not args.no_mlflow:
        log_to_mlflow(results, encoder, len(book.rules), MLFLOW_TRACKING_URI)


if __name__ == "__main__":
    main()
