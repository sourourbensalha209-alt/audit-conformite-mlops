"""Essai de bout en bout : découpage en clauses + Legal-BERT, mesuré sur des contrats annotés par des juristes.

    python scripts/eval_pipeline.py --model models/best_model_v2 --contracts 10

Pour chaque seuil de confiance, le tableau final donne :
  rappel %      parmi les (contrat, catégorie) annotés par les juristes, part retrouvée
  fantômes %    parmi les paragraphes SANS annotation, part qui reçoit quand même une catégorie
  abstention %  part des segments laissés "à vérifier par un humain" faute de confiance
  Autre %       part des segments classés « Autre » (modèle de l'itération 3 : aucune catégorie annoncée)

C'est cette table qui permet de choisir le seuil de confiance : on cherche le seuil où les fantômes
baissent nettement sans que le rappel s'effondre. Legal-BERT n'a pas de classe "autre" : sans seuil, il
annonce une catégorie sur n'importe quel paragraphe.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.segment import segment  # noqa: E402
from src.data.segment_eval import load_docs, score_pipeline  # noqa: E402


def pick(docs: list[dict], n: int, max_chars: int) -> list[dict]:
    """n contrats répartis des plus courts aux plus longs, en écartant les très gros (lents à classer)."""
    docs = sorted((d for d in docs if len(d["text"]) <= max_chars), key=lambda d: len(d["text"]))
    if n >= len(docs):
        return docs
    step = (len(docs) - 1) / max(n - 1, 1)
    return [docs[round(i * step)] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/best_model_v2")
    ap.add_argument("--test", default="data/raw/cuad/test.jsonl")
    ap.add_argument("--contracts", type=int, default=10)
    ap.add_argument("--max-chars", type=int, default=120000)
    ap.add_argument("--mlflow", action="store_true", help="enregistre le tableau dans MLflow (expérience 03-bout-en-bout)")
    args = ap.parse_args()

    from src.models.predict import ClauseClassifier

    model_dir = Path(args.model)
    clf = ClauseClassifier(model_dir if model_dir.is_absolute() else ROOT / model_dir, threshold=0.0)
    docs = pick(load_docs(ROOT / args.test), args.contracts, args.max_chars)
    print(f"{len(docs)} contrats, périphérique : {clf.device}\n")

    segs_by_doc, preds_by_doc = [], []
    t0 = time.time()
    for i, d in enumerate(docs, 1):
        segs = segment(d["text"])
        preds = [(p.label, p.confidence) for p in clf.predict([s.text for s in segs])]
        segs_by_doc.append(segs)
        preds_by_doc.append(preds)
        print(f"  [{i}/{len(docs)}] {d['title'][:50]:50s} {len(segs):4d} segments   ({time.time() - t0:.0f} s)")

    print(f"\n{sum(len(s) for s in segs_by_doc)} segments classés\n")
    print(f"{'seuil':>6} | {'rappel %':>9} | {'fantômes %':>10} | {'abstention %':>12} | {'Autre %':>8}")
    for r in score_pipeline(docs, segs_by_doc, preds_by_doc):
        print(f"{r['seuil']:>6} | {r['rappel %']:>9} | {r['fantômes %']:>10} | {r['abstention %']:>12} | {r['Autre %']:>8}")
    print(f"\n(rappel calculé sur {r['contrat-catégories annotées']} couples contrat-catégorie annotés)")

    if args.mlflow:
        try:
            import mlflow

            try:
                from src.config import MLFLOW_TRACKING_URI
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            except Exception:  # noqa: BLE001
                pass
            mlflow.set_experiment("03-bout-en-bout")
            with mlflow.start_run(run_name=f"bout-en-bout-{model_dir.name}"):
                mlflow.set_tag("etage", "classification+decoupage")
                mlflow.log_params({"modele": model_dir.name, "contrats": len(docs), "segments": sum(len(x) for x in segs_by_doc)})
                for r in score_pipeline(docs, segs_by_doc, preds_by_doc):
                    k = f"s{int(round(r['seuil'] * 100)):02d}"
                    mlflow.log_metrics({f"rappel_{k}": r["rappel %"], f"fantomes_{k}": r["fantômes %"],
                                        f"abstention_{k}": r["abstention %"], f"autre_{k}": r["Autre %"]})
            print("Tableau enregistré dans MLflow (expérience 03-bout-en-bout).")
        except Exception as e:  # noqa: BLE001
            print(f"(MLflow non renseigné : {e})")


if __name__ == "__main__":
    main()
