"""Enregistre dans le MLflow local les resultats d'un benchmark entraine sur Colab.

Colab ne voit pas ton MLflow local. Ce script relit le fichier de resultats
telecharge depuis Colab et cree un run MLflow par modele, dans la meme
experience que la baseline.

Usage :
    python -m src.models.log_benchmark                              # iteration 1
    python -m src.models.log_benchmark reports/benchmark_v2.json    # iteration 2

Les fichiers images et CSV presents dans reports/ et portant le meme prefixe
que le fichier de resultats sont joints au meilleur run.
"""

import json
import sys
from pathlib import Path

import mlflow

from src.config import ROOT, REPORTS, MLFLOW_TRACKING_URI

METRIC_KEYS = {"f1_macro", "f1_weighted", "accuracy", "f1_macro_rare",
               "train_minutes", "latency_ms_per_clause", "params_millions",
               "best_epoch", "epochs_run"}
SKIP_KEYS = {"key", "label"}


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else REPORTS / "benchmark_transformers.json"
    if not src.is_absolute():
        src = ROOT / src
    if not src.exists():
        sys.exit(f"{src} introuvable. Place-y le fichier telecharge depuis Colab.")

    results = json.loads(src.read_text(encoding="utf-8"))
    best = max(results, key=lambda r: r["f1_macro"])
    artifacts = [p for p in REPORTS.glob(f"{src.stem}*") if p.suffix in {".png", ".csv"}]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("01-classification-clauses")

    for r in results:
        metrics = {k: v for k, v in r.items() if k in METRIC_KEYS}
        params = {k: v for k, v in r.items() if k not in METRIC_KEYS | SKIP_KEYS}
        with mlflow.start_run(run_name=f"{r['key']}-colab"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("etage", "classification")
            mlflow.set_tag("type", "transformer")
            mlflow.set_tag("entraine_sur", "colab")
            mlflow.set_tag("iteration", str(r.get("iteration", 1)))
            if r is best:
                mlflow.set_tag("meilleur_modele", "oui")
                for path in artifacts:
                    mlflow.log_artifact(str(path))
        print(f"  {r['label']:20s} F1 macro {r['f1_macro']:.3f}  -> run MLflow cree")

    print(f"\nMeilleur modele de ce fichier : {best['label']} ({best['f1_macro']:.3f})")
    print("Ouvre MLflow (mlflow ui --port 5000) pour comparer avec les runs precedents.")


if __name__ == "__main__":
    main()
