"""Modele de reference : TF-IDF + SVM lineaire, trace dans MLflow.

C'est la baseline obligatoire contre laquelle tous les autres modeles
de classification seront compares.
"""

import json

import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import DATA_PROCESSED, REPORTS, MLFLOW_TRACKING_URI, load_params


def load_split(name: str) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(DATA_PROCESSED / f"clauses_{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("label"):
                texts.append(row["text"])
                labels.append(row["label"])
    return texts, labels


def main() -> None:
    p = load_params("classify_baseline")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("01-classification-clauses")

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    if not X_train:
        raise SystemExit(
            "Aucune clause etiquetee. Renseigne le champ 'label' depuis les "
            "annotations CUAD dans src/data/prepare.py avant d'entrainer."
        )

    with mlflow.start_run(run_name="tfidf-svm-baseline"):
        mlflow.log_params(p)
        mlflow.set_tag("etage", "classification")
        mlflow.set_tag("type", "baseline")

        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, p["ngram_max"]),
                max_features=p["max_features"],
            )),
            ("clf", LinearSVC(C=p["C"])),
        ])
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        report_path = REPORTS / "metrics_baseline.json"
        report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(report_path))

        print(json.dumps(metrics, indent=2))
        print("\n" + classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
