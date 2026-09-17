"""Modele 2 : DistilBERT fine-tune sur la classification de clauses.

Il doit battre la baseline TF-IDF + SVM (F1 macro 0.670).
Tout est trace dans MLflow, dans la meme experience que la baseline,
pour que la comparaison soit directe.

Usage :
    python -m src.models.classify_bert

Duree indicative :
    GPU (Colab T4)  : 5 a 10 minutes
    CPU (portable)  : 45 a 90 minutes

Pour un essai rapide avant de lancer l'entrainement complet :
    python -m src.models.classify_bert --smoke
"""

import argparse
import json
import sys

import mlflow
import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.config import DATA_PROCESSED, REPORTS, MLFLOW_TRACKING_URI, load_params


class ClauseDataset(Dataset):
    """Enveloppe minimale autour des encodages du tokenizer."""

    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_split(name: str) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    path = DATA_PROCESSED / f"clauses_{name}.jsonl"
    if not path.exists():
        sys.exit(f"{path} introuvable. Lance d'abord : python -m src.data.prepare")
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("label"):
                texts.append(row["text"])
                labels.append(row["label"])
    return texts, labels


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="essai rapide sur 300 exemples et 1 epoque")
    args = parser.parse_args()

    p = load_params("classify_bert")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Peripherique detecte : {device}")
    if device == "cpu" and not args.smoke:
        print("Pas de GPU. L'entrainement complet prendra 45 a 90 minutes.")
        print("Pour verifier que tout marche d'abord : --smoke\n")

    X_train, y_train_raw = load_split("train")
    X_test, y_test_raw = load_split("test")

    if args.smoke:
        X_train, y_train_raw = X_train[:300], y_train_raw[:300]
        X_test, y_test_raw = X_test[:100], y_test_raw[:100]
        p = {**p, "epochs": 1}

    # Les etiquettes de test absentes du train sont ecartees :
    # un modele ne peut pas predire une classe qu'il n'a jamais vue.
    encoder = LabelEncoder().fit(y_train_raw)
    known = set(encoder.classes_)
    keep = [i for i, lbl in enumerate(y_test_raw) if lbl in known]
    X_test = [X_test[i] for i in keep]
    y_test_raw = [y_test_raw[i] for i in keep]

    y_train = encoder.transform(y_train_raw).tolist()
    y_test = encoder.transform(y_test_raw).tolist()
    n_labels = len(encoder.classes_)

    print(f"train {len(X_train):,} | test {len(X_test):,} | {n_labels} categories\n")

    tokenizer = AutoTokenizer.from_pretrained(p["model_name"])
    enc_train = tokenizer(X_train, truncation=True, padding="max_length",
                          max_length=p["max_length"])
    enc_test = tokenizer(X_test, truncation=True, padding="max_length",
                         max_length=p["max_length"])

    model = AutoModelForSequenceClassification.from_pretrained(
        p["model_name"], num_labels=n_labels
    )

    training_args = TrainingArguments(
        output_dir=str(REPORTS / "bert_checkpoints"),
        num_train_epochs=p["epochs"],
        per_device_train_batch_size=p["batch_size"],
        per_device_eval_batch_size=p["batch_size"] * 2,
        learning_rate=float(p["learning_rate"]),
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to=[],          # on logge nous-memes dans MLflow
        disable_tqdm=False,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ClauseDataset(enc_train, y_train),
        eval_dataset=ClauseDataset(enc_test, y_test),
        compute_metrics=compute_metrics,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("01-classification-clauses")

    run_name = "distilbert-smoke" if args.smoke else "distilbert-finetune"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**p, "device": device, "n_labels": n_labels})
        mlflow.set_tag("etage", "classification")
        mlflow.set_tag("type", "transformer")

        trainer.train()

        eval_metrics = trainer.evaluate()
        metrics = {
            "f1_macro": eval_metrics["eval_f1_macro"],
            "f1_weighted": eval_metrics["eval_f1_weighted"],
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        mlflow.log_metrics(metrics)

        preds = np.argmax(trainer.predict(trainer.eval_dataset).predictions, axis=-1)
        report_txt = classification_report(
            y_test, preds, labels=list(range(n_labels)), target_names=encoder.classes_, zero_division=0
        )

        if not args.smoke:
            out = REPORTS / "metrics_bert.json"
            out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(out))

            model_dir = REPORTS / "bert_model"
            trainer.save_model(str(model_dir))
            tokenizer.save_pretrained(str(model_dir))

        print(json.dumps(metrics, indent=2))
        print("\n" + report_txt)

    baseline = REPORTS / "metrics_baseline.json"
    if baseline.exists() and not args.smoke:
        ref = json.loads(baseline.read_text(encoding="utf-8"))["f1_macro"]
        delta = metrics["f1_macro"] - ref
        signe = "+" if delta >= 0 else ""
        print(f"\nComparaison : baseline {ref:.3f} -> DistilBERT "
              f"{metrics['f1_macro']:.3f} ({signe}{delta:.3f})")


if __name__ == "__main__":
    main()
