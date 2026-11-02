"""Inférence : type de clause + confiance + abstention. Étape 3 de la chaîne d'audit.

Utilise le modèle Legal-BERT itération 2 (models/best_model_v2).

Dossier du modèle attendu :
    config.json, model.safetensors, tokenizer.json, tokenizer_config.json,
    special_tokens_map.json, vocab.txt, labels.json
(le fichier d'export s'appelle `model_2.safetensors` : le renommer en `model.safetensors`.)

Les noms de catégories viennent de labels.json, PAS de config.json (qui ne contient que
LABEL_0...LABEL_40). L'ordre de labels.json est l'ordre alphabétique de l'entraînement
(LabelEncoder) : ne jamais le trier ni le modifier.

    clf = ClauseClassifier("models/best_model_v2", threshold=0.5)
    clf.predict(["The liability of Supplier shall not exceed ..."])
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Prediction:
    label: str
    confidence: float
    top3: list[tuple[str, float]]
    abstain: bool           # confiance sous le seuil : la clause part en "à vérifier par un humain"
    probs: list[float] | None = None    # probabilité de CHAQUE étiquette (même ordre que labels.json)


class ClauseClassifier:
    def __init__(self, model_dir: str | Path, threshold: float = 0.5, max_length: int = 256,
                 device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = Path(model_dir)
        if not (model_dir / "model.safetensors").exists():
            hint = ""
            if (model_dir / "model_2.safetensors").exists():
                hint = (f"\nLe fichier s'appelle model_2.safetensors : renommez-le en model.safetensors "
                        f"(Rename-Item \"{model_dir / 'model_2.safetensors'}\" model.safetensors).")
            raise FileNotFoundError(f"model.safetensors introuvable dans {model_dir}.{hint}")
        self.labels: list[str] = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
        self.threshold = threshold
        self.max_length = max_length            # 256 : valeur d'entraînement (metrics.json)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            id2label=dict(enumerate(self.labels)),
            label2id={l: i for i, l in enumerate(self.labels)},
        ).to(self.device).eval()
        if self.model.config.num_labels != len(self.labels):
            raise ValueError(f"{self.model.config.num_labels} sorties dans le modèle, "
                             f"{len(self.labels)} noms dans labels.json")

    def predict(self, texts: list[str], batch_size: int = 32) -> list[Prediction]:
        torch = self._torch
        out: list[Prediction] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, truncation=True, padding=True, max_length=self.max_length,
                                 return_tensors="pt").to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu()
            top_p, top_i = probs.topk(3, dim=-1)
            for row, p, ix in zip(probs, top_p, top_i):
                top3 = [(self.labels[int(j)], float(v)) for v, j in zip(p, ix)]
                out.append(Prediction(top3[0][0], top3[0][1], top3, top3[0][1] < self.threshold, row.tolist()))
        return out
