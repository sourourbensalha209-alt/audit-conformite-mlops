# Agent d'audit de conformité documentaire

Service automatisé qui analyse un contrat ou une politique de confidentialité et produit
un rapport d'audit clause par clause, avec verdict, justification et citation de
l'article réglementaire applicable.

**Module Projet — 5ème année Ingénierie Data Science & IA**

---

## Membres et rôles

| Membre | Rôle | Périmètre |
| --- | --- | --- |
| À compléter | Data Engineer | Ingestion, OCR, découpage, Spark, DVC |
| À compléter | ML Engineer | Classification, retrieval, jugement, MLflow |
| À compléter | MLOps Engineer | Kubeflow, Docker, FastAPI, monitoring |

---

## Architecture

```
Document PDF/DOCX
      |
      v
[1] OCR + detection de mise en page
      |
      v
[2] Decoupage en clauses
      |
      v
[3] Classification du type de clause
      |
      v
[4] Retrieval hybride dans le corpus reglementaire
      |
      v
[5] Jugement de conformite (SLM)
      |
      v
[6] Scoring + rapport
      |
      v
[7] API FastAPI + monitoring
      |
      v
   Feedback juriste --> reentrainement
```

---

## Démarrage rapide

```bash
# 1. Environnement
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Données
python -m src.data.download          # télécharge CUAD + le corpus RGPD
dvc add data/raw                     # versionne le corpus brut
git add data/raw.dvc .gitignore && git commit -m "data(raw): corpus v1"

# 3. Pipeline
dvc repro                            # exécute toutes les étapes déclarées

# 4. Suivi des expériences
mlflow ui --port 5000                # http://localhost:5000

# 5. API
uvicorn src.api.main:app --reload    # http://localhost:8000/docs
```

---

## Structure du dépôt

```
.
├── data/                  # données (non versionnées par git, gérées par DVC)
│   ├── raw/               # v1 : corpus brut
│   ├── interim/           # v2 : texte extrait et nettoyé
│   └── processed/         # v3 : clauses découpées et augmentées
├── src/
│   ├── data/              # download, extraction OCR, préparation
│   ├── models/            # classification de clauses
│   ├── retrieval/         # BM25, dense, hybride
│   ├── judge/             # jugement de conformité
│   └── api/               # service FastAPI
├── tests/
├── notebooks/
├── docs/                  # fiche projet, diagrammes, rapports d'évaluation
├── dvc.yaml               # pipeline de données reproductible
├── params.yaml            # hyperparamètres centralisés
└── docker-compose.yml     # API + serveur MLflow
```

---

## État d'avancement

- [ ] Éval 1 — Idée de projet (10 %)
- [ ] Éval 2 — Données + classification (20 %)
- [ ] Éval 3 — RAG + jugement + API (20 %)
- [ ] Éval 4 — Orchestration + monitoring + soutenance (50 %)
