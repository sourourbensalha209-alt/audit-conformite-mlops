"""Orchestration du réentraînement : des corrections des juristes à un nouveau jeu d'entraînement.

    python -m src.orchestration.retrain            # collecte + décision (+ paquet Colab si réentraînement décidé)
    python -m src.orchestration.retrain --force    # prépare le paquet même sans déclencheur

La boucle complète :
  1. COLLECTE   les corrections qui portent un passage du contrat ET sa bonne catégorie deviennent des exemples
                étiquetés (data/feedback/reentrainement/exemples_juristes.jsonl) ; doublons et catégories inconnues écartés ;
  2. DÉCISION   on réentraîne si assez de NOUVEAUX exemples (20 par défaut), ou si le monitoring signale une dérive
                ou trop de corrections ; chaque décision est tracée (decision.json, historique, MLflow si disponible) ;
  3. PAQUET     reentrainement_colab.zip : exemples + étiquettes + mode d'emploi, à ajouter au jeu d'entraînement
                dans le notebook de l'itération 4 (GPU Colab) ;
  4. PROMOTION  le modèle obtenu n'est mis en service que s'il bat l'actuel (src/orchestration/promote.py).

Planification : à lancer chaque nuit (Planificateur de tâches Windows, cron) ou depuis la page Monitoring.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from src import monitoring
from src.orchestration.registry import production_model_dir

ROOT = Path(__file__).resolve().parents[2]
FEEDBACK_FILE = ROOT / "data" / "feedback" / "corrections.jsonl"
WORK_DIR = ROOT / "data" / "feedback" / "reentrainement"
SEUIL_EXEMPLES = 20        # nouveaux exemples nécessaires pour justifier un réentraînement
MIN_EXEMPLES_ALERTE = 5    # sur alerte du monitoring, il faut quand même quelques exemples à apprendre
MIN_CHARS = 25             # même minimum qu'un segment (src/data/segment.py)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _labels(model_dir: Path | None = None) -> list[str] | None:
    path = (model_dir or production_model_dir()) / "labels.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def collect(feedback_path: Path | None = None, labels: list[str] | None = None) -> tuple[list[dict], dict]:
    """Corrections -> exemples étiquetés (texte, label). Retourne aussi ce qui a été écarté, et pourquoi."""
    ecartes = {"sans_passage": 0, "categorie_inconnue": 0, "doublon": 0}
    vus, exemples = set(), []
    for c in _read_jsonl(feedback_path or FEEDBACK_FILE):
        texte = " ".join(str(c.get("texte") or "").split())
        label = c.get("categorie_corrigee")
        if not label or len(texte) < MIN_CHARS:
            ecartes["sans_passage"] += 1
            continue
        if labels is not None and label not in labels:
            ecartes["categorie_inconnue"] += 1
            continue
        cle = (texte.lower(), label)
        if cle in vus:
            ecartes["doublon"] += 1
            continue
        vus.add(cle)
        exemples.append({"text": texte, "label": label, "source": "juriste",
                         "job_id": c.get("job_id"), "date": c.get("date")})
    return exemples, ecartes


def decide(n_nouveaux: int, alertes: list[dict], seuil: int = SEUIL_EXEMPLES, force: bool = False) -> dict:
    raisons = []
    if n_nouveaux >= seuil:
        raisons.append(f"{n_nouveaux} nouveaux exemples de juristes (seuil : {seuil}).")
    for a in alertes:
        if a.get("niveau") in ("derive", "qualite"):
            raisons.append(f"Monitoring : {a['message']}")
    if force:
        raisons.append("Lancement forcé.")
    if raisons and not force and n_nouveaux < min(seuil, MIN_EXEMPLES_ALERTE):
        # une alerte sans assez d'exemples corrigés : on la signale, mais il n'y a pas de quoi apprendre
        return {"reentrainer": False, "raisons": raisons,
                "conseil": f"Faire corriger quelques audits récents par un juriste : au moins {MIN_EXEMPLES_ALERTE} "
                           f"exemples sont nécessaires (actuellement {n_nouveaux})."}
    return {"reentrainer": bool(raisons), "raisons": raisons or ["Aucun déclencheur : le modèle actuel est conservé."]}


INSTRUCTIONS = """# Réentraînement avec les corrections des juristes

Ce paquet contient :
- `exemples_juristes.jsonl` : {n} passages de contrats corrigés par un juriste (`text`, `label`) ;
- `labels.json` : les étiquettes du modèle actuel ({modele}), dans le même ordre ;
- `decision.json` : pourquoi le réentraînement a été décidé.

## Dans le notebook `notebooks/iteration4_colab.ipynb` (GPU T4)

Après la cellule qui construit `X_train`, `y_train` (et avant l'entraînement), ajouter :

```python
import json
ex = [json.loads(l) for l in open("exemples_juristes.jsonl", encoding="utf-8")]
ex = [e for e in ex if e["label"] in set(encoder.classes_)]
REPETITION = 3                      # un exemple corrigé par un juriste compte plus qu'un exemple CUAD
X_train = list(X_train) + [e["text"] for e in ex] * REPETITION
y_train = list(y_train) + list(encoder.transform([e["label"] for e in ex])) * REPETITION
print(len(ex), "exemples de juristes ajoutés")
```

Le jeu de TEST ne change pas : c'est la condition pour comparer honnêtement le nouveau modèle à l'ancien.

## Ensuite, sur le PC

1. Dézipper le modèle obtenu dans `models/best_model_v5/` (avec `metrics.json` et `labels.json`).
2. `python -m src.orchestration.promote models/best_model_v5`
   Le modèle n'est mis en service que s'il bat le modèle actuel (F1 macro, et pas de recul sur les catégories rares).
3. Si la promotion est acceptée : `dvc add models/best_model_v5`, puis redémarrer le service.
"""


def run(force: bool = False, seuil: int = SEUIL_EXEMPLES, feedback_path: Path | None = None,
        work_dir: Path | None = None, log_path: Path | None = None) -> dict:
    """Une exécution complète de l'orchestrateur. Retourne la décision (aussi écrite dans work_dir)."""
    work = work_dir or WORK_DIR
    work.mkdir(parents=True, exist_ok=True)
    model_dir = production_model_dir()
    labels = _labels(model_dir)

    exemples, ecartes = collect(feedback_path, labels)
    deja = {(e["text"].lower(), e["label"]) for e in _read_jsonl(work / "exemples_utilises.jsonl")}
    nouveaux = [e for e in exemples if (e["text"].lower(), e["label"]) not in deja]
    alertes = monitoring.summary(log_path, feedback_path)["alertes"]
    decision = decide(len(nouveaux), alertes, seuil, force)

    etat = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modele_en_production": model_dir.name,
        "exemples_total": len(exemples),
        "exemples_nouveaux": len(nouveaux),
        "seuil": seuil,
        "ecartes": ecartes,
        **decision,
        "paquet": None,
    }
    with (work / "exemples_juristes.jsonl").open("w", encoding="utf-8") as f:
        for e in exemples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    if decision["reentrainer"]:
        paquet = work / "reentrainement_colab.zip"
        with zipfile.ZipFile(paquet, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(work / "exemples_juristes.jsonl", "exemples_juristes.jsonl")
            if labels is not None:
                z.writestr("labels.json", json.dumps(labels, indent=2, ensure_ascii=False))
            z.writestr("decision.json", json.dumps(etat, indent=2, ensure_ascii=False))
            z.writestr("LISEZ-MOI.md", INSTRUCTIONS.format(n=len(exemples), modele=model_dir.name))
        etat["paquet"] = paquet.name
        with (work / "exemples_utilises.jsonl").open("a", encoding="utf-8") as f:
            for e in nouveaux:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    (work / "decision.json").write_text(json.dumps(etat, indent=2, ensure_ascii=False), encoding="utf-8")
    with (work / "historique.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(etat, ensure_ascii=False) + "\n")
    _log_mlflow(etat, work / "decision.json")
    return etat


def state(work_dir: Path | None = None) -> dict | None:
    path = (work_dir or WORK_DIR) / "decision.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _log_mlflow(etat: dict, artifact: Path) -> None:
    """Trace la décision dans MLflow quand il est installé (poste de développement), sans bloquer sinon."""
    try:
        import mlflow
    except ImportError:
        return
    try:
        mlflow.set_experiment("reentrainement")
        with mlflow.start_run(run_name=f"decision-{etat['date']}"):
            mlflow.log_params({"modele_en_production": etat["modele_en_production"], "seuil": etat["seuil"]})
            mlflow.log_metrics({"exemples_total": etat["exemples_total"], "exemples_nouveaux": etat["exemples_nouveaux"],
                                "reentrainer": int(etat["reentrainer"])})
            mlflow.log_artifact(str(artifact))
    except Exception as exc:  # noqa: BLE001  (MLflow injoignable : la décision reste écrite sur le disque)
        print(f"MLflow non disponible : {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="préparer le paquet même sans déclencheur")
    ap.add_argument("--seuil", type=int, default=SEUIL_EXEMPLES)
    args = ap.parse_args()
    etat = run(force=args.force, seuil=args.seuil)
    print(f"Modèle en production : {etat['modele_en_production']}")
    print(f"Exemples de juristes : {etat['exemples_total']} ({etat['exemples_nouveaux']} nouveaux, seuil {etat['seuil']})")
    print("Réentraîner :", "OUI" if etat["reentrainer"] else "non")
    for r in etat["raisons"]:
        print("  -", r)
    if etat.get("conseil"):
        print("Conseil :", etat["conseil"])
    if etat["paquet"]:
        print(f"Paquet pour Colab : {WORK_DIR / etat['paquet']}")


if __name__ == "__main__":
    main()
