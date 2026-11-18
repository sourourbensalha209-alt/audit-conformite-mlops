"""Promotion d'un modèle réentraîné (champion / challenger) : il n'est mis en service que s'il fait mieux.

    python -m src.orchestration.promote models/best_model_v5

Conditions, toutes obligatoires :
  * le dossier est complet (model.safetensors, labels.json, metrics.json) ;
  * mêmes étiquettes, dans le même ordre, et même jeu de test (n_test) que le modèle en production :
    sinon les scores ne sont pas comparables ;
  * F1 macro strictement meilleur ;
  * pas de recul de plus d'un point sur les catégories rares (f1_macro_rare).

Si c'est accepté : models/production.json désigne le nouveau modèle (le service le charge au redémarrage),
et la promotion est ajoutée à models/promotions.jsonl. Sinon, rien ne change et les raisons sont affichées.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.orchestration.registry import MODELS, PRODUCTION_FILE, production_model_dir

FICHIERS = ("model.safetensors", "labels.json", "metrics.json")
RECUL_RARE_MAX = 0.01


def _metrics(model_dir: Path) -> dict:
    m = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    return m[0] if isinstance(m, list) else m


def evaluate(candidat: Path, champion: Path) -> dict:
    raisons_refus = []
    manquants = [f for f in FICHIERS if not (candidat / f).exists()]
    if manquants:
        return {"accepte": False, "raisons": [f"Fichiers manquants dans {candidat.name} : {', '.join(manquants)}"]}

    mc, mp = _metrics(candidat), _metrics(champion)
    lc = json.loads((candidat / "labels.json").read_text(encoding="utf-8"))
    lp = json.loads((champion / "labels.json").read_text(encoding="utf-8"))
    if lc != lp:
        raisons_refus.append("Étiquettes différentes de celles du modèle en production : scores non comparables.")
    if mc.get("n_test") != mp.get("n_test"):
        raisons_refus.append(f"Jeu de test différent ({mc.get('n_test')} contre {mp.get('n_test')} exemples).")

    f1c, f1p = mc.get("f1_macro", 0.0), mp.get("f1_macro", 0.0)
    rc, rp = mc.get("f1_macro_rare"), mp.get("f1_macro_rare")
    if f1c <= f1p:
        raisons_refus.append(f"F1 macro {f1c:.3f}, pas meilleur que {f1p:.3f} (modèle en production).")
    if rc is not None and rp is not None and rc < rp - RECUL_RARE_MAX:
        raisons_refus.append(f"Recul sur les catégories rares : {rc:.3f} contre {rp:.3f}.")

    comparaison = {"candidat": candidat.name, "champion": champion.name, "f1_macro": [f1c, f1p], "f1_macro_rare": [rc, rp]}
    if raisons_refus:
        return {"accepte": False, "raisons": raisons_refus, "comparaison": comparaison}
    return {"accepte": True, "raisons": [f"F1 macro {f1c:.3f} > {f1p:.3f}" +
                                         (f", catégories rares {rc:.3f} (avant {rp:.3f})" if rc is not None else "")],
            "comparaison": comparaison}


def promote(candidat: Path, production_file: Path | None = None) -> dict:
    production_file = production_file or PRODUCTION_FILE
    champion = production_model_dir(production_file)
    if candidat.resolve() == champion.resolve():
        return {"accepte": False, "raisons": [f"{candidat.name} est déjà le modèle en production."]}
    result = evaluate(candidat, champion)
    entree = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), **result}
    if result["accepte"]:
        production_file.write_text(json.dumps({"modele": candidat.name, "remplace": champion.name, **entree},
                                              indent=2, ensure_ascii=False), encoding="utf-8")
    with (production_file.parent / "promotions.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entree, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Promotion champion / challenger d'un modèle réentraîné")
    ap.add_argument("candidat", help="dossier du nouveau modèle, ex. models/best_model_v5")
    args = ap.parse_args()
    candidat = Path(args.candidat)
    if not candidat.is_absolute():
        candidat = (MODELS.parent / candidat).resolve()
    if not candidat.is_dir():
        sys.exit(f"{candidat} introuvable.")
    r = promote(candidat)
    print("PROMOTION ACCEPTÉE" if r["accepte"] else "PROMOTION REFUSÉE : le modèle en production est conservé.")
    for x in r["raisons"]:
        print("  -", x)
    if r["accepte"]:
        print(f"\nÀ faire : dvc add models/{candidat.name}   puis redémarrer le service.")


if __name__ == "__main__":
    main()
