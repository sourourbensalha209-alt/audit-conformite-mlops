"""Monitoring du service d'audit : chaque audit laisse une trace, et l'on surveille le service ET le modèle.

Ce qui est enregistré (data/monitoring/audits.jsonl, une ligne par audit) :
  * service : statut (termine / erreur), durée, taille du document ;
  * modèle  : confiance moyenne, part des paragraphes sous le seuil (abstention), part classée « Autre ».

Dérive des données : on compare les audits récents à la RÉFÉRENCE mesurée sur 30 contrats jamais vus
(reports/bout_en_bout_best_model_v4_30contrats_frais.json : 79,4 % « Autre », 1,8 % d'abstention).
Si les contrats reçus ne ressemblent plus à ceux de l'entraînement (autre langue, autre type de contrat,
PDF mal lu), ces deux parts s'écartent : une alerte le signale avant que la qualité ne baisse en silence.

Retour des juristes : le taux de correction (corrections / audits) mesure directement la qualité perçue.

Exposé par l'API :  GET /monitoring  (JSON, pour l'interface)   ·   GET /metrics  (format Prometheus)
"""

from __future__ import annotations

import json
import statistics
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = ROOT / "data" / "monitoring" / "audits.jsonl"
FEEDBACK_FILE = ROOT / "data" / "feedback" / "corrections.jsonl"

TAU = 0.5                              # même seuil que le rapport (src/judge/report.py)
REFERENCE = {                          # mesuré sur 30 contrats jamais vus (rapport bout-en-bout, modèle v4)
    "part_autre": 0.794,
    "part_abstention": 0.018,
    "source": "reports/bout_en_bout_best_model_v4_30contrats_frais.json",
}
SEUILS = {                             # au-delà : alerte
    "ecart_part_autre": 0.15,          # ± 15 points autour de 79,4 %
    "part_abstention_max": 0.10,       # plus de 10 % de paragraphes incertains (référence : 1,8 %)
    "taux_erreur_max": 0.10,           # plus de 10 % d'audits en erreur
    "duree_max_s": 120.0,              # audit moyen de plus de 2 minutes
    "taux_correction_max": 0.30,       # plus de 30 % des audits corrigés par un juriste
}
FENETRE = 20                           # les alertes portent sur les 20 derniers audits
MIN_AUDITS_DERIVE = 5                  # dérive et taux de correction se jugent sur plusieurs audits, jamais sur un seul

_lock = threading.Lock()


def model_stats(probs: list[list[float]], labels: list[str], neutral: str = "Autre", tau: float = TAU) -> dict:
    """Indicateurs du modèle sur un document, à partir des probabilités de chaque paragraphe."""
    if not probs:
        return {"paragraphes": 0, "confiance_moyenne": None, "part_abstention": None, "part_autre": None}
    best = [max(range(len(p)), key=p.__getitem__) for p in probs]
    conf = [p[i] for p, i in zip(probs, best)]
    n = len(probs)
    return {
        "paragraphes": n,
        "confiance_moyenne": round(statistics.fmean(conf), 4),
        "part_abstention": round(sum(c < tau for c in conf) / n, 4),
        "part_autre": round(sum(labels[i] == neutral for i in best) / n, 4),
    }


def record(event: dict, path: Path | None = None) -> None:
    """Ajoute un audit au journal (sûr même si plusieurs audits se terminent en même temps)."""
    path = path or LOG_FILE
    event = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:          # ligne tronquée (arrêt brutal) : ignorée
            continue
    return rows


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 4) if values else None


def summary(log_path: Path | None = None, feedback_path: Path | None = None) -> dict:
    """Vue d'ensemble pour l'interface : compteurs, indicateurs récents, alertes et série par audit."""
    audits = _read(log_path or LOG_FILE)
    corrections = _read(feedback_path or FEEDBACK_FILE)
    ok = [a for a in audits if a.get("statut") == "termine"]
    recent_all = audits[-FENETRE:]
    recent = [a for a in recent_all if a.get("statut") == "termine"]

    indicateurs = {
        "audits": len(audits),
        "termines": len(ok),
        "erreurs": len(audits) - len(ok),
        "corrections": len(corrections),
        "taux_erreur": round((len(recent_all) - len(recent)) / len(recent_all), 4) if recent_all else None,
        "duree_moyenne_s": _mean([a.get("duree_s") for a in recent]),
        "confiance_moyenne": _mean([a.get("confiance_moyenne") for a in recent]),
        "part_abstention": _mean([a.get("part_abstention") for a in recent]),
        "part_autre": _mean([a.get("part_autre") for a in recent]),
        "taux_correction": round(len({c.get("job_id") for c in corrections}) / len(ok), 4) if ok else None,
    }

    alertes = []
    i = indicateurs
    derive_evaluable = len(recent) >= MIN_AUDITS_DERIVE
    if derive_evaluable and i["part_autre"] is not None and abs(i["part_autre"] - REFERENCE["part_autre"]) > SEUILS["ecart_part_autre"]:
        alertes.append({"niveau": "derive", "message":
                        f"Part « Autre » {i['part_autre']:.0%} contre {REFERENCE['part_autre']:.0%} en référence : "
                        "les contrats reçus ne ressemblent plus à ceux de l'entraînement."})
    if derive_evaluable and i["part_abstention"] is not None and i["part_abstention"] > SEUILS["part_abstention_max"]:
        alertes.append({"niveau": "derive", "message":
                        f"{i['part_abstention']:.0%} de paragraphes incertains (référence {REFERENCE['part_abstention']:.1%}) : "
                        "le modèle hésite plus que d'habitude."})
    if i["taux_erreur"] is not None and i["taux_erreur"] > SEUILS["taux_erreur_max"]:
        alertes.append({"niveau": "service", "message": f"{i['taux_erreur']:.0%} des derniers audits ont échoué."})
    if i["duree_moyenne_s"] is not None and i["duree_moyenne_s"] > SEUILS["duree_max_s"]:
        alertes.append({"niveau": "service", "message": f"Durée moyenne {i['duree_moyenne_s']:.0f} s : service lent."})
    if len(ok) >= MIN_AUDITS_DERIVE and i["taux_correction"] is not None and i["taux_correction"] > SEUILS["taux_correction_max"]:
        alertes.append({"niveau": "qualite", "message":
                        f"{i['taux_correction']:.0%} des audits corrigés par un juriste : envisager un réentraînement."})

    serie = [{k: a.get(k) for k in ("date", "document", "statut", "duree_s", "confiance_moyenne",
                                    "part_abstention", "part_autre", "paragraphes")} for a in audits[-50:]]
    return {"indicateurs": indicateurs, "alertes": alertes, "reference": REFERENCE, "seuils": SEUILS,
            "fenetre": FENETRE, "serie": serie,
            "derive": {"evaluable": derive_evaluable, "audits": len(recent), "minimum": MIN_AUDITS_DERIVE}}


def prometheus(log_path: Path | None = None, feedback_path: Path | None = None) -> str:
    """Mêmes indicateurs au format texte de Prometheus (outil standard de supervision)."""
    s = summary(log_path, feedback_path)
    i = s["indicateurs"]
    lines = [
        "# HELP audit_total Nombre d'audits par statut.",
        "# TYPE audit_total counter",
        f'audit_total{{statut="termine"}} {i["termines"]}',
        f'audit_total{{statut="erreur"}} {i["erreurs"]}',
        "# HELP audit_corrections_total Corrections enregistrées par les juristes.",
        "# TYPE audit_corrections_total counter",
        f"audit_corrections_total {i['corrections']}",
    ]
    gauges = {
        "audit_duree_moyenne_secondes": ("Durée moyenne des derniers audits.", i["duree_moyenne_s"]),
        "modele_confiance_moyenne": ("Confiance moyenne du classifieur (derniers audits).", i["confiance_moyenne"]),
        "modele_part_abstention": ("Part des paragraphes sous le seuil de confiance.", i["part_abstention"]),
        "modele_part_autre": ("Part des paragraphes classés Autre.", i["part_autre"]),
        "audit_alertes": ("Nombre d'alertes actives.", len(s["alertes"])),
    }
    for name, (doc, value) in gauges.items():
        if value is None:
            continue
        lines += [f"# HELP {name} {doc}", f"# TYPE {name} gauge", f"{name} {value}"]
    return "\n".join(lines) + "\n"
