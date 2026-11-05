"""Rapport d'audit d'un contrat : une page HTML autonome (aucun serveur, aucun accès internet).

Le rapport ne reprend QUE ce qui a été mesuré (voir README, « Phase 5 ») :
  * identité du contrat (nom, parties, dates)        : extracteur à règles, mesuré sur 102 contrats
  * clauses retrouvées, avec règle de contrôle        : classifieur Legal-BERT v4, mesuré sur 30 contrats jamais regardés
  * clauses attendues manquantes                      : « manquante » seulement là où c'est fiable, « à vérifier » ailleurs
  * contrôles RGPD à vérifier sur les clauses de données personnelles
Il ne produit AUCUN verdict automatique « conforme / non conforme » : les modèles ouverts testés ne distinguent pas ces cas.
"""

from __future__ import annotations

import html
import re
from datetime import datetime

from ..data.autre import NEUTRAL_LABEL
from ..data.header import HEADER_CATEGORIES, detect_header
from ..data.segment import segment
from ..retrieval.rules import rule_for
from .cases import _pick_check, is_personal_data_clause

TAU = 0.5
# « manquante » n'est affirmé que pour les catégories où la mesure (30 contrats, critère fixé d'avance) n'a donné AUCUNE fausse alerte.
# Pour les autres (dates, parties, durée), « non trouvée » est souvent faux : le rapport écrit « à vérifier ».
RELIABLE_MISSING = {"Cap On Liability", "Governing Law"}
GRAVITE = {"high": "élevée", "medium": "moyenne", "low": "faible"}


def analyse(text: str, labels: list[str], probs_by_segment: list[list[float]], book, tau: float = TAU) -> dict:
    segs = segment(text)
    if len(segs) != len(probs_by_segment):
        raise ValueError(f"{len(segs)} segments mais {len(probs_by_segment)} lignes de probabilités")
    checks = {c["id"]: c for c in book.rgpd_checks}
    found_segments, best = [], {}
    for n, (seg, p) in enumerate(zip(segs, probs_by_segment), 1):
        i = max(range(len(p)), key=p.__getitem__)
        label, conf = labels[i], p[i]
        if label == NEUTRAL_LABEL or conf < tau:
            continue
        rule = rule_for(book, label)
        item = {"n": n, "categorie": label, "confiance": conf, "texte": seg.text,
                "regle": ({"id": rule["id"], "titre": rule["titre"], "regle": rule["regle"], "gravite": rule["gravite"]} if rule else None),
                "rgpd": None}
        if checks and is_personal_data_clause(seg.text):
            chk = _pick_check(seg.text, checks)
            item["rgpd"] = {"id": chk["id"], "theme": chk["theme"], "regle": chk["regle"], "refs": chk["refs"]}
        found_segments.append(item)
        if label not in best or conf > best[label]["confiance"]:
            best[label] = item

    header = detect_header(text)
    identite = [{"categorie": c, "trouve": header[c] is not None, "texte": _display(c, header[c].text) if header[c] else ""} for c in HEADER_CATEGORIES]

    attendues = []
    for r in book.rules:
        if not r["absence_a_signaler"]:
            continue
        cat = r["category"]
        if cat in HEADER_CATEGORIES:
            found = header[cat]
            preuve = {"texte": found.text, "confiance": None, "source": "extracteur à règles"} if found else None
        else:
            hit = best.get(cat)
            preuve = {"texte": hit["texte"], "confiance": hit["confiance"], "source": "classifieur"} if hit else None
        statut = "presente" if preuve else ("manquante" if cat in RELIABLE_MISSING else "a_verifier")
        attendues.append({"categorie": cat, "titre": r["titre"], "regle": r["regle"], "gravite": r["gravite"],
                          "statut": statut, "preuve": preuve})
    order = {"manquante": 0, "a_verifier": 1, "presente": 2}
    attendues.sort(key=lambda a: (order[a["statut"]], a["categorie"]))

    return {"identite": identite, "attendues": attendues, "clauses": found_segments,
            "stats": {"segments": len(segs), "clauses": len(found_segments),
                      "categories": len({c["categorie"] for c in found_segments}),
                      "rgpd": sum(1 for c in found_segments if c["rgpd"])}}


def _display(category: str, text: str) -> str:
    """Affichage seulement (la mesure de l'extracteur n'est pas touchée) : un titre ne commence pas par un numéro d'annexe isolé."""
    return re.sub(r"^(?:\d+\s+)+", "", text) if category == "Document Name" else text


_BADGE = {"presente": ("Présente", "ok"), "manquante": ("Manquante", "ko"), "a_verifier": ("À vérifier", "warn")}

_CSS = """
body{font-family:'Segoe UI',Arial,sans-serif;color:#1f2937;margin:0;background:#f8f6f1;line-height:1.45}
.wrap{max-width:1040px;margin:0 auto;padding:24px}
header{background:#14213d;color:#fff;padding:26px 0}
header h1{font-family:Georgia,'Times New Roman',serif;font-size:28px;margin:0 0 6px}
header .sub{color:#e5a93c;font-size:15px}
header .meta{font-size:13px;color:#cbd5e1;margin-top:10px}
h2{font-family:Georgia,'Times New Roman',serif;color:#14213d;border-bottom:2px solid #e5a93c;padding-bottom:4px;margin-top:34px}
.note{background:#fff7e6;border-left:4px solid #e5a93c;padding:10px 14px;margin:16px 0;font-size:14px}
table{border-collapse:collapse;width:100%;background:#fff;font-size:14px}
th{background:#14213d;color:#fff;text-align:left;padding:8px 10px}
td{border-bottom:1px solid #e5e0d3;padding:8px 10px;vertical-align:top}
.badge{display:inline-block;padding:2px 10px;border-radius:10px;font-weight:600;font-size:13px;white-space:nowrap}
.ok{background:#dcfce7;color:#14532d}.ko{background:#fee2e2;color:#7f1d1d}.warn{background:#ffedd5;color:#7c2d12}
.small{font-size:12.5px;color:#4a5568}.excerpt{color:#374151}
.kpi span{display:inline-block;margin:0 22px 8px 0;font-size:14px}.kpi b{font-size:22px;color:#14213d}
footer{font-size:12.5px;color:#4a5568;margin:34px 0 20px}
"""


def _e(s) -> str:
    return html.escape(str(s))


def _cut(t: str, n: int) -> str:
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " …"


def render_html(a: dict, meta: dict) -> str:
    s = a["stats"]
    rows_id = "".join(
        f"<tr><td><b>{_e(i['categorie'])}</b></td><td>" +
        (f"<span class='badge ok'>Trouvé</span></td><td class='excerpt'>{_e(_cut(i['texte'], 220))}</td></tr>" if i["trouve"]
         else "<span class='badge warn'>À vérifier</span></td><td class='small'>Non trouvé automatiquement.</td></tr>")
        for i in a["identite"])
    rows_att = ""
    for x in a["attendues"]:
        label, cls = _BADGE[x["statut"]]
        if x["preuve"]:
            conf = f" · confiance {x['preuve']['confiance']:.2f}" if x["preuve"]["confiance"] is not None else ""
            detail = f"<div class='excerpt'>{_e(_cut(x['preuve']['texte'], 260))}</div><div class='small'>Trouvée par : {_e(x['preuve']['source'])}{conf}</div>"
        elif x["statut"] == "manquante":
            detail = "<div class='small'>Aucun passage trouvé. Pour cette catégorie, « manquante » a été fiable à 100 % sur 30 contrats testés.</div>"
        else:
            detail = "<div class='small'>Non trouvée automatiquement. Pour cette catégorie, le test a donné trop de fausses alertes : à relire par un humain.</div>"
        rows_att += (f"<tr><td><b>{_e(x['titre'])}</b><div class='small'>Gravité {GRAVITE.get(x['gravite'], _e(x['gravite']))}</div></td>"
                     f"<td><span class='badge {cls}'>{label}</span></td>"
                     f"<td><div class='small'>{_e(x['regle'])}</div>{detail}</td></tr>")
    rows_cl = ""
    for c in a["clauses"]:
        rule = (f"<b>[{_e(c['regle']['id'])}] {_e(c['regle']['titre'])}</b><div class='small'>{_e(_cut(c['regle']['regle'], 230))}</div>"
                if c["regle"] else "<span class='small'>Pas de règle associée.</span>")
        rgpd = (f"<div class='small' style='margin-top:6px'><span class='badge warn'>RGPD</span> Contrôle {_e(c['rgpd']['id'])} · "
                f"{_e(c['rgpd']['theme'])} — {_e(', '.join(c['rgpd']['refs']))}. <b>À vérifier par un humain.</b></div>" if c["rgpd"] else "")
        rows_cl += (f"<tr><td>{c['n']}</td><td><b>{_e(c['categorie'])}</b><div class='small'>confiance {c['confiance']:.2f}</div></td>"
                    f"<td class='excerpt'>{_e(_cut(c['texte'], 330))}</td><td>{rule}{rgpd}</td></tr>")
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rapport d'audit — {_e(meta['titre'])}</title><style>{_CSS}</style></head><body>
<header><div class="wrap"><h1>Rapport d'audit de contrat</h1>
<div class="sub">{_e(meta['titre'])}</div>
<div class="meta">Généré le {_e(meta['date'])} · modèle {_e(meta['modele'])} · corpus de lois version {_e(meta['version_corpus'])}</div></div></header>
<div class="wrap">
<div class="note"><b>Aperçu technique, pas un avis juridique.</b> Ce rapport ne rend aucun verdict « conforme / non conforme » : les modèles
de langage ouverts testés ne distinguent pas ces cas (voir README, Phase 5). Il signale ce qui est mesuré comme fiable et marque
« à vérifier » ce qui demande un relecteur humain. Les règles de contrôle sont des contrôles de l'équipe, rattachés aux définitions CUAD,
sans relecture juridique.</div>
<div class="kpi"><span><b>{s['segments']}</b> paragraphes analysés</span> · <span><b>{s['clauses']}</b> clauses retrouvées</span> ·
<span><b>{s['categories']}</b> catégories distinctes</span> · <span><b>{s['rgpd']}</b> clause(s) de données personnelles</span></div>
<h2>1. Identité du contrat</h2>
<table><tr><th>Information</th><th>État</th><th>Passage trouvé</th></tr>{rows_id}</table>
<h2>2. Clauses attendues</h2>
<table><tr><th>Clause attendue</th><th>État</th><th>Détail</th></tr>{rows_att}</table>
<h2>3. Clauses retrouvées dans le contrat</h2>
<table><tr><th>#</th><th>Catégorie</th><th>Extrait</th><th>Règle de contrôle associée</th></tr>{rows_cl}</table>
<footer>Mesures de référence (README) : classification des clauses, rappel 79 % et 9,5 % de fausses détections sur 30 contrats jamais regardés ;
extraction de l'identité mesurée sur 102 contrats ; clauses manquantes : précision 62,5 % contre 22,4 % de base.</footer>
</div></body></html>"""


def build_meta(titre: str, modele: str, version_corpus: str) -> dict:
    return {"titre": titre, "modele": modele, "version_corpus": version_corpus, "date": datetime.now().strftime("%d/%m/%Y %H:%M")}
