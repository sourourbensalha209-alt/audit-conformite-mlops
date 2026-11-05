"""Mesures du juge. Tout est calculé à partir des sorties brutes du modèle, jamais de chiffres saisis à la main.

Familles de cas (voir src/judge/cases.py) :
  playbook, rgpd   clauses réelles de contrats + règle ADAPTÉE ; verdict de référence = annotation de l'équipe
  mismatch         clause + règle SANS RAPPORT ; verdict attendu : a_verifier (construit exprès, aucune annotation)
  neutral          paragraphe neutre + règle quelconque ; verdict attendu : a_verifier
"""

from __future__ import annotations

from collections import Counter

from .parse import check_citations, parse_output
from .prompt import VERDICTS

LABELED = ("playbook", "rgpd")
UNRELATED = ("mismatch", "neutral")

# Critères fixés AVANT de lancer le modèle (docs/criteres_phase5.md). Ne pas les modifier après avoir vu un résultat.
CRITERIA = {
    "json_valide_min": 95.0,          # % de réponses en JSON propre
    "sources_inventees_max": 5.0,     # % de réponses citant une source qui n'était pas dans le prompt
    "abstention_hors_sujet_min": 70.0,  # % de a_verifier sur les paires sans rapport
    "accord_min": 60.0,               # % d'accord avec l'équipe sur les cas annotés ...
    "accord_marge_majoritaire": 10.0,   # ... ET au moins 10 points au-dessus de la réponse toujours majoritaire
    "fausse_abstention_max": 30.0,    # % de a_verifier alors que l'équipe a tranché
}


def pct(a: int, b: int) -> float:
    return round(100 * a / b, 1) if b else 0.0


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Accord entre deux annotateurs, corrigé du hasard (1 = parfait, 0 = hasard). None si trop peu de cas."""
    if len(a) < 5 or len(a) != len(b):
        return None
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[v] * cb[v] for v in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else round((po - pe) / (1 - pe), 2)


def final_labels(rows: list[dict]) -> tuple[dict, dict]:
    """rows : lignes de la feuille d'annotation ({id, verdict_sourour, verdict_ghayth}).
    Retourne (étiquette finale par cas, statistiques d'accord entre annotateurs).
    Un cas annoté par une seule personne garde son étiquette ; un cas annoté par les deux sans accord est écarté."""
    final, a_list, b_list, conflicts = {}, [], [], []
    for r in rows:
        a, b = (r.get("verdict_sourour") or "").strip(), (r.get("verdict_ghayth") or "").strip()
        a, b = (a if a in VERDICTS else ""), (b if b in VERDICTS else "")
        if a and b:
            a_list.append(a); b_list.append(b)
            if a == b:
                final[r["id"]] = a
            else:
                conflicts.append(r["id"])
        elif a or b:
            final[r["id"]] = a or b
    agree = pct(sum(x == y for x, y in zip(a_list, b_list)), len(a_list))
    return final, {"cas_annotes_par_les_deux": len(a_list), "accord_brut_pct": agree if a_list else None,
                   "kappa": cohen_kappa(a_list, b_list), "desaccords": conflicts}


def evaluate(cases: dict[str, dict], results: list[dict], labels: dict[str, str]) -> dict:
    """results : [{case_id, model, condition ('with_rule'|'without_rule'), raw}]. Un bloc de mesures par (modèle, condition)."""
    groups: dict[tuple, list] = {}
    for r in results:
        groups.setdefault((r["model"], r["condition"]), []).append(r)
    out = {}
    for (model, cond), items in groups.items():
        parsed = [(r, cases[r["case_id"]], parse_output(r["raw"])) for r in items]
        n = len(parsed)
        valid = [p for _, _, p in parsed if p["valid"]]
        cited = invalid_cit = 0
        for r, c, p in parsed:
            allowed = c.get("allowed_sources", []) if cond == "with_rule" else []
            n_c, n_bad = check_citations(p["sources"], allowed) if p["valid"] else (0, 0)
            cited += n_c > 0
            invalid_cit += n_bad > 0
        unrel = [(c, p) for _, c, p in parsed if c["family"] in UNRELATED]
        lab = [(c, p, labels[c["id"]]) for _, c, p in parsed if c["family"] in LABELED and c["id"] in labels]
        related = [(c, p) for _, c, p in parsed if c["family"] in LABELED]          # TOUTES les paires liées, annotées ou non
        abst_related = pct(sum(p["verdict"] == "a_verifier" for _, p in related), len(related)) if related else None
        majority = Counter(l for _, _, l in lab).most_common(1)
        acc = pct(sum(p["verdict"] == l for _, p, l in lab), len(lab))
        trancher = [(p, l) for _, p, l in lab if l != "a_verifier"]
        by_fam = {f: pct(sum(p["verdict"] == l for c, p, l in lab if c["family"] == f), sum(1 for c, _, _ in lab if c["family"] == f))
                  for f in LABELED}
        out[f"{model}|{cond}"] = {
            "modele": model, "condition": cond, "reponses": n,
            "json_valide_pct": pct(len(valid), n),
            "verdict_lu_pct": pct(sum(p["valid"] or p["recovered"] for _, _, p in parsed), n),
            "reponses_citant_une_source_pct": pct(cited, n),
            "sources_inventees_pct": pct(invalid_cit, n),
            "abstention_hors_sujet_pct": pct(sum(p["verdict"] == "a_verifier" for _, p in unrel), len(unrel)) if unrel else None,
            "cas_hors_sujet": len(unrel),
            # Mesure ajoutée APRÈS le premier essai (pas un critère fixé d'avance) : un juge utile s'abstient sur les paires sans
            # rapport mais PAS sur les paires liées. Un juge qui répond toujours pareil obtient un écart proche de 0.
            "abstention_paires_liees_pct": abst_related,
            "discrimination_pts": (round(pct(sum(p["verdict"] == "a_verifier" for _, p in unrel), len(unrel)) - abst_related, 1)
                                   if unrel and abst_related is not None else None),
            "cas_annotes": len(lab),
            "accord_equipe_pct": acc,
            "reponse_majoritaire_pct": pct(majority[0][1], len(lab)) if majority else 0.0,
            "accord_par_famille_pct": by_fam,
            "fausse_abstention_pct": pct(sum(p["verdict"] == "a_verifier" for p, _ in trancher), len(trancher)),
            "repartition_verdicts": dict(Counter(p["verdict"] for _, _, p in parsed)),
        }
    return out


def check_criteria(block: dict) -> list[tuple[str, float, str, bool]]:
    """Critères fixés d'avance, pour un bloc (modèle, 'with_rule'). Retourne (nom, valeur, exigence, atteint)."""
    c = CRITERIA
    seuil_accord = max(c["accord_min"], block["reponse_majoritaire_pct"] + c["accord_marge_majoritaire"])
    return [
        ("JSON valide", block["json_valide_pct"], f">= {c['json_valide_min']:.0f} %", block["json_valide_pct"] >= c["json_valide_min"]),
        ("Sources inventées", block["sources_inventees_pct"], f"<= {c['sources_inventees_max']:.0f} %", block["sources_inventees_pct"] <= c["sources_inventees_max"]),
        ("Abstention sur les cas sans rapport", block["abstention_hors_sujet_pct"], f">= {c['abstention_hors_sujet_min']:.0f} %",
         (block["abstention_hors_sujet_pct"] or 0) >= c["abstention_hors_sujet_min"]),
        ("Accord avec l'équipe", block["accord_equipe_pct"], f">= {seuil_accord:.0f} % (60 % et majoritaire + 10)", block["accord_equipe_pct"] >= seuil_accord),
        ("Fausse abstention", block["fausse_abstention_pct"], f"<= {c['fausse_abstention_max']:.0f} %", block["fausse_abstention_pct"] <= c["fausse_abstention_max"]),
    ]
