"""Mesure du juge : réponses du modèle (Colab) contre les annotations de l'équipe et les paires construites exprès.

    python scripts/eval_judge.py
    python scripts/eval_judge.py --sheet data/judge/copie_sourour.xlsx data/judge/copie_ghayth.xlsx   # deux copies séparées

À lancer APRÈS avoir fini d'annoter data/judge/a_remplir.xlsx, et après avoir placé judge_results.jsonl dans data/judge/.
Les critères sont ceux de docs/criteres_phase5.md, fixés avant toute mesure.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, ROOT.as_posix())

from src.judge.metrics import LABELED, check_criteria, evaluate, final_labels  # noqa: E402
from src.judge.sheet import merge_sheets, read_sheet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/judge/cases.jsonl")
    ap.add_argument("--results", default="data/judge/judge_results.jsonl")
    ap.add_argument("--sheet", nargs="+", default=["data/judge/a_remplir.xlsx"],
                    help="une ou plusieurs copies de la feuille (.xlsx ou .csv) : chacune remplit sa colonne, on les fusionne")
    args = ap.parse_args()

    for name in [args.cases, args.results, *args.sheet]:
        if not (ROOT / name).exists():
            sys.exit(f"{name} introuvable.")
    cases = {c["id"]: c for c in (json.loads(l) for l in open(ROOT / args.cases, encoding="utf-8"))}
    results = [json.loads(l) for l in open(ROOT / args.results, encoding="utf-8")]
    labels, agree = final_labels(merge_sheets([read_sheet(ROOT / name) for name in args.sheet]))

    todo = [c for c in cases.values() if c["family"] in LABELED]
    print(f"=== Annotations de l'équipe : {len(labels)} cas étiquetés sur {len(todo)} à annoter ===")
    print("  répartition :", dict(Counter(labels.values())))
    if agree["cas_annotes_par_les_deux"]:
        print(f"  cas annotés par les deux : {agree['cas_annotes_par_les_deux']} | accord brut {agree['accord_brut_pct']} % | kappa {agree['kappa']}")
        if agree["kappa"] is not None and agree["kappa"] < 0.4:
            print("  ATTENTION : accord faible entre annotateurs (kappa < 0,4) : les conclusions sur l'accord avec l'équipe sont peu fiables.")
        if agree["desaccords"]:
            print("  désaccords, écartés de la mesure :", ", ".join(agree["desaccords"]))
    if len(labels) < 0.6 * len(todo):
        print(f"  ATTENTION : annotation incomplète ({len(labels)}/{len(todo)}) : terminez la feuille avant de conclure.")

    blocks = evaluate(cases, results, labels)
    print("\n=== Mesures par modèle et condition ===")
    head = f"{'modèle':26s} {'condition':13s} | {'JSON %':>6} | {'sources inventées %':>19} | {'accord équipe %':>15} | {'majoritaire %':>13} | {'fausse abst. %':>14} | {'abst. hors-sujet %':>18}"
    print(head)
    for b in blocks.values():
        ab = "-" if b["abstention_hors_sujet_pct"] is None else b["abstention_hors_sujet_pct"]
        print(f"{b['modele'].split('/')[-1]:26s} {b['condition']:13s} | {b['json_valide_pct']:>6} | {b['sources_inventees_pct']:>19} | "
              f"{b['accord_equipe_pct']:>15} | {b['reponse_majoritaire_pct']:>13} | {b['fausse_abstention_pct']:>14} | {ab:>18}")

    print("\n=== Ce que répond vraiment chaque modèle (pour juger s'il distingue quelque chose) ===")
    for b in blocks.values():
        disc = "" if b["discrimination_pts"] is None else (f" | abstention hors-sujet - abstention sur paires liées = {b['discrimination_pts']:+.1f} pts "
                                                            f"(proche de 0 : le modèle ne distingue pas)")
        print(f"  {b['modele'].split('/')[-1]:24s} {b['condition']:13s} {b['repartition_verdicts']}{disc}")
    print("  (le pouvoir de discrimination est une mesure ajoutée APRÈS le premier essai : ce n'est pas un critère fixé d'avance)")

    print("\n=== Critères fixés d'avance (condition « avec règle ») ===")
    verdicts = {}
    for key, b in blocks.items():
        if b["condition"] != "with_rule":
            continue
        print(f"\n  {b['modele']}")
        rows = check_criteria(b)
        for name, value, need, ok in rows:
            print(f"    {'OK ' if ok else 'NON'}  {name:38s} {value if value is not None else '-':>6}  (exigé : {need})")
        verdicts[key] = all(ok for *_, ok in rows)
        print(f"    -> {'TOUS LES CRITÈRES ATTEINTS' if verdicts[key] else 'CRITÈRES NON TOUS ATTEINTS'}")

    if agree.get("accord_brut_pct") is not None:
        from src.judge.metrics import CRITERIA

        maj = next((b["reponse_majoritaire_pct"] for b in blocks.values() if b["condition"] == "with_rule"), 0.0)
        seuil = max(CRITERIA["accord_min"], maj + CRITERIA["accord_marge_majoritaire"])       # le seuil EFFECTIF, comme dans check_criteria
        print(f"\n  Plafond de référence : vous deux ne vous accordez que dans {agree['accord_brut_pct']} % des cas partagés. "
              f"Le seuil d'accord effectif ({seuil:.0f} %) est {'SUPÉRIEUR à' if seuil > agree['accord_brut_pct'] else 'compatible avec'} cet accord humain.")

    print("\n=== Apport de la règle fournie (même modèle, avec contre sans) ===")
    for key, b in blocks.items():
        if b["condition"] != "with_rule":
            continue
        w = blocks.get(f"{b['modele']}|without_rule")
        if w:
            print(f"  {b['modele'].split('/')[-1]:26s} accord {b['accord_equipe_pct']} % avec règle contre {w['accord_equipe_pct']} % sans "
                  f"({b['accord_equipe_pct'] - w['accord_equipe_pct']:+.1f} pts) | sources inventées {b['sources_inventees_pct']} % contre {w['sources_inventees_pct']} %")

    out = ROOT / "reports" / "juge_resultats.json"
    out.write_text(json.dumps({"annotations": {"etiquetes": len(labels), "a_annoter": len(todo), **agree},
                               "mesures": blocks, "criteres_atteints": verdicts}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultats enregistrés : {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
