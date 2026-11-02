"""Essai de bout en bout : découpage en clauses + Legal-BERT, mesuré sur des contrats annotés par des juristes.

    python scripts/eval_pipeline.py --model models/best_model_v2 --contracts 10

Pour chaque seuil de confiance, le tableau final donne :
  rappel %      parmi les (contrat, catégorie) annotés par les juristes, part retrouvée
  fantômes %    parmi les paragraphes SANS annotation, part qui reçoit quand même une catégorie
  abstention %  part des segments laissés "à vérifier par un humain" faute de confiance
  Autre %       part des segments classés « Autre » (modèle de l'itération 3 : aucune catégorie annoncée)

C'est cette table qui permet de choisir le seuil de confiance : on cherche le seuil où les fantômes
baissent nettement sans que le rappel s'effondre. Legal-BERT n'a pas de classe "autre" : sans seuil, il
annonce une catégorie sur n'importe quel paragraphe.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.segment import segment  # noqa: E402
from src.data.header import HEADER_CATEGORIES, score_header  # noqa: E402
from src.data.segment_eval import CRITERE, critere_atteint, diagnose, load_docs, pick_contracts, score_pipeline, sweep_autre  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/best_model_v2")
    ap.add_argument("--test", default="data/raw/cuad/test.jsonl")
    ap.add_argument("--contracts", type=int, default=10)
    ap.add_argument("--max-chars", type=int, default=120000)
    ap.add_argument("--fresh", action="store_true",
                    help="écarte les 8 contrats déjà utilisés pour diagnostiquer le système : mesure sur des contrats jamais regardés")
    ap.add_argument("--no-cache", action="store_true", help="reclasse tout, même si un résultat identique est en mémoire")
    ap.add_argument("--mlflow", action="store_true", help="enregistre le tableau dans MLflow (expérience 03-bout-en-bout)")
    args = ap.parse_args()

    from src.models.predict import ClauseClassifier

    model_dir = Path(args.model)
    clf = ClauseClassifier(model_dir if model_dir.is_absolute() else ROOT / model_dir, threshold=0.0)
    corpus = load_docs(ROOT / args.test)
    already_seen = {d["title"] for d in pick_contracts(corpus, 8, args.max_chars)} if args.fresh else set()
    docs = pick_contracts(corpus, args.contracts, args.max_chars, already_seen)
    print(f"{len(docs)} contrats{' (jamais utilisés pour le diagnostic)' if args.fresh else ''}, périphérique : {clf.device}\n")

    segs_by_doc = [segment(d["text"]) for d in docs]
    cache_path = ROOT / "reports" / f"_cache_pipeline_{model_dir.name}_{len(docs)}{'_fresh' if args.fresh else ''}.json"
    signature = {"titres": [d["title"] for d in docs], "segments": [len(x) for x in segs_by_doc], "labels": clf.labels}
    probs_by_doc = None
    if cache_path.exists() and not args.no_cache:
        saved = json.loads(cache_path.read_text(encoding="utf-8"))
        if saved["signature"] == signature:
            probs_by_doc = saved["probs"]
            print("Résultats déjà calculés pour ce modèle et ces contrats : réutilisés (--no-cache pour refaire).")
    if probs_by_doc is None:
        probs_by_doc = []
        t0 = time.time()
        for i, (d, segs) in enumerate(zip(docs, segs_by_doc), 1):
            out = clf.predict([s.text for s in segs])
            probs_by_doc.append([p.probs for p in out])
            print(f"  [{i}/{len(docs)}] {d['title'][:50]:50s} {len(segs):4d} segments   ({time.time() - t0:.0f} s)")
        cache_path.write_text(json.dumps({"signature": signature, "probs": probs_by_doc}), encoding="utf-8")
    preds_by_doc = [[(clf.labels[max(range(len(p)), key=p.__getitem__)], max(p)) for p in probs] for probs in probs_by_doc]

    # Mode hybride : si le modèle ne connaît pas les 4 catégories d'identité, c'est l'extracteur à règles qui les traite.
    # Le classifieur est alors jugé sur ses seules catégories ; un bloc de signature classé comme une clause est un fantôme.
    identity = set(HEADER_CATEGORIES)
    hybrid = not (identity & set(clf.labels))
    all_docs = docs
    if hybrid:
        docs = [dict(d, spans=[sp for sp in d["spans"] if sp[0] not in identity]) for d in all_docs]
        print(f"\nMode hybride : le classifieur est jugé sur ses {len(clf.labels) - 1} catégories ; l'extracteur à règles traite {', '.join(sorted(identity))}.")

    print(f"\n{sum(len(s) for s in segs_by_doc)} segments classés\n")
    print(f"{'seuil':>6} | {'rappel %':>9} | {'fantômes %':>10} | {'abstention %':>12} | {'Autre %':>8}")
    for r in score_pipeline(docs, segs_by_doc, preds_by_doc):
        print(f"{r['seuil']:>6} | {r['rappel %']:>9} | {r['fantômes %']:>10} | {r['abstention %']:>12} | {r['Autre %']:>8}")
    print(f"\n(rappel calculé sur {r['contrat-catégories annotées']} couples contrat-catégorie annotés)")

    diag = diagnose(docs, segs_by_doc, preds_by_doc)
    print("\nDiagnostic : où les clauses sont-elles perdues ? (réponse la plus probable, sans seuil)")
    print(f"  {'longueur du segment':>20} | {'segments avec clause':>20} | {'bonne étiquette %':>17} | {'classés Autre %':>15}")
    for r in diag["par_longueur"]:
        print(f"  {r['longueur']:>20} | {r['segments avec clause']:>20} | {r['bonne étiquette %']:>17} | {r['classés Autre %']:>15}")
    h, o = diag["en_tetes"], diag["autres_categories"]
    if h["total"]:
        print(f"  Catégories d'en-tête restantes au classifieur ({', '.join(diag['en_tetes_categories'])}) : "
              f"{h['trouvés']}/{h['total']} retrouvées ({h['rappel %']} %)")
    print(f"  Autres catégories               : {o['trouvés']}/{o['total']} retrouvées ({o['rappel %']} %)")
    print("  Les plus manquées :", ", ".join(f"{c} {f}/{t}" for c, f, t in diag["pires"]))

    sweep, fx, h2, found_h, total_h = [], None, {}, 0, 0
    if "Autre" in clf.labels:
        sweep = sweep_autre(docs, segs_by_doc, probs_by_doc, clf.labels)
        print("\nRéglage de « Autre » : on ne répond « Autre » que si le modèle en est sûr à au moins a (argmax = le plus probable)")
        print(f"{'a':>7} | {'seuil':>5} | {'rappel %':>9} | {'fantômes %':>10} | {'abstention %':>12} | {'Autre %':>8}")
        for r in sweep:
            print(f"{r['règle']:>7} | {r['seuil']:>5} | {r['rappel %']:>9} | {r['fantômes %']:>10} | {r['abstention %']:>12} | {r['Autre %']:>8}")
        ok = critere_atteint(sweep)
        print(f"\nCritère fixé d'avance (seuil {CRITERE['seuil']}) : rappel >= {CRITERE['rappel_min']:.0f} % ET fantômes <= {CRITERE['fantomes_max']:.0f} %")
        print("  ->", f"ATTEINT avec a = {', '.join(ok)}" if ok else "NON ATTEINT par ces réglages")
        fixed = next((r for r in sweep if r["règle"] == "argmax" and r["seuil"] == CRITERE["seuil"]), None)
        if fixed:
            reached = fixed["rappel %"] >= CRITERE["rappel_min"] and fixed["fantômes %"] <= CRITERE["fantomes_max"]
            print(f"\nRÉGLAGE UNIQUE FIXÉ D'AVANCE (réponse la plus probable, seuil {CRITERE['seuil']}) : "
                  f"rappel {fixed['rappel %']} %, fantômes {fixed['fantômes %']} %  ->  {'CRITÈRE ATTEINT' if reached else 'CRITÈRE NON ATTEINT'}")
            print("(c'est cette ligne qui fait foi ; les autres réglages ne sont qu'indicatifs)")

    if hybrid:
        h2 = score_header(all_docs)
        found_h, total_h = sum(f for f, t in h2.values()), sum(t for f, t in h2.values())
        print("\nExtracteur à règles (bonne valeur retrouvée, par contrat) :")
        for cat, (f, t) in h2.items():
            print(f"  {cat:16s} {f:>3}/{t:<3} ({100 * f / max(t, 1):.0f} %)")
        print(f"  Ensemble          {found_h:>3}/{total_h:<3} ({100 * found_h / max(total_h, 1):.0f} %)")
        if sweep:
            print("\nRappel d'ENSEMBLE (classifieur + extracteur), sur toutes les catégories annotées :")
            for r in sweep:
                if r["seuil"] == CRITERE["seuil"]:
                    tot = (r["trouvés"] + found_h) / max(r["contrat-catégories annotées"] + total_h, 1)
                    print(f"  règle {r['règle']:>6}, seuil {r['seuil']} : rappel d'ensemble {100 * tot:.1f} %  (classifieur seul {r['rappel %']} %, fantômes {r['fantômes %']} %)")

    # Résultats conservés dans le projet (à versionner avec le code : ils sont la preuve des chiffres annoncés)
    summary = {
        "modele": model_dir.name, "contrats": len(docs), "jamais_utilises_pour_le_diagnostic": bool(args.fresh),
        "segments": sum(len(x) for x in segs_by_doc), "mode": "hybride" if hybrid else "classifieur seul",
        "tableau_seuils": score_pipeline(docs, segs_by_doc, preds_by_doc), "reglages_autre": sweep, "diagnostic": diag,
        "critere": CRITERE,
    }
    if sweep:
        fx = next((r for r in sweep if r["règle"] == "argmax" and r["seuil"] == CRITERE["seuil"]), None)
        if fx:
            summary["reglage_unique_fixe_d_avance"] = {**fx, "atteint": fx["rappel %"] >= CRITERE["rappel_min"] and fx["fantômes %"] <= CRITERE["fantomes_max"]}
    if hybrid:
        summary["extracteur_identite"] = {c: {"trouves": f, "total": t} for c, (f, t) in h2.items()}
        if fx:
            summary["rappel_d_ensemble_pct"] = round(100 * (fx["trouvés"] + found_h) / max(fx["contrat-catégories annotées"] + total_h, 1), 1)
    out_json = ROOT / "reports" / f"bout_en_bout_{model_dir.name}_{len(docs)}contrats{'_frais' if args.fresh else ''}.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultats enregistrés : {out_json.relative_to(ROOT)}")

    if args.mlflow:
        try:
            import mlflow

            try:
                from src.config import MLFLOW_TRACKING_URI
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            except Exception:  # noqa: BLE001
                pass
            mlflow.set_experiment("03-bout-en-bout")
            with mlflow.start_run(run_name=f"bout-en-bout-{model_dir.name}-{len(docs)}contrats{'-frais' if args.fresh else ''}"):
                mlflow.set_tag("etage", "classification+decoupage")
                mlflow.log_params({"modele": model_dir.name, "contrats": len(docs), "segments": sum(len(x) for x in segs_by_doc)})
                for r in score_pipeline(docs, segs_by_doc, preds_by_doc):
                    k = f"s{int(round(r['seuil'] * 100)):02d}"
                    mlflow.log_metrics({f"rappel_{k}": r["rappel %"], f"fantomes_{k}": r["fantômes %"],
                                        f"abstention_{k}": r["abstention %"], f"autre_{k}": r["Autre %"]})
                for r in sweep:
                    k = f"a{r['règle'].replace('.', '')}_s{int(round(r['seuil'] * 100)):02d}"
                    mlflow.log_metrics({f"rappel_{k}": r["rappel %"], f"fantomes_{k}": r["fantômes %"]})
            print("Tableau enregistré dans MLflow (expérience 03-bout-en-bout).")
        except Exception as e:  # noqa: BLE001
            print(f"(MLflow non renseigné : {e})")


if __name__ == "__main__":
    main()
