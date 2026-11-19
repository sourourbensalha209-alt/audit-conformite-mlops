"""Génère des contrats SYNTHÉTIQUES (en anglais, style CUAD) pour la démonstration Big Data.

    python scripts/generer_contrats_demo.py --n 40 --out data/demo_contrats

Les vrais contrats CUAD sont versionnés par DVC et pèsent 1,7 Go : pour une démonstration rapide de Kafka et de
Spark, on fabrique des contrats plausibles à partir d'une banque de clauses écrites pour le projet. Chaque clause a
une probabilité d'être présente (par exemple 45 % pour la limitation de responsabilité) : les agrégats calculés par
Spark (« clause le plus souvent manquante ») doivent donc retrouver ces tendances, ce qui permet de vérifier la chaîne.

Ce ne sont PAS des données d'évaluation du modèle : les résultats de classification du projet sont mesurés sur CUAD.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ENTREPRISES = ["NovaSoft Solutions SARL", "Medina Distribution SA", "Atlas Components Ltd", "Carthage Logistics SA",
               "Sahel Data Systems Inc.", "Blue Harbor Trading LLC", "Jasmine Pharma SARL", "Olive Tree Foods SA",
               "Northwind Analytics Corp.", "Bizerte Marine Services SA", "Kairouan Textiles SARL", "Delta Cloud Inc."]
VILLES = ["Tunis", "Sfax", "Sousse", "Bizerte", "Nabeul", "Paris", "Wilmington, Delaware", "London"]
TYPES = {
    "DISTRIBUTION AGREEMENT": ("Supplier", "Distributor", "the products listed in Exhibit A"),
    "SUPPLY AGREEMENT": ("Supplier", "Buyer", "the components described in Schedule 1"),
    "SOFTWARE LICENSE AGREEMENT": ("Licensor", "Licensee", "the software described in Exhibit A"),
    "SERVICES AGREEMENT": ("Provider", "Client", "the services described in the Statement of Work"),
    "RESELLER AGREEMENT": ("Vendor", "Reseller", "the products listed in Annex 1"),
}
MOIS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
        "November", "December"]

# (titre de clause, probabilité de présence, variantes du texte). {A} et {B} : rôles des parties ; {objet} : l'objet.
CLAUSES = [
    ("TERM", 0.95, [
        "This Agreement shall commence on the Effective Date and shall continue for an initial term of {ans} years, "
        "expiring on {echeance}.",
        "The initial term of this Agreement is {ans} years from the Effective Date and ends on {echeance}."]),
    ("RENEWAL", 0.50, [
        "Upon expiration of the initial term, this Agreement shall automatically renew for successive one-year periods "
        "unless either party gives written notice of non-renewal at least ninety (90) days before the end of the term.",
        "This Agreement will renew automatically for additional terms of twelve (12) months unless terminated by either "
        "party with sixty (60) days prior written notice."]),
    ("TERMINATION FOR CONVENIENCE", 0.60, [
        "Either party may terminate this Agreement at any time, without cause, upon six (6) months prior written notice "
        "to the other party.",
        "{B} may terminate this Agreement for convenience by giving {A} not less than ninety (90) days written notice."]),
    ("EXCLUSIVITY", 0.40, [
        "{A} appoints {B} as its exclusive partner for {objet} in the Territory and shall not appoint any other "
        "distributor or reseller in the Territory during the term of this Agreement.",
        "During the term, {A} shall not sell {objet} to any third party located in the Territory other than {B}."]),
    ("NON-COMPETE", 0.30, [
        "During the term of this Agreement and for twelve (12) months thereafter, {B} shall not manufacture, distribute "
        "or promote any product that competes with {objet}.",
        "{B} agrees not to engage, directly or indirectly, in any business that competes with {A} in the Territory for "
        "the duration of this Agreement."]),
    ("MINIMUM PURCHASE", 0.30, [
        "{B} shall purchase {objet} having an aggregate value of not less than EUR {montant} during each contract year.",
        "The minimum annual commitment of {B} under this Agreement is EUR {montant}."]),
    ("LICENSE GRANT", 0.35, [
        "{A} grants {B} a non-exclusive, non-transferable license to use its trademarks in the Territory solely to "
        "market {objet}.",
        "Subject to the terms of this Agreement, {A} hereby grants to {B} a limited license to use, reproduce and "
        "display {objet} for its internal business purposes."]),
    ("INTELLECTUAL PROPERTY", 0.40, [
        "All intellectual property rights in any improvements, modifications or derivative works developed by {B} "
        "shall be assigned to and owned exclusively by {A}.",
        "{B} hereby assigns to {A} all right, title and interest in any work product created under this Agreement."]),
    ("PERSONAL DATA", 0.50, [
        "Each party shall process personal data of customers only for the purposes of this Agreement and in accordance "
        "with applicable data protection laws, including Regulation (EU) 2016/679, and shall notify the other party of "
        "any personal data breach without undue delay.",
        "{B} shall implement appropriate technical and organisational security measures to protect personal data and "
        "shall process such personal data only on the documented instructions of {A}."]),
    ("AUDIT RIGHTS", 0.50, [
        "{A} may, upon fifteen (15) days prior written notice and no more than once per year, inspect and audit the "
        "books and records of {B} relating to this Agreement.",
        "{B} shall keep accurate records and allow {A} or its independent auditors to audit such records during normal "
        "business hours."]),
    ("INSURANCE", 0.50, [
        "{B} shall maintain, at its own expense, general liability insurance with a coverage of not less than "
        "EUR {montant} per occurrence and shall provide a certificate of insurance upon request.",
        "Each party shall maintain adequate insurance coverage with reputable insurers for its activities under this "
        "Agreement."]),
    ("LIMITATION OF LIABILITY", 0.45, [
        "In no event shall either party's aggregate liability under this Agreement exceed the total amounts paid by "
        "{B} to {A} during the twelve (12) months preceding the claim.",
        "Except for breach of confidentiality, neither party shall be liable for any indirect or consequential damages, "
        "and each party's total liability shall not exceed EUR {montant}."]),
    ("CONFIDENTIALITY", 0.90, [
        "Each party shall keep the Confidential Information of the other party strictly confidential and shall not "
        "disclose it to any third party without prior written consent.",
        "The receiving party shall use the disclosing party's Confidential Information solely for the purposes of this "
        "Agreement and protect it with at least reasonable care."]),
    ("ASSIGNMENT", 0.80, [
        "Neither party may assign or transfer this Agreement without the prior written consent of the other party.",
        "{B} shall not assign any of its rights or obligations under this Agreement without the prior written consent "
        "of {A}, which shall not be unreasonably withheld."]),
    ("GOVERNING LAW", 0.90, [
        "This Agreement shall be governed by and construed in accordance with the laws of {droit}. Any dispute shall be "
        "submitted to the exclusive jurisdiction of the courts of {ville}.",
        "The laws of {droit} govern this Agreement, without regard to its conflict of laws principles."]),
    ("NOTICES", 1.00, [
        "All notices under this Agreement shall be in writing and delivered by hand, registered mail or email to the "
        "addresses set out above."]),
    ("ENTIRE AGREEMENT", 1.00, [
        "This Agreement constitutes the entire agreement between the parties and supersedes all prior negotiations, "
        "representations and agreements relating to its subject matter."]),
]


def contrat(rng: random.Random, i: int) -> tuple[str, str, list[str]]:
    titre, (role_a, role_b, objet) = rng.choice(list(TYPES.items()))
    a, b = rng.sample(ENTREPRISES, 2)
    jour, mois, an = rng.randint(1, 28), rng.choice(MOIS), rng.randint(2019, 2025)
    ans = rng.randint(1, 5)
    vars_ = {"A": role_a, "B": role_b, "objet": objet, "ans": ans, "echeance": f"{mois} {jour}, {an + ans}",
             "montant": f"{rng.randint(5, 90) * 10_000:,}", "ville": rng.choice(VILLES),
             "droit": rng.choice(["Tunisia", "France", "the State of New York", "England and Wales"])}
    paras = [titre,
             f"This {titre.title()} (the \"Agreement\") is entered into on {mois} {jour}, {an} (the \"Effective Date\") "
             f"by and between {a}, having its registered office in {rng.choice(VILLES)} (\"{role_a}\"), and {b}, having "
             f"its registered office in {rng.choice(VILLES)} (\"{role_b}\").",
             f"WHEREAS {role_a} wishes to provide {objet} and {role_b} wishes to obtain them, the parties agree as follows:"]
    presentes, n = [], 0
    for nom, proba, variantes in CLAUSES:
        if rng.random() < proba:
            n += 1
            paras.append(f"{n}. {nom}")
            paras.append(rng.choice(variantes).format(**vars_))
            presentes.append(nom)
    paras.append("IN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date.")
    nom_fichier = f"demo_{i:03d}_{titre.split()[0].lower()}.txt"
    return nom_fichier, "\n\n".join(paras) + "\n", presentes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="data/demo_contrats")
    ap.add_argument("--graine", type=int, default=2026, help="même graine = mêmes contrats (démonstration reproductible)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.graine)
    verite = {}
    for i in range(1, args.n + 1):
        nom, texte, presentes = contrat(rng, i)
        (out / nom).write_text(texte, encoding="utf-8")
        verite[nom] = presentes
    (out / "_verite.json").write_text(json.dumps(verite, indent=2), encoding="utf-8")
    absences = {c[0]: sum(c[0] not in p for p in verite.values()) for c in CLAUSES}
    print(f"{args.n} contrats synthétiques écrits dans {out}/ (la vérité terrain est dans _verite.json)")
    print("Clauses le plus souvent absentes :", ", ".join(f"{k} {v}/{args.n}" for k, v in
                                                          sorted(absences.items(), key=lambda x: -x[1])[:4]))


if __name__ == "__main__":
    main()
