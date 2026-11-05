"""Tests du rapport d'audit : ce qu'il affirme, ce qu'il refuse d'affirmer, et sa sécurité d'affichage."""

from pathlib import Path

import pytest

from src.judge.report import RELIABLE_MISSING, _display, analyse, build_meta, render_html
from src.retrieval.rules import load_rules

ROOT = Path(__file__).resolve().parents[1]
TEXT = ("This Agreement is made on March 5, 2019 by and between ACME Inc. and Beta LLC for the supply of goods.\n\n"
        "This Agreement shall be governed by the laws of the State of New York, without regard to conflicts of law principles.\n\n"
        "Each party shall keep the <script>alert(1)</script> information of the other confidential at all times and for all purposes.")
LABELS = ["Audit Rights", "Autre", "Cap On Liability", "Governing Law"]


@pytest.fixture(scope="module")
def book():
    path = ROOT / "data" / "rules" / "rules_contractuelles.json"
    if not path.exists():
        pytest.skip("lancer d'abord : python scripts/build_rules.py")
    return load_rules(path)


def _probs(*top):
    """Une probabilité 0,9 sur l'étiquette indiquée pour chaque segment."""
    return [[0.9 if l == t else 0.1 / (len(LABELS) - 1) for l in LABELS] for t in top]


def test_ce_que_le_rapport_affirme_et_ce_qu_il_refuse(book):
    a = analyse(TEXT, LABELS, _probs("Autre", "Governing Law", "Audit Rights"), book)
    statut = {x["categorie"]: x["statut"] for x in a["attendues"]}
    assert statut["Governing Law"] == "presente"
    assert statut["Cap On Liability"] == "manquante"            # catégorie fiable : « manquante » est affirmé
    assert statut["Expiration Date"] == "a_verifier"             # classifieur peu fiable sur cette catégorie : on n'affirme pas
    assert statut["Effective Date"] == "a_verifier"              # extracteur : « non trouvé » est souvent faux
    assert statut["Agreement Date"] == "presente"
    assert RELIABLE_MISSING == {"Cap On Liability", "Governing Law"}
    assert a["attendues"][0]["statut"] == "manquante"            # les manquantes d'abord
    assert [c["categorie"] for c in a["clauses"]] == ["Governing Law", "Audit Rights"]
    assert a["clauses"][0]["regle"]["id"].startswith("R") and a["stats"]["segments"] == 3


def test_le_rapport_ne_rend_aucun_verdict_automatique(book):
    page = render_html(analyse(TEXT, LABELS, _probs("Autre", "Governing Law", "Autre"), book), build_meta("Test", "modele", "abc123"))
    assert "Aperçu technique, pas un avis juridique" in page and "Manquante" in page and "À vérifier" in page
    # les mots « conforme » ne figurent que dans l'avertissement qui dit que le rapport n'en rend pas
    assert page.count("conforme") <= 2 and "Non conforme" not in page and "✔ Conforme" not in page


def test_affichage_sur_et_autonome(book):
    page = render_html(analyse(TEXT, LABELS, _probs("Autre", "Governing Law", "Audit Rights"), book), build_meta("T<b>", "m", "v"))
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page      # le texte du contrat est échappé
    assert "T<b>" not in page and "T&lt;b&gt;" in page                                # le titre aussi
    assert "http://" not in page and "https://" not in page                          # aucune ressource externe : marche hors ligne
    assert "<style>" in page and "stylesheet" not in page


def test_titre_sans_numero_d_annexe_et_incoherences(book):
    assert _display("Document Name", "4 COOPERATION AGREEMENT") == "COOPERATION AGREEMENT"
    assert _display("Parties", "4 Foo Inc.") == "4 Foo Inc."                          # seul le titre est nettoyé
    with pytest.raises(ValueError):
        analyse(TEXT, LABELS, _probs("Autre"), book)                                  # 3 paragraphes mais 1 ligne de probabilités
