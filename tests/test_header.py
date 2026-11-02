"""Tests de l'extracteur d'identité et des exemples du classifieur. Textes inspirés de vrais contrats CUAD."""

from src.data.autre import neutral_segments, positive_segments
from src.data.header import detect_header, same_value, score_header

CONTRAT = ("EXHIBIT B COOPERATION AGREEMENT This Agreement dated March 5, 2019 (the \"Effective Date\") is made by and between "
           "ACME Holdings, Inc., a Delaware corporation, and Beta Partners LLC. The parties agree as follows.\n\n"
           "1. TERM. This Agreement shall continue for five years.\n\n"
           "IN WITNESS WHEREOF, the parties have signed.\n\nBy: /s/ Jane Roe Name: Jane Roe Title: CEO ACME HOLDINGS, INC.")


def test_extraction_des_quatre_informations_d_identite():
    f = detect_header(CONTRAT)
    assert f["Document Name"] and "COOPERATION AGREEMENT" in f["Document Name"].text
    assert f["Parties"] and "ACME Holdings" in f["Parties"].text
    assert f["Agreement Date"] and f["Agreement Date"].text == "March 5, 2019"
    assert f["Effective Date"] and f["Effective Date"].text == "March 5, 2019"


def test_aucune_information_inventee_sur_un_texte_banal():
    f = detect_header("The parties shall keep ordinary business records of the work performed under this paragraph.")
    assert f["Agreement Date"] is None and f["Effective Date"] is None and f["Parties"] is None


def test_variantes_de_dates():
    assert detect_header("This Agreement is entered into as of 28 February 2019 by and between A and B.")["Agreement Date"].text == "28 February 2019"
    assert detect_header("Made this 5th day of June, 2010, by and between A and B.")["Agreement Date"] is not None
    assert detect_header("This Agreement is effective as of 12/09/2019 between A and B.")["Effective Date"].text == "12/09/2019"


def test_comparaison_de_valeurs():
    assert same_value("EXHIBIT B COOPERATION AGREEMENT", ["COOPERATION AGREEMENT"])           # le texte annoté est contenu
    assert same_value("March 5, 2019", ["entered into as of March 5, 2019"])                   # et inversement
    assert not same_value("March 5, 2019", ["April 8, 2020"])
    assert not same_value("x", ["x"])                                                          # trop court pour être une preuve


def test_score_d_ensemble():
    a = CONTRAT.index("March 5, 2019")
    doc = {"title": "t", "text": CONTRAT, "spans": [("Agreement Date", a, a + 13), ("Parties", 0, 0)]}
    s = score_header([doc])
    assert s["Agreement Date"] == (1, 1) and s["Parties"] == (0, 0) and s["Effective Date"] == (0, 0)


def test_les_blocs_de_signature_sont_autre_pour_le_classifieur():
    """Un bloc annoté « Parties » ne doit plus être exclu des exemples « Autre » : l'extracteur s'en charge."""
    a = CONTRAT.index("By: /s/ Jane Roe")
    doc = {"title": "t", "text": CONTRAT, "spans": [("Parties", a, a + 40)]}
    assert any("Jane Roe" in s.text for s in neutral_segments(doc))               # désormais éligible comme « Autre »
    assert not any("Jane Roe" in s.text for s in neutral_segments(doc, ignore=frozenset()))   # l'ancien comportement l'excluait


def test_clause_sous_forme_de_segment():
    clause = "This Agreement shall be governed by and construed in accordance with the laws of the State of New York."
    text = f"Preamble text that is long enough to be kept as a separate paragraph of the contract.\n\n9. GOVERNING LAW\n\n{clause} Disputes shall be resolved in New York courts."
    a = text.index(clause)
    doc = {"title": "d", "text": text, "spans": [("Governing Law", a, a + len(clause)), ("Document Name", 0, 10)]}
    rows = positive_segments(doc)
    assert [r["label"] for r in rows] == ["Governing Law"]                       # la catégorie d'identité est écartée
    assert clause in rows[0]["text"] and "GOVERNING LAW" in rows[0]["text"]       # le segment contient la clause ET son titre
    assert len(rows[0]["text"]) > len(clause)                                     # c'est bien la forme « segment », plus longue


def test_choix_des_contrats_repartis_et_exclusion():
    from src.data.segment_eval import pick_contracts

    docs = [{"title": f"c{i}", "text": "x" * (1000 * (i + 1)), "spans": []} for i in range(40)]
    premiers = pick_contracts(docs, 8)
    assert len(premiers) == 8 and premiers[0]["title"] == "c0" and premiers[-1]["title"] == "c39"   # du plus court au plus long
    nouveaux = pick_contracts(docs, 20, exclude_titles={d["title"] for d in premiers})
    assert len(nouveaux) == 20
    assert not ({d["title"] for d in nouveaux} & {d["title"] for d in premiers})                   # aucun contrat déjà vu
    assert pick_contracts(docs, 8) == premiers                                                      # choix reproductible
