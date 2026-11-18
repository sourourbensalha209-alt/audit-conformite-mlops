from src.data.paragraphs import rebuild_paragraphs
from src.data.segment import segment

# Ce que pdfplumber rend pour une page de contrat : une ligne par ligne imprimée, aucune ligne vide.
PDF_LINES = [
    "DISTRIBUTION AGREEMENT",
    "This Distribution Agreement is entered into as of January 1, 2020 between Acme Inc. and",
    "Beta Ltd. (the \"Distributor\").",
    "1. Term. This Agreement shall commence on the Effective Date and continue for a period of",
    "30 days after written notice by either party, unless terminated earlier in accordance with",
    "Section 8.",
    "2. GOVERNING LAW",
    "This Agreement shall be governed by the laws of the State of New York, without regard to its",
    "conflict of laws principles.",
    "3. Limitation of Liability. In no event shall either party's aggregate liability exceed the fees",
    "paid in the twelve months preceding the claim. Neither party shall be liable for indirect dam-",
    "ages.",
]


def test_une_clause_par_paragraphe():
    paras = rebuild_paragraphs(PDF_LINES).split("\n\n")
    assert paras[0] == "DISTRIBUTION AGREEMENT"
    assert paras[1].startswith("This Distribution Agreement") and paras[1].endswith("(the \"Distributor\").")
    assert paras[2].startswith("1. Term.") and paras[2].endswith("Section 8.")   # "30 days" ne coupe pas
    assert paras[3] == "2. GOVERNING LAW"
    assert paras[4].startswith("This Agreement shall be governed")
    assert paras[5].startswith("3. Limitation")
    assert len(paras) == 6


def test_mot_coupe_recolle():
    assert "indirect damages." in rebuild_paragraphs(PDF_LINES)


def test_segment_retrouve_les_clauses():
    segs = [s.text for s in segment(rebuild_paragraphs(PDF_LINES))]
    assert any(s.startswith("2. GOVERNING LAW This Agreement shall be governed") for s in segs)   # titre rattaché
    assert any(s.startswith("3. Limitation of Liability") for s in segs)


def test_vide():
    assert rebuild_paragraphs(["", "  "]) == ""
