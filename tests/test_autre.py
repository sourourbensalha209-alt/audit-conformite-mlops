"""Tests de la classe « Autre » : sélection des paragraphes neutres et prise en compte dans la mesure."""

from src.data.autre import MARGIN, NEUTRAL_LABEL, neutral_segments, sample_autre
from src.data.segment import segment
from src.data.segment_eval import score_pipeline

CLAUSE = "This Agreement shall be governed by the laws of the State of New York, without regard to conflicts."
NEUTRE_1 = "The Administrator shall be the agent of the Fund for receipt of requests for redemption from each account."
NEUTRE_2 = "Capitalized terms used but not defined in this section have the meanings given to them elsewhere in the agreement."


def fake_doc(n_neutres=6):
    filler = [f"Paragraph {i}: the parties shall keep ordinary business records of the work performed under item {i}." for i in range(n_neutres)]
    text = "\n\n".join([NEUTRE_1] + filler * 1 + [CLAUSE] + [NEUTRE_2])
    a = text.index(CLAUSE)
    return {"title": "doc", "text": text, "spans": [("Governing Law", a, a + len(CLAUSE))]}


def test_une_clause_annotee_n_est_jamais_un_exemple_autre():
    doc = fake_doc()
    textes = [s.text for s in neutral_segments(doc)]
    assert not any("governed by the laws" in t for t in textes)
    assert any("Administrator" in t for t in textes)


def test_le_voisinage_immediat_d_une_clause_est_ecarte():
    avant = "Notice period for the parties under this heading applies to all notices given hereunder in writing."
    text = f"{NEUTRE_1}\n\n{avant}\n\n{CLAUSE}"
    a = text.index(CLAUSE)
    doc = {"title": "d", "text": text, "spans": [("Governing Law", a, a + len(CLAUSE))]}
    assert len(avant) < MARGIN + 10                                  # le paragraphe est collé à la clause
    textes = [s.text for s in neutral_segments(doc)]
    assert avant not in textes                                       # trop proche : écarté (peut être le début de la clause)


def test_le_tirage_est_reproductible_et_repartit_sur_les_contrats():
    docs = [dict(fake_doc(8), title=f"contrat{i}") for i in range(10)]
    r1, st = sample_autre(docs, 20, seed=1)
    r2, _ = sample_autre(docs, 20, seed=1)
    assert [r["text"] for r in r1] == [r["text"] for r in r2]
    assert all(r["label"] == NEUTRAL_LABEL for r in r1)
    assert len({r["source"] for r in r1}) >= 5                       # plusieurs contrats, pas un seul qui domine
    assert st["plafond par contrat"] >= 3


def test_la_mesure_distingue_autre_d_un_fantome():
    doc = fake_doc(0)
    segs = segment(doc["text"])
    # segments : NEUTRE_1, CLAUSE, NEUTRE_2
    avec_fantome = [("Audit Rights", 0.9), ("Governing Law", 0.9), ("Audit Rights", 0.9)]
    avec_autre = [(NEUTRAL_LABEL, 0.99), ("Governing Law", 0.9), (NEUTRAL_LABEL, 0.99)]
    r1 = score_pipeline([doc], [segs], [avec_fantome], thresholds=(0.5,))[0]
    r2 = score_pipeline([doc], [segs], [avec_autre], thresholds=(0.5,))[0]
    assert r1["fantômes %"] == 100.0 and r1["Autre %"] == 0.0
    assert r2["fantômes %"] == 0.0 and r2["Autre %"] == round(100 * 2 / 3, 1)
    assert r1["rappel %"] == r2["rappel %"] == 100.0                 # la bonne clause est retrouvée dans les deux cas
