"""Tests du découpage en clauses. Les textes sont des extraits (légèrement raccourcis) de vrais contrats CUAD."""

from src.data.segment import HARD_MAX, MAX_CHARS, segment
from src.data.segment_eval import annotate_segments, score_pipeline


def texts(src, **kw):
    return [s.text for s in segment(src, **kw)]


def test_decoupe_aux_lignes_vides_et_positions_exactes():
    src = ("Either party may terminate this Agreement without cause at any time upon thirty days notice.\n\n"
           "This Agreement shall be governed by the laws of the State of New York, without regard to conflicts.")
    segs = segment(src)
    assert len(segs) == 2
    for s in segs:                                    # start/end pointent bien dans le texte d'origine
        assert " ".join(src[s.start:s.end].split()) == s.text


def test_titre_rattache_a_sa_clause():
    src = ("LIMITATION OF LIABILITY\n\n"
           "i-on will not be liable under any circumstances for any lost profits or other consequential damages.")
    assert texts(src) == ["LIMITATION OF LIABILITY i-on will not be liable under any circumstances for any "
                          "lost profits or other consequential damages."]


def test_numero_de_page_et_pied_de_page_ecartes():
    src = ("The Customer shall pay all amounts due to the Company within thirty days of the invoice date.\n\n"
           "5\n\nSource: LOHA CO. LTD., F-1, 12/9/2019\n\n\n\n\n\n"
           "21. Law application \n\n"
           "It will be governed by the law of the People's Republic of China.")
    out = texts(src)
    assert not any("Source:" in t for t in out)
    assert "21. Law application It will be governed by the law of the People's Republic of China." in out
    assert not any(t == "5" for t in out)


def test_phrase_coupee_par_un_saut_de_page_est_recollee():
    src = ("Either party may elect not to renew this Agreement for any reason whatsoever, provided it gives notice to renew the\n\n"
           "4 Agreement, which notice must be given not less than fifteen (15) days before the end of the term.")
    out = texts(src)
    assert len(out) == 1 and "renew the 4 Agreement, which notice" in out[0]


def test_texte_long_coupe_sans_depasser_la_fenetre_du_modele():
    phrases = [f"Section {i}. The Supplier shall deliver the goods and the Buyer shall pay the invoice within thirty days." for i in range(1, 41)]
    src = " ".join(phrases)                                     # un seul paragraphe de plus de 4 000 caractères
    segs = segment(src)
    assert len(segs) > 4
    assert all(len(s.text) <= HARD_MAX for s in segs)
    assert " ".join(s.text for s in segs) == " ".join(src.split())      # aucun texte perdu


def test_phrase_unique_tres_longue_est_coupee_sans_perte():
    src = "The Licensee shall " + ", and ".join(f"comply with obligation number {i} of the schedule" for i in range(1, 60)) + "."
    assert len(src) > 2 * HARD_MAX and "." not in src[:-1]       # une seule phrase, aucune fin de phrase intérieure
    segs = segment(src)
    assert len(segs) >= 2 and all(len(s.text) <= HARD_MAX for s in segs)
    assert " ".join(s.text for s in segs) == " ".join(src.split())


def test_phrase_entiere_gardee_jusqu_a_la_fenetre_du_modele():
    phrase = "The Supplier shall " + " ".join(["indemnify and hold harmless the Buyer"] * 22) + "."   # ~ 850 caractères
    assert MAX_CHARS < len(phrase) <= HARD_MAX
    assert len(segment(phrase)) == 1


def test_mesure_de_bout_en_bout_sur_un_contrat_jouet():
    src = ("This Agreement shall be governed by the laws of the State of New York, without regard to conflicts.\n\n"
           "The parties shall exchange invoices every month and keep ordinary business records of the work.")
    doc = {"title": "x", "text": src, "spans": [("Governing Law", 0, 100)]}
    segs = segment(src)
    assert annotate_segments(doc, segs)[0] == {"Governing Law"} and annotate_segments(doc, segs)[1] == set()
    # le classifieur annonce Governing Law (0,9) sur la clause, et Audit Rights (0,6) sur le paragraphe neutre
    preds = [("Governing Law", 0.9), ("Audit Rights", 0.6)]
    rows = {r["seuil"]: r for r in score_pipeline([doc], [segs], [preds], thresholds=(0.0, 0.7))}
    assert rows[0.0]["rappel %"] == 100.0 and rows[0.0]["fantômes %"] == 100.0     # sans seuil : un fantôme
    assert rows[0.7]["rappel %"] == 100.0 and rows[0.7]["fantômes %"] == 0.0       # avec seuil 0,7 : plus de fantôme
    assert rows[0.7]["abstention %"] == 50.0
