"""Tests du juge : prompt, lecture des réponses, citations, mesures, cas d'évaluation, feuille d'annotation, clauses manquantes."""

import json
from pathlib import Path

import pytest

from src.judge.cases import FAMILIES, _FAMILY_OF, build_cases, is_personal_data_clause, looks_like_junk
from src.judge.metrics import CRITERIA, check_criteria, cohen_kappa, evaluate, final_labels
from src.judge.missing import score_missing
from src.judge.parse import check_citations, normalize_verdict, parse_output
from src.judge.prompt import build_messages
from src.retrieval.corpus import Article
from src.retrieval.rules import load_rules

ROOT = Path(__file__).resolve().parents[1]
CLAUSE = "Supplier's total liability under this Agreement shall not exceed the fees paid by Customer in the twelve months preceding the claim."
CASE = {"id": "J001", "family": "playbook", "category": "Cap On Liability", "clause": CLAUSE,
        "rule": {"id": "R06", "titre": "Plafonnement de la responsabilité", "regle": "La responsabilité de chaque partie est plafonnée."},
        "sources": [{"id": "R06", "citation": "Règle de contrôle R06", "text": "x"}], "allowed_sources": ["R06"], "expected": None}


@pytest.fixture(scope="module")
def book():
    path = ROOT / "data" / "rules" / "rules_contractuelles.json"
    if not path.exists():
        pytest.skip("lancer d'abord : python scripts/build_rules.py")
    return load_rules(path)


# --- prompt ---------------------------------------------------------------------------------------------------------
def test_prompt_avec_regle():
    m = build_messages(CASE, with_rule=True)
    texte = m[1]["content"]
    assert m[0]["role"] == "system" and "[R06]" in texte and "Plafonnement" in texte and CLAUSE in texte
    assert "a_verifier" in texte and "JSON" in texte and "aucune" in texte           # pas de source juridique en plus de la règle


def test_prompt_avec_articles_et_sans_regle():
    c = dict(CASE, sources=CASE["sources"] + [{"id": "rgpd:28", "citation": "RGPD, art. 28", "text": "Le sous-traitant ne traite que sur instruction."}])
    assert "[rgpd:28] RGPD, art. 28" in build_messages(c, True)[1]["content"]
    sans = build_messages(c, with_rule=False)[1]["content"]
    assert "Plafonnement" not in sans and "rgpd:28" not in sans and '"sources" doit être []' in sans and CLAUSE in sans


# --- lecture des réponses -------------------------------------------------------------------------------------------
def test_lecture_json_propre_et_variantes():
    p = parse_output('{"verdict": "non_conforme", "justification": "Pas de plafond.", "sources": ["R06"]}')
    assert p["valid"] and p["verdict"] == "non_conforme" and p["sources"] == ["R06"]
    fence = parse_output('Voici ma réponse :\n```json\n{"verdict": "Non conforme", "justification": "x", "sources": "R06"}\n```\nMerci.')
    assert fence["valid"] and fence["verdict"] == "non_conforme" and fence["sources"] == ["R06"]
    assert parse_output('{"verdict": "à vérifier", "justification": "", "sources": []}')["verdict"] == "a_verifier"


def test_reponses_invalides_ou_recuperees():
    assert not parse_output("Je ne sais pas.")["valid"]
    inconnu = parse_output('{"verdict": "peut-être", "sources": []}')
    assert not inconnu["valid"] and inconnu["verdict"] is None
    texte = parse_output('Le verdict: conforme car la clause est claire.')            # pas de JSON, verdict lisible
    assert not texte["valid"] and texte["recovered"] and texte["verdict"] == "conforme"
    assert normalize_verdict(None) is None and normalize_verdict("A vérifier") == "a_verifier"


def test_controle_des_citations():
    assert check_citations(["R06", "[rgpd:28]"], ["R06", "rgpd:28"]) == (2, 0)
    assert check_citations(["R06", "rgpd:99"], ["R06"]) == (2, 1)                     # rgpd:99 n'était pas dans le prompt
    assert check_citations(["R06"], []) == (1, 1)                                      # condition sans règle : toute citation est inventée
    assert check_citations([], ["R06"]) == (0, 0)


# --- mesures --------------------------------------------------------------------------------------------------------
def _case(cid, family, expected=None, allowed=("R06",)):
    return {"id": cid, "family": family, "category": None, "clause": "x", "allowed_sources": list(allowed), "expected": expected}


def _raw(verdict, sources=()):
    return json.dumps({"verdict": verdict, "justification": "j", "sources": list(sources)})


def test_mesures_du_juge():
    cases = {c["id"]: c for c in [_case("a", "playbook"), _case("b", "playbook"), _case("c", "playbook"), _case("d", "mismatch", "a_verifier"),
                                  _case("e", "neutral", "a_verifier")]}
    labels = {"a": "conforme", "b": "non_conforme", "c": "conforme"}
    res = [{"case_id": "a", "model": "m", "condition": "with_rule", "raw": _raw("conforme", ["R06"])},
           {"case_id": "b", "model": "m", "condition": "with_rule", "raw": _raw("conforme", ["R99"])},          # faux verdict + source inventée
           {"case_id": "c", "model": "m", "condition": "with_rule", "raw": _raw("a_verifier")},                 # fausse abstention
           {"case_id": "d", "model": "m", "condition": "with_rule", "raw": _raw("a_verifier")},
           {"case_id": "e", "model": "m", "condition": "with_rule", "raw": "n'importe quoi"}]
    b = evaluate(cases, res, labels)["m|with_rule"]
    assert b["json_valide_pct"] == 80.0 and b["sources_inventees_pct"] == 20.0
    assert b["accord_equipe_pct"] == 33.3 and b["reponse_majoritaire_pct"] == 66.7        # 1 accord sur 3 ; « conforme » domine à 2/3
    assert b["abstention_hors_sujet_pct"] == 50.0 and b["fausse_abstention_pct"] == 33.3        # 1 fausse abstention sur 3 cas tranchés
    noms = {n: ok for n, _, _, ok in check_criteria(b)}
    assert not noms["JSON valide"] and not noms["Sources inventées"] and not noms["Accord avec l'équipe"]


def test_sans_regle_toute_source_est_inventee():
    cases = {"a": _case("a", "playbook")}
    res = [{"case_id": "a", "model": "m", "condition": "without_rule", "raw": _raw("conforme", ["R06"])}]
    b = evaluate(cases, res, {"a": "conforme"})["m|without_rule"]
    assert b["sources_inventees_pct"] == 100.0 and b["abstention_hors_sujet_pct"] is None


def test_accord_entre_annotateurs():
    assert cohen_kappa(["a", "a", "b", "b", "a"], ["a", "a", "b", "b", "a"]) == 1.0
    assert cohen_kappa(["a"] * 4, ["a"] * 4) is None                                  # trop peu de cas
    rows = [{"id": "1", "verdict_sourour": "conforme", "verdict_ghayth": "conforme"},
            {"id": "2", "verdict_sourour": "conforme", "verdict_ghayth": "non_conforme"},      # désaccord : écarté
            {"id": "3", "verdict_sourour": "", "verdict_ghayth": "a_verifier"},               # un seul annotateur : retenu
            {"id": "4", "verdict_sourour": "n'importe quoi", "verdict_ghayth": ""}]            # verdict invalide : ignoré
    final, stats = final_labels(rows)
    assert final == {"1": "conforme", "3": "a_verifier"} and stats["desaccords"] == ["2"] and stats["cas_annotes_par_les_deux"] == 2


# --- fabrication des cas --------------------------------------------------------------------------------------------
def _docs(n_docs=8):
    cats = ["Cap On Liability", "Non-Compete", "License Grant", "Governing Law", "Audit Rights", "Insurance", "Exclusivity", "Renewal Term"]
    docs = []
    for i in range(n_docs):
        paras, spans, pos = [], [], 0
        for j, cat in enumerate(cats):
            clause = f"Clause {cat} numéro {i}-{j}: the parties agree that this provision applies to all activities under the agreement and survives termination."
            neutral = " ".join(f"Neutral filler paragraph {i}-{j} about notices, headings and administration of the agreement." for _ in range(3))
            far = [" ".join(f"Far neutral paragraph {i}-{j}-{k} about administration, notices and headings of the agreement." for _ in range(3))
                   for k in range(3)]                                  # 3 paragraphes neutres ; ceux du milieu sont à plus de 150 car. d'une clause
            paras += [neutral, clause] + far
            pos += len(neutral) + 2
            spans.append((cat, pos, pos + len(clause)))
            pos += len(clause) + 2 + sum(len(f) + 2 for f in far)
        docs.append({"title": f"contrat{i}", "text": "\n\n".join(paras), "spans": spans})
    return docs


def test_fabrication_des_cas(book):
    articles = {"rgpd:28": Article("rgpd:28", "rgpd", "RGPD", "28", "Sous-traitant ...")}
    cases, stats = build_cases(_docs(), book, articles, n_matched=12, n_rgpd=3, n_mismatch=5, n_neutral=5)
    fam = {f: [c for c in cases if c["family"] == f] for f in ("playbook", "rgpd", "mismatch", "neutral")}
    assert len(fam["playbook"]) == 12 and len(fam["mismatch"]) == 5 and len(fam["neutral"]) == 5
    assert len({c["id"] for c in cases}) == len(cases)
    assert all(c["expected"] == "a_verifier" for c in fam["mismatch"] + fam["neutral"])
    assert all(c["expected"] is None and c["a_annoter"] for c in fam["playbook"])
    assert len({c["category"] for c in fam["playbook"]}) >= 6                          # réparti sur les catégories
    # une paire « sans rapport » n'associe jamais une clause à une règle de sa propre famille
    by_id = {r["id"]: r["category"] for r in book.rules}
    for c in fam["mismatch"]:
        assert _FAMILY_OF.get(c["category"]) != _FAMILY_OF.get(by_id[c["rule"]["id"]]) or _FAMILY_OF.get(c["category"]) is None
    # le tirage est reproductible
    again, _ = build_cases(_docs(), book, articles, n_matched=12, n_rgpd=3, n_mismatch=5, n_neutral=5)
    assert [c["clause"] for c in again] == [c["clause"] for c in cases]


def test_clauses_de_donnees_personnelles_et_bruit():
    vraie = "Each Party shall protect Personal Data under applicable Data Protection laws and shall not disclose Personal Data to third parties."
    assert is_personal_data_clause(vraie)
    assert not is_personal_data_clause("The Contractor shall comply with all laws, including data protection law, in its operations here.")   # une seule mention
    assert not looks_like_junk(vraie) and looks_like_junk("CONTENTS Clause Page 1. Interpretation 3 2. Appointment 4 3. Duties 4")
    assert not is_personal_data_clause("Attention: [Redacted - Personal Information], Manager. Email: [Redacted - Personal Information] Personal Data")


# --- feuille Excel --------------------------------------------------------------------------------------------------
def test_feuille_d_annotation(tmp_path, book):
    openpyxl = pytest.importorskip("openpyxl")
    from src.judge.sheet import assign, read_sheet, write_sheet

    cases, _ = build_cases(_docs(), book, {}, n_matched=14, n_rgpd=0, n_mismatch=3, n_neutral=3)
    who = write_sheet(tmp_path / "a_remplir.xlsx", cases)
    todo = [c for c in cases if c["a_annoter"]]
    assert len(who) == len(todo) == 14 and sum(v == "Les deux" for v in who.values()) >= 1
    assert assign(cases) == who
    wb = openpyxl.load_workbook(tmp_path / "a_remplir.xlsx")
    ws = wb["Cas"]
    assert wb.sheetnames == ["Consignes", "Cas"] and ws.max_row == len(todo) + 1
    assert any("conforme" in dv.formula1 for dv in ws.data_validations.dataValidation)
    ws["H2"] = "conforme"; ws["I3"] = "a_verifier"
    wb.save(tmp_path / "rempli.xlsx")
    rows = read_sheet(tmp_path / "rempli.xlsx")
    assert rows[0]["verdict_sourour"] == "conforme" and rows[1]["verdict_ghayth"] == "a_verifier" and rows[2]["verdict_sourour"] == ""


# --- clauses manquantes ---------------------------------------------------------------------------------------------
def test_verdict_manquante():
    docs = [{"spans": [("Governing Law", 0, 10)]}, {"spans": []}, {"spans": [("Governing Law", 0, 10)]}, {"spans": []}]
    present = [{"Governing Law"}, {"Governing Law"}, set(), set()]     # 2 bien vus, 1 à tort, 1 ratée
    r = score_missing(docs, present, ["Governing Law"])
    # absents en vrai : docs 2 et 4 ; dit « manquante » : docs 3 et 4 -> bien vu 1, à tort 1, raté 1
    assert r["total"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1} and r["precision_manquante_pct"] == 50.0 and r["rappel_manquante_pct"] == 50.0
    assert r["taux_de_base_pct"] == 50.0 and r["critere_atteint"] is False
