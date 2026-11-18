"""Boucle de réentraînement : collecte des corrections, décision, paquet, promotion champion / challenger."""

import json
import zipfile

import pytest

from src.orchestration import promote as P
from src.orchestration import registry as R
from src.orchestration import retrain as T

LABELS = ["Autre", "Cap On Liability", "Governing Law"]
PASSAGE = "In no event shall either party's aggregate liability exceed the fees paid in the preceding twelve months."


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# ---------- collecte ----------

def test_collecte_garde_seulement_les_corrections_exploitables(tmp_path):
    fb = tmp_path / "corrections.jsonl"
    _write(fb, [
        {"job_id": "a", "texte": PASSAGE, "categorie_corrigee": "Cap On Liability"},
        {"job_id": "b", "texte": "  " + PASSAGE.upper() + " ", "categorie_corrigee": "Cap On Liability"},   # doublon
        {"job_id": "c", "verdict_corrige": "manquante"},                                                     # pas de passage
        {"job_id": "d", "texte": PASSAGE, "categorie_corrigee": "Catégorie inventée"},                       # inconnue
    ])
    exemples, ecartes = T.collect(fb, LABELS)
    assert [e["label"] for e in exemples] == ["Cap On Liability"]
    assert ecartes == {"sans_passage": 1, "categorie_inconnue": 1, "doublon": 1}


# ---------- décision ----------

def test_decision_seuil_d_exemples():
    assert T.decide(19, [], seuil=20)["reentrainer"] is False
    assert T.decide(20, [], seuil=20)["reentrainer"] is True


def test_derive_sans_exemple_ne_suffit_pas():
    d = T.decide(0, [{"niveau": "derive", "message": "Part Autre 98 %"}], seuil=20)
    assert d["reentrainer"] is False and "conseil" in d


def test_derive_avec_exemples_declenche():
    d = T.decide(3, [{"niveau": "qualite", "message": "40 % des audits corrigés"}], seuil=20)
    assert d["reentrainer"] is True and any("Monitoring" in r for r in d["raisons"])


def test_alerte_de_service_ne_declenche_pas():
    assert T.decide(3, [{"niveau": "service", "message": "lent"}], seuil=20)["reentrainer"] is False


# ---------- exécution complète ----------

@pytest.fixture
def modele(tmp_path, monkeypatch):
    """Faux modèle en production (labels.json + metrics.json), désigné par AUDIT_MODEL."""
    models = tmp_path / "models"
    v4 = models / "best_model_v4"
    v4.mkdir(parents=True)
    (v4 / "labels.json").write_text(json.dumps(LABELS))
    (v4 / "metrics.json").write_text(json.dumps({"f1_macro": 0.719, "f1_macro_rare": 0.667, "n_test": 4788}))
    (v4 / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(R, "MODELS", models)
    monkeypatch.setattr(P, "MODELS", models)
    monkeypatch.delenv("AUDIT_MODEL", raising=False)
    return models


def test_run_prepare_le_paquet_et_ne_reutilise_pas_les_exemples(tmp_path, modele, monkeypatch):
    monkeypatch.setattr(T, "_log_mlflow", lambda *a: None)          # pas d'écriture MLflow pendant les tests
    fb, work = tmp_path / "corrections.jsonl", tmp_path / "work"
    _write(fb, [{"job_id": str(i), "texte": f"{PASSAGE} (variante {i})", "categorie_corrigee": "Cap On Liability"}
                for i in range(3)])
    log = tmp_path / "audits.jsonl"
    e1 = T.run(seuil=3, feedback_path=fb, work_dir=work, log_path=log)
    assert e1["reentrainer"] is True and e1["exemples_nouveaux"] == 3
    with zipfile.ZipFile(work / e1["paquet"]) as z:
        assert {"exemples_juristes.jsonl", "labels.json", "LISEZ-MOI.md", "decision.json"} <= set(z.namelist())
    e2 = T.run(seuil=3, feedback_path=fb, work_dir=work, log_path=log)        # rien de nouveau depuis
    assert e2["reentrainer"] is False and e2["exemples_nouveaux"] == 0
    assert T.state(work)["reentrainer"] is False
    assert len((work / "historique.jsonl").read_text().splitlines()) == 2


# ---------- registre et promotion ----------

def _candidat(models, nom, f1, rare, n_test=4788, labels=LABELS):
    d = models / nom
    d.mkdir()
    (d / "labels.json").write_text(json.dumps(labels))
    (d / "metrics.json").write_text(json.dumps([{"f1_macro": f1, "f1_macro_rare": rare, "n_test": n_test}]))
    (d / "model.safetensors").write_bytes(b"x")
    return d


def test_registre_par_defaut_puis_variable_d_environnement(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIT_MODEL", raising=False)
    assert R.production_model_name(tmp_path / "absent.json") == "best_model_v4"
    monkeypatch.setenv("AUDIT_MODEL", "best_model_v9")
    assert R.production_model_name(tmp_path / "absent.json") == "best_model_v9"


def test_promotion_acceptee(modele):
    prod = modele / "production.json"
    r = P.promote(_candidat(modele, "best_model_v5", 0.735, 0.670), prod)
    assert r["accepte"] is True
    assert R.production_model_name(prod) == "best_model_v5"
    assert len((modele / "promotions.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("f1, rare, n_test, labels, motif", [
    (0.700, 0.670, 4788, LABELS, "pas meilleur"),
    (0.730, 0.600, 4788, LABELS, "catégories rares"),
    (0.730, 0.670, 1000, LABELS, "Jeu de test différent"),
    (0.730, 0.670, 4788, LABELS[::-1], "Étiquettes différentes"),
])
def test_promotion_refusee(modele, f1, rare, n_test, labels, motif):
    prod = modele / "production.json"
    r = P.promote(_candidat(modele, "best_model_v5", f1, rare, n_test, labels), prod)
    assert r["accepte"] is False and any(motif in x for x in r["raisons"])
    assert not prod.exists()                                   # le modèle en production n'a pas changé


def test_promotion_dossier_incomplet(modele):
    d = modele / "best_model_v5"
    d.mkdir()
    r = P.promote(d, modele / "production.json")
    assert r["accepte"] is False and "manquants" in r["raisons"][0]
