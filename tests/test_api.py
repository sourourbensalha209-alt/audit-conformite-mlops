"""Tests de l'API sans charger le modèle : la chaîne d'audit est remplacée par une fausse."""

import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import src.api.main as api  # noqa: E402
import src.pipeline as pipeline  # noqa: E402

FAUX_RESULTAT = {
    "identite": [],
    "attendues": [{"categorie": "Cap On Liability", "titre": "Plafond", "regle": "Plafonner.", "gravite": "high",
                   "statut": "manquante", "preuve": None}],
    "clauses": [],
    "stats": {"segments": 3, "clauses": 0, "categories": 0, "rgpd": 0},
}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "audit_file", lambda path, title: (FAUX_RESULTAT, "<html>rapport</html>"))
    monkeypatch.setattr(api, "FEEDBACK_FILE", tmp_path / "corrections.jsonl")
    monkeypatch.setattr(api.monitoring, "LOG_FILE", tmp_path / "audits.jsonl")
    api.JOBS.clear()
    api.REPORTS.clear()
    return TestClient(api.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_format_refuse(client):
    r = client.post("/audit", files={"file": ("contrat.exe", b"x")})
    assert r.status_code == 400


def test_audit_complet(client):
    r = client.post("/audit", files={"file": ("contrat.txt", b"1. Term. This Agreement ...")})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    job = client.get(f"/audit/{job_id}").json()      # TestClient exécute la tâche de fond avant de rendre la main
    assert job["status"] == "termine"
    assert job["findings"][0]["verdict"] == "manquante"
    assert job["resultat"]["stats"]["segments"] == 3

    assert "rapport" in client.get(f"/audit/{job_id}/rapport").text
    assert client.get("/audits").json()[0]["job_id"] == job_id


def test_rapport_inconnu(client):
    assert client.get("/audit/inconnu/rapport").status_code == 404


def test_feedback_enregistre(client):
    job_id = client.post("/audit", files={"file": ("c.txt", b"texte")}).json()["job_id"]
    r = client.post("/feedback", json={"job_id": job_id, "clause_ref": "Cap On Liability", "verdict_corrige": "presente"})
    assert r.status_code == 201
    ligne = json.loads(api.FEEDBACK_FILE.read_text(encoding="utf-8").splitlines()[0])
    assert ligne["verdict_corrige"] == "presente"


def test_interface_servie(client):
    if not api.FRONTEND_DIR.exists():
        pytest.skip("dossier frontend absent")
    assert "Agent d'audit" in client.get("/app/").text


def test_monitoring_apres_un_audit(client):
    client.post("/audit", files={"file": ("c.txt", b"texte")})
    m = client.get("/monitoring").json()
    assert m["indicateurs"]["audits"] == 1 and m["indicateurs"]["termines"] == 1
    metrics = client.get("/metrics").text
    assert 'audit_total{statut="termine"} 1' in metrics
