"""Data lake (zone bronze) et traitement d'un message Kafka, sans Kafka ni Spark : bibliothèque standard seulement."""

import base64
import json

import pytest

from src.bigdata import lake
from src.bigdata.stream import encode_document, process_message

RESULTAT = {
    "monitoring": {"paragraphes": 17, "confiance_moyenne": 0.9, "part_abstention": 0.06, "part_autre": 0.41},
    "stats": {"segments": 17, "rgpd": 1},
    "attendues": [{"categorie": "Cap On Liability", "titre": "Plafond", "statut": "manquante", "preuve": None},
                  {"categorie": "Governing Law", "titre": "Droit", "statut": "presente", "preuve": {"texte": "x"}}],
    "clauses": [{"categorie": "Governing Law", "confiance": 0.99, "texte": "TEXTE CONFIDENTIEL"},
                {"categorie": "Données personnelles", "confiance": None, "texte": "TEXTE CONFIDENTIEL"}],
}


def test_evenement_d_audit_compact_et_sans_texte_du_contrat():
    e = lake.audit_event("j1", "c.pdf", RESULTAT, 7.1, "best_model_v4", source="api")
    assert e["paragraphes"] == 17 and e["part_autre"] == 0.41 and e["rgpd"] == 1
    assert e["attendues"] == [{"categorie": "Cap On Liability", "statut": "manquante"},
                              {"categorie": "Governing Law", "statut": "presente"}]
    assert "TEXTE CONFIDENTIEL" not in json.dumps(e)            # le data lake ne stocke pas le contenu des contrats


def test_zone_bronze_partitionnee_par_jour(tmp_path):
    e = lake.audit_event("j1", "c.pdf", RESULTAT, 7.1, "best_model_v4")
    path = lake.write_bronze("audits", e, tmp_path)
    assert path.parent.name == f"jour={e['date'][:10]}" and path.name == "j1.json"
    assert json.loads(path.read_text(encoding="utf-8"))["job_id"] == "j1"
    assert not list(tmp_path.rglob("*.tmp"))                    # écriture atomique : pas de fichier temporaire laissé


def test_evenement_de_correction():
    e = lake.correction_event({"job_id": "j1", "clause_ref": "§3", "texte": "passage", "categorie_corrigee": "Cap On Liability"})
    assert e["avec_passage"] is True and "texte" not in e


def test_message_kafka_audite_et_verse_dans_le_lake(tmp_path):
    doc = tmp_path / "contrat.txt"
    doc.write_text("1. Term. This Agreement ...", encoding="utf-8")
    vus = []

    def faux_audit(path, title):
        vus.append((path.name, path.read_text(encoding="utf-8")))
        return RESULTAT, "<html/>"

    e = process_message(encode_document(doc), faux_audit, "best_model_v4", tmp_path / "lake")
    assert vus == [("contrat.txt", "1. Term. This Agreement ...")]
    assert e["source"] == "kafka" and e["document"] == "contrat.txt"
    assert len(list((tmp_path / "lake" / "bronze" / "audits").rglob("*.json"))) == 1


def test_message_kafka_refuse_format_et_chemin(tmp_path):
    def jamais(*a):
        raise AssertionError("ne doit pas être appelé")

    msg = json.dumps({"document": "../../virus.exe", "contenu_b64": base64.b64encode(b"x").decode()}).encode()
    with pytest.raises(ValueError, match="Format"):
        process_message(msg, jamais, "m", tmp_path)
