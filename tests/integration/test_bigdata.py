"""Tests d'intégration Big Data, lancés par la CI avec un vrai Kafka (service Docker) et un vrai Spark (Java).

En local, ils sont ignorés automatiquement s'il n'y a ni Kafka (variable KAFKA_BOOTSTRAP) ni pyspark.
Le modèle n'est pas chargé : l'audit est remplacé par une fonction rapide, on teste ici le transport et le calcul.
"""

import json
import os
import time
import uuid

import pytest

from src.bigdata import lake, stream

RESULTATS = {
    "a": {"attendues": [{"categorie": "Cap On Liability", "statut": "manquante"},
                        {"categorie": "Governing Law", "statut": "presente"}],
          "clauses": [{"categorie": "Governing Law", "confiance": 0.99}]},
    "b": {"attendues": [{"categorie": "Cap On Liability", "statut": "presente"},
                        {"categorie": "Governing Law", "statut": "presente"}],
          "clauses": [{"categorie": "Governing Law", "confiance": 0.97}, {"categorie": "Cap On Liability", "confiance": 0.88}]},
}


def faux_audit(path, title):
    if "illisible" in path.name:
        raise ValueError("Aucune clause trouvée")
    r = RESULTATS["a" if "a" in path.stem.split("_")[-1] else "b"]
    return {**r, "monitoring": {"paragraphes": 10, "confiance_moyenne": 0.9, "part_abstention": 0.0, "part_autre": 0.8},
            "stats": {"rgpd": 0}}, "<html/>"


# ---------- Kafka ----------

@pytest.mark.skipif(not os.getenv("KAFKA_BOOTSTRAP"), reason="pas de Kafka (KAFKA_BOOTSTRAP non défini)")
def test_kafka_de_bout_en_bout(tmp_path, monkeypatch):
    confluent = pytest.importorskip("confluent_kafka")
    bootstrap = os.environ["KAFKA_BOOTSTRAP"]
    suffixe = uuid.uuid4().hex[:6]                              # sujets neufs : le test est rejouable
    monkeypatch.setattr(stream, "TOPIC_IN", f"contrats-{suffixe}")
    monkeypatch.setattr(stream, "TOPIC_OUT", f"audits-{suffixe}")
    monkeypatch.setattr(stream, "TOPIC_ERR", f"erreurs-{suffixe}")
    monkeypatch.setattr(stream, "GROUP", f"auditeurs-{suffixe}")

    from confluent_kafka.admin import AdminClient, NewTopic
    admin = AdminClient({"bootstrap.servers": bootstrap})
    futures = admin.create_topics([NewTopic(t, num_partitions=3, replication_factor=1)
                                   for t in (stream.TOPIC_IN, stream.TOPIC_OUT, stream.TOPIC_ERR)])
    for f in futures.values():
        f.result(30)

    docs = []
    for nom in ("contrat_a.txt", "contrat_b.txt", "contrat_a2.txt", "scan_illisible.txt"):
        p = tmp_path / nom
        p.write_text("1. Term. This Agreement shall commence on the Effective Date.", encoding="utf-8")
        docs.append(p)
    assert stream.produce(docs, bootstrap) == 4

    stats = stream.work(bootstrap, max_messages=4, idle_timeout_s=60, audit_fn=faux_audit, lake_dir=tmp_path / "lake")
    assert stats == {"traites": 3, "erreurs": 1}

    consumer = confluent.Consumer({"bootstrap.servers": bootstrap, "group.id": f"lecteur-{suffixe}",
                                   "auto.offset.reset": "earliest"})
    consumer.subscribe([stream.TOPIC_OUT, stream.TOPIC_ERR])
    recus, fin = {stream.TOPIC_OUT: 0, stream.TOPIC_ERR: 0}, time.time() + 60
    while sum(recus.values()) < 4 and time.time() < fin:
        m = consumer.poll(1.0)
        if m is not None and not m.error():
            recus[m.topic()] += 1
    consumer.close()
    assert recus == {stream.TOPIC_OUT: 3, stream.TOPIC_ERR: 1}
    assert len(list((tmp_path / "lake" / "bronze" / "audits").rglob("*.json"))) == 3
    print(f"\nKafka : 4 contrats envoyés, 3 audités, 1 en file d'erreurs ({bootstrap})")


# ---------- Spark ----------

@pytest.fixture(scope="module")
def spark():
    pytest.importorskip("pyspark")
    from src.bigdata.spark_job import spark_session
    s = spark_session("tests-audit")
    yield s
    s.stop()


def test_spark_bronze_silver_gold(tmp_path, spark):
    from src.bigdata.spark_job import run

    racine = tmp_path / "lake"
    for i, cle in enumerate(["a", "b", "a", "b", "a"]):
        r, _ = faux_audit(tmp_path / f"c_{cle}.txt", "t")
        e = lake.audit_event(f"job{i}", f"contrat_{i}.pdf", r, 10.0 + i, "best_model_v4", source="kafka" if i % 2 else "api")
        lake.write_bronze("audits", e, racine)
    lake.write_bronze("audits", e, racine)                          # même job_id réécrit : un seul fichier
    lake.write_bronze("corrections", lake.correction_event(
        {"job_id": "job0", "clause_ref": "§2", "texte": "x", "categorie_corrigee": "Cap On Liability"}), racine)

    resume = run(spark, racine)
    print("\n" + json.dumps(resume, indent=2, ensure_ascii=False, default=str))

    assert resume["audits"] == 5
    cap = next(r for r in resume["par_categorie"] if r["categorie"] == "Cap On Liability")
    assert (cap["contrats"], cap["manquante"], cap["presente"]) == (5, 3, 2) and cap["taux_manquante"] == 0.6
    law = next(r for r in resume["clauses_trouvees"] if r["categorie"] == "Governing Law")
    assert law["occurrences"] == 5
    assert sum(r["via_kafka"] for r in resume["par_jour"]) == 2
    assert resume["corrections"] == [{"categorie": "Cap On Liability", "corrections": 1}]
    for zone in ("silver/audits", "silver/clauses_attendues", "gold/par_categorie", "gold/resume.json"):
        assert (racine / zone).exists()
