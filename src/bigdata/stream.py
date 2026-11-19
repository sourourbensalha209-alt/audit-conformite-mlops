"""Ingestion en flux avec Kafka : des milliers de contrats audités en parallèle, sans passer par l'interface.

    python -m src.bigdata.stream produire  dossier_de_contrats/     # envoie chaque contrat dans Kafka
    python -m src.bigdata.stream travailler                          # un worker : lit, audite, publie (Ctrl+C pour arrêter)

Sujets (topics) Kafka :
    contrats           contrats à auditer (clé = nom du document, valeur = document encodé en base64)
    audits             résultat de chaque audit (même événement que celui versé dans le data lake)
    contrats.erreurs   contrats qui n'ont pas pu être audités, avec l'erreur (« dead letter queue »)

Montée en charge : tous les workers partagent le groupe « auditeurs ». Kafka répartit les partitions du sujet
entre eux : lancer 3 workers = 3 contrats audités en même temps, sans rien changer au code.
Fiabilité : un message n'est validé (commit) qu'APRÈS son traitement. Si un worker tombe, le contrat est repris
par un autre (traitement « au moins une fois » ; le data lake dédoublonne ensuite par job_id dans Spark).
Limite : un message Kafka est limité à 1 Mo par défaut ; au-delà, déposer le fichier dans le data lake
et n'envoyer que son chemin (patron « claim check »).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.bigdata import lake

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_IN, TOPIC_OUT, TOPIC_ERR = "contrats", "audits", "contrats.erreurs"
GROUP = "auditeurs"
FORMATS = (".pdf", ".docx", ".txt")


def encode_document(path: Path) -> bytes:
    return json.dumps({
        "document": path.name,
        "contenu_b64": base64.b64encode(path.read_bytes()).decode("ascii"),
        "envoye_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }).encode("utf-8")


def process_message(value: bytes, audit_fn: Callable, modele: str, lake_dir: Path | None = None) -> dict:
    """Un message Kafka -> audit -> événement versé dans la zone bronze. Lève une exception si l'audit échoue."""
    msg = json.loads(value)
    name = Path(msg["document"]).name                       # pas de chemin venant du message
    if not name.lower().endswith(FORMATS):
        raise ValueError(f"Format non pris en charge : {name}")
    work = Path(tempfile.mkdtemp())
    try:
        path = work / name
        path.write_bytes(base64.b64decode(msg["contenu_b64"]))
        debut = datetime.now(timezone.utc)
        result, _html = audit_fn(path, path.stem)
        duree = round((datetime.now(timezone.utc) - debut).total_seconds(), 1)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    event = lake.audit_event(f"kafka-{uuid.uuid4().hex[:12]}", name, result, duree, modele, source="kafka")
    lake.write_bronze("audits", event, lake_dir)
    return event


def ensure_topics(bootstrap: str = BOOTSTRAP, partitions: int = 3, attente_max_s: float = 90) -> None:
    """Crée les sujets s'ils n'existent pas, en attendant que Kafka soit prêt (utile au démarrage de docker compose).

    Sans cela, un worker lancé avant le premier envoi attendrait jusqu'à 5 minutes de voir apparaître le sujet.
    """
    import time

    from confluent_kafka import KafkaException
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": bootstrap})
    fin = time.time() + attente_max_s
    while True:
        try:
            existants = set(admin.list_topics(timeout=5).topics)
            break
        except KafkaException as exc:
            if time.time() > fin:
                raise
            print(f"Kafka pas encore prêt ({exc}), nouvel essai...")
            time.sleep(3)
    nouveaux = [NewTopic(t, num_partitions=partitions, replication_factor=1)
                for t in (TOPIC_IN, TOPIC_OUT, TOPIC_ERR) if t not in existants]
    for topic, future in admin.create_topics(nouveaux).items() if nouveaux else []:
        try:
            future.result(30)
            print(f"Sujet créé : {topic}")
        except KafkaException as exc:          # créé entre-temps par un autre worker : sans gravité
            if "TOPIC_ALREADY_EXISTS" not in str(exc):
                raise


def produce(paths: list[Path], bootstrap: str = BOOTSTRAP) -> int:
    from confluent_kafka import Producer

    ensure_topics(bootstrap)
    producer = Producer({"bootstrap.servers": bootstrap, "acks": "all"})
    n = 0
    for p in paths:
        producer.produce(TOPIC_IN, key=p.name.encode("utf-8"), value=encode_document(p))
        producer.poll(0)
        n += 1
    producer.flush(30)
    return n


def work(bootstrap: str = BOOTSTRAP, max_messages: int | None = None, idle_timeout_s: float | None = None,
         audit_fn: Callable | None = None, lake_dir: Path | None = None) -> dict:
    """Boucle d'un worker. S'arrête après max_messages, ou après idle_timeout_s secondes sans message."""
    from confluent_kafka import Consumer, Producer

    if audit_fn is None:
        from src.pipeline import audit_file as audit_fn          # modèle chargé une seule fois par worker
    from src.orchestration.registry import production_model_name
    modele = production_model_name()

    ensure_topics(bootstrap)
    consumer = Consumer({"bootstrap.servers": bootstrap, "group.id": GROUP, "auto.offset.reset": "earliest",
                         "enable.auto.commit": False})
    producer = Producer({"bootstrap.servers": bootstrap})
    consumer.subscribe([TOPIC_IN])
    stats = {"traites": 0, "erreurs": 0}
    attente = 0.0
    try:
        while max_messages is None or stats["traites"] + stats["erreurs"] < max_messages:
            msg = consumer.poll(1.0)
            if msg is None:
                attente += 1.0
                if idle_timeout_s is not None and attente >= idle_timeout_s:
                    break
                continue
            attente = 0.0
            if msg.error():
                print("Erreur Kafka :", msg.error())
                continue
            try:
                event = process_message(msg.value(), audit_fn, modele, lake_dir)
                producer.produce(TOPIC_OUT, key=msg.key(), value=json.dumps(event, ensure_ascii=False).encode("utf-8"))
                stats["traites"] += 1
                print(f"Audité : {event['document']} ({event['duree_s']} s)")
            except Exception as exc:  # noqa: BLE001  (un contrat illisible ne doit pas arrêter le worker)
                producer.produce(TOPIC_ERR, key=msg.key(), value=json.dumps(
                    {"erreur": str(exc)[:500], "date": datetime.now(timezone.utc).isoformat(timespec="seconds")}).encode("utf-8"))
                stats["erreurs"] += 1
                print(f"Échec : {msg.key()!r} : {exc}")
            producer.poll(0)
            consumer.commit(message=msg, asynchronous=False)        # validé seulement après traitement
    finally:
        producer.flush(10)
        consumer.close()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingestion des contrats par Kafka")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("produire", help="envoyer des contrats dans Kafka")
    p.add_argument("chemins", nargs="+", help="fichiers ou dossiers de contrats (.pdf, .docx, .txt)")
    w = sub.add_parser("travailler", help="lancer un worker d'audit")
    w.add_argument("--max", type=int, default=None, help="nombre de contrats à traiter avant de s'arrêter")
    args = ap.parse_args()

    if args.cmd == "produire":
        paths = []
        for c in map(Path, args.chemins):
            paths += sorted(f for f in c.rglob("*") if f.suffix.lower() in FORMATS) if c.is_dir() else [c]
        print(f"{produce(paths)} contrats envoyés dans le sujet « {TOPIC_IN} » ({BOOTSTRAP}).")
    else:
        print(f"Worker du groupe « {GROUP} » à l'écoute de « {TOPIC_IN} » ({BOOTSTRAP}). Ctrl+C pour arrêter.")
        try:
            print(work(max_messages=args.max))
        except KeyboardInterrupt:
            print("Arrêt demandé.")


if __name__ == "__main__":
    main()
