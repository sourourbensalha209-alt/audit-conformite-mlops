"""Traitement distribué avec Spark : du data lake brut (bronze) aux indicateurs métier (gold).

    python -m src.bigdata.spark_job --lake data/lake                        # avec pyspark installé (Java requis)
    spark-submit src/bigdata/spark_job.py --lake /app/data/lake            # sur un cluster ou l'image apache/spark

BRONZE -> SILVER  : lecture de tous les événements JSON, typage des dates, suppression des doublons
                    (un contrat retraité par Kafka après une panne n'est compté qu'une fois),
                    une table par niveau de détail (audits, clauses attendues, clauses trouvées), en Parquet.
SILVER -> GOLD    : agrégats que l'on ne peut pas calculer contrat par contrat :
                    * par catégorie : sur combien de contrats la clause est présente / manquante / à vérifier ;
                    * par jour : volume, durée moyenne, confiance du modèle, part « Autre » (dérive dans le temps) ;
                    * clauses trouvées : fréquence et confiance moyenne par catégorie ;
                    * corrections des juristes par catégorie (là où le modèle se trompe le plus).

Ce fichier n'importe que pyspark et la bibliothèque standard, pour pouvoir être lancé tel quel par spark-submit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def spark_session(app: str = "audit-data-lake"):
    from pyspark.sql import SparkSession

    return (SparkSession.builder.master("local[*]").appName(app)
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.ui.enabled", "false")
            .getOrCreate())


def run(spark, lake: str | Path) -> dict:
    from pyspark.sql import functions as F

    lake = Path(lake)
    bronze_audits = lake / "bronze" / "audits"
    if not bronze_audits.exists():
        raise FileNotFoundError(f"Aucun audit dans {bronze_audits} : lancer des audits (interface ou Kafka) d'abord.")

    # ---------- BRONZE -> SILVER ----------
    raw = spark.read.option("pathGlobFilter", "*.json").json(str(bronze_audits))   # découvre la partition « jour »
    audits = (raw.withColumn("date", F.to_timestamp("date"))
              .dropDuplicates(["job_id"])
              .cache())
    silver = lake / "silver"
    (audits.drop("attendues", "clauses")
     .write.mode("overwrite").partitionBy("jour").parquet(str(silver / "audits")))

    attendues = (audits.select("job_id", "jour", "document", F.explode("attendues").alias("a"))
                 .select("job_id", "jour", "document", F.col("a.categorie").alias("categorie"),
                         F.col("a.statut").alias("statut")))
    attendues.write.mode("overwrite").parquet(str(silver / "clauses_attendues"))

    trouvees = (audits.select("job_id", "jour", F.explode("clauses").alias("c"))
                .select("job_id", "jour", F.col("c.categorie").alias("categorie"),
                        F.col("c.confiance").cast("double").alias("confiance")))
    trouvees.write.mode("overwrite").parquet(str(silver / "clauses_trouvees"))

    # ---------- SILVER -> GOLD ----------
    gold = lake / "gold"
    par_categorie = (attendues.groupBy("categorie").agg(
        F.count("*").alias("contrats"),
        F.sum(F.when(F.col("statut") == "presente", 1).otherwise(0)).alias("presente"),
        F.sum(F.when(F.col("statut") == "manquante", 1).otherwise(0)).alias("manquante"),
        F.sum(F.when(F.col("statut") == "a_verifier", 1).otherwise(0)).alias("a_verifier"))
        .withColumn("taux_manquante", F.round(F.col("manquante") / F.col("contrats"), 4))
        .orderBy(F.desc("taux_manquante"), "categorie"))

    par_jour = (audits.groupBy("jour").agg(
        F.count("*").alias("audits"),
        F.round(F.avg("duree_s"), 2).alias("duree_moyenne_s"),
        F.round(F.avg("confiance_moyenne"), 4).alias("confiance_moyenne"),
        F.round(F.avg("part_autre"), 4).alias("part_autre"),
        F.sum(F.when(F.col("source") == "kafka", 1).otherwise(0)).alias("via_kafka"))
        .orderBy("jour"))

    clauses = (trouvees.groupBy("categorie").agg(
        F.count("*").alias("occurrences"),
        F.countDistinct("job_id").alias("contrats"),
        F.round(F.avg("confiance"), 4).alias("confiance_moyenne"))
        .orderBy(F.desc("occurrences"), "categorie"))

    tables = {"par_categorie": par_categorie, "par_jour": par_jour, "clauses_trouvees": clauses}

    bronze_corr = lake / "bronze" / "corrections"
    if bronze_corr.exists() and any(bronze_corr.rglob("*.json")):
        corr = spark.read.option("pathGlobFilter", "*.json").json(str(bronze_corr))
        if "categorie_corrigee" not in corr.columns:
            corr = corr.withColumn("categorie_corrigee", F.lit(None).cast("string"))
        tables["corrections"] = (corr.withColumn("categorie", F.coalesce("categorie_corrigee", "clause_ref"))
                                 .groupBy("categorie").agg(F.count("*").alias("corrections"))
                                 .orderBy(F.desc("corrections"), "categorie"))

    for name, df in tables.items():
        df.write.mode("overwrite").parquet(str(gold / name))

    resume = {
        "audits": audits.count(),
        "contrats_distincts": audits.select("document").distinct().count(),
        **{name: [r.asDict() for r in df.collect()] for name, df in tables.items()},
    }
    gold.mkdir(parents=True, exist_ok=True)
    (gold / "resume.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    audits.unpersist()
    return resume


def main() -> None:
    ap = argparse.ArgumentParser(description="Job Spark : bronze -> silver -> gold")
    ap.add_argument("--lake", default=str(Path(__file__).resolve().parents[2] / "data" / "lake"))
    args = ap.parse_args()
    spark = spark_session()
    try:
        r = run(spark, args.lake)
    finally:
        spark.stop()
    print(f"{r['audits']} audits, {r['contrats_distincts']} contrats distincts -> {args.lake}/gold")
    print("Catégories le plus souvent manquantes :")
    for row in r["par_categorie"][:5]:
        print(f"  {row['categorie']:30s} {row['manquante']}/{row['contrats']} ({row['taux_manquante']:.0%})")


if __name__ == "__main__":
    main()
