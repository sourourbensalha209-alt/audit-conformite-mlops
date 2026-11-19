#!/usr/bin/env bash
# Démonstration complète de la couche Big Data (dans GitHub Codespaces, ou toute machine avec Docker).
#
#   bash scripts/demo_bigdata.sh            # 40 contrats synthétiques
#   bash scripts/demo_bigdata.sh 100        # 100 contrats
#
# 1. démarre tout : API + interface, MLflow, Kafka, 2 workers d'audit ;
# 2. génère des contrats synthétiques et les envoie dans Kafka ;
# 3. suit les 2 workers qui les auditent en parallèle (zone bronze du data lake) ;
# 4. lance le job Spark : bronze -> silver -> gold ;
# 5. affiche les agrégats : ils apparaissent aussi dans l'interface (page Monitoring, carte « Data lake »).
set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-40}"
export KAFKA_BOOTSTRAP=localhost:9092

if [ ! -f models/best_model_v4/model.safetensors ]; then
  echo "Modèle absent (models/best_model_v4). Lancer d'abord : bash .devcontainer/post-create.sh"
  exit 1
fi
mkdir -p data/lake data/feedback data/monitoring

echo "== 1/5  Démarrage : API, MLflow, Kafka et 2 workers (la première fois : quelques minutes) =="
docker compose --profile bigdata up --build -d
echo -n "Attente de l'API "
until curl -fs http://localhost:8000/health > /dev/null; do echo -n "."; sleep 3; done
echo " prête."

AVANT=$(find data/lake/bronze/audits -name '*.json' 2>/dev/null | wc -l)

echo "== 2/5  Génération de $N contrats synthétiques et envoi dans Kafka (sujet « contrats ») =="
rm -rf data/demo_contrats && python scripts/generer_contrats_demo.py --n "$N" --out data/demo_contrats
python -m src.bigdata.stream produire data/demo_contrats

echo "== 3/5  Les 2 workers auditent les contrats en parallèle =="
CIBLE=$((AVANT + N))
DEBUT=$(date +%s)
while true; do
  FAIT=$(find data/lake/bronze/audits -name '*.json' 2>/dev/null | wc -l)
  printf "\r   %d / %d contrats audités (%d s)" "$((FAIT - AVANT))" "$N" "$(( $(date +%s) - DEBUT ))"
  [ "$FAIT" -ge "$CIBLE" ] && break
  [ $(( $(date +%s) - DEBUT )) -gt 900 ] && { echo; echo "Délai dépassé : voir  docker compose logs worker"; break; }
  sleep 3
done
echo
echo "   Répartition entre les workers :"
docker compose --profile bigdata logs --no-color worker 2>/dev/null | grep -o '^[^|]*| Audité' | sort | uniq -c || true

echo "== 4/5  Job Spark : bronze -> silver -> gold =="
docker compose --profile bigdata run --rm spark

echo "== 5/5  Résultat (zone gold) =="
python - <<'PY'
import json
r = json.load(open("data/lake/gold/resume.json", encoding="utf-8"))
print(f"{r['audits']} audits dans le data lake, {r['contrats_distincts']} contrats distincts\n")
print(f"{'Clause attendue':32s} {'manquante':>10s} {'taux':>6s}")
for row in r["par_categorie"]:
    print(f"{row['categorie']:32s} {row['manquante']:>4d} / {row['contrats']:<4d} {row['taux_manquante']:>6.0%}")
print("\nClauses trouvées le plus souvent :")
for row in r["clauses_trouvees"][:8]:
    print(f"  {row['categorie']:30s} {row['occurrences']:>4d}  (confiance moyenne {row['confiance_moyenne']})")
PY
echo
echo "Interface : onglet PORTS -> port 8000 -> page Monitoring, carte « Data lake »."
echo "Arrêter tout :  docker compose --profile bigdata down"
