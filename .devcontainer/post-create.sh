#!/usr/bin/env bash
# Préparation du Codespace : outils, puis modèle récupéré depuis DagsHub avec DVC.
set -euo pipefail

echo "== Outils : DVC (récupérer le modèle) et client Kafka (envoyer les contrats) =="
pip install --quiet dvc==3.56.0 "pathspec==0.12.1" confluent-kafka==2.6.1

mkdir -p data/lake data/feedback data/monitoring data/demo_contrats

DAGSHUB_USER="${DAGSHUB_USER:-sourourbensalha209-alt}"
if [ -n "${DAGSHUB_TOKEN:-}" ]; then
  echo "== Récupération du modèle depuis DagsHub =="
  dvc remote modify origin --local auth basic
  dvc remote modify origin --local user "$DAGSHUB_USER"
  dvc remote modify origin --local password "$DAGSHUB_TOKEN"
  dvc pull models/best_model_v4.dvc
  echo "Modèle prêt : models/best_model_v4"
else
  echo "!! Secret DAGSHUB_TOKEN absent : le modèle n'a pas été téléchargé."
  echo "   Ajoutez-le (GitHub > Settings > Codespaces > Secrets), puis lancez : bash .devcontainer/post-create.sh"
fi

echo
echo "Prêt. Pour la démonstration Big Data :  bash scripts/demo_bigdata.sh"
