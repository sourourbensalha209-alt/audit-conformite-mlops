# Image du service d'audit : API FastAPI + interface web (frontend/) + chaîne d'audit.
#
#   docker compose up --build        puis   http://localhost:8000
#
# Le modèle (models/best_model_v4, 440 Mo, versionné par DVC) n'est PAS copié dans l'image :
# docker-compose.yml le monte en lecture seule. L'image reste légère et un nouveau modèle ne demande pas de reconstruction.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# OCR (PDF scannés) et conversion PDF -> images
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Dépendances d'abord : cette couche est réutilisée tant que les requirements ne changent pas
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# 2) Code et données nécessaires à l'audit (règles, corpus de lois)
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY configs/ ./configs/
COPY params.yaml ./
COPY data/rules/ ./data/rules/
COPY data/corpus/ ./data/corpus/

# 3) Utilisateur sans droits d'administration
RUN useradd --create-home audit && mkdir -p data/feedback data/monitoring data/lake models && chown -R audit:audit /app
USER audit

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
