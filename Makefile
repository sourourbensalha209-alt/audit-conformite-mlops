.PHONY: install data pipeline api mlflow test lint clean docker docker-stop retrain promote bigdata spark

install:
	pip install -r requirements.txt

data:
	python -m src.data.download

pipeline:
	dvc repro

api:
	uvicorn src.api.main:app --reload --port 8000

mlflow:
	mlflow ui --port 5000

test:
	pytest -q

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +

docker:
	docker compose up --build -d

docker-stop:
	docker compose down

retrain:
	python -m src.orchestration.retrain

promote:
	python -m src.orchestration.promote $(MODELE)

bigdata:
	docker compose --profile bigdata up --build -d

spark:
	docker compose run --rm spark
