.PHONY: install data pipeline api mlflow test lint clean

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
