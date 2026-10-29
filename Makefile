.PHONY: install data train test lint api app docker-build docker-run all

install:
	pip install -r requirements-dev.txt
	pip install -e .

data:
	python scripts/download_data.py

train: data
	python -m churn.train

test:
	pytest -q

lint:
	ruff check src api app tests

api:
	uvicorn api.main:app --reload

app:
	streamlit run app/streamlit_app.py

docker-build:
	docker build -t churn-api .

docker-run:
	docker run --rm -p 8000:8000 churn-api

all: install train test
