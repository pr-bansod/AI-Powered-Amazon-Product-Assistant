run-docker-compose:
	uv sync
	docker compose up --build

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/*/*.ipynb

run-evals-retriever:
	uv sync
	PYTHONPATH=${PWD}/src:$$PYTHONPATH:${PWD} uv run --env-file .env python -m evals.eval_retriever

# Testing commands
test:
	uv sync
	PYTHONPATH=${PWD}/src:$$PYTHONPATH uv run pytest

test-unit:
	uv sync
	PYTHONPATH=${PWD}/src:$$PYTHONPATH uv run pytest -m unit

test-integration:
	uv sync
	PYTHONPATH=${PWD}/src:$$PYTHONPATH uv run pytest -m integration

test-coverage:
	uv sync
	PYTHONPATH=${PWD}/src:$$PYTHONPATH uv run pytest --cov=src --cov-report=html --cov-report=term-missing

test-verbose:
	uv sync
	PYTHONPATH=${PWD}/src:$$PYTHONPATH uv run pytest -vv

test-watch:
	uv sync
	PYTHONPATH=${PWD}/src:$$PYTHONPATH uv run pytest-watch

test-no-api:
	uv sync
	PYTHONPATH=${PWD}/src:$$PYTHONPATH uv run pytest -m "not requires_api"