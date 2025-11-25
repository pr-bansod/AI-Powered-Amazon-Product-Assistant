run-docker-compose:
	uv sync
	docker compose up --build

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/*/*.ipynb

run-evals-retriever:
	uv sync
	PYTHONPATH=${PWD}/src:$$PYTHONPATH:${PWD} uv run --env-file .env python -m evals.eval_retriever

# Test commands
test:
	uv sync
	python -m pytest tests/ -v

test-cov:
	uv sync
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

format: ## Format code
	uv run ruff format

lint: ## Lint and type check
	uv run ruff check --fix
	uv run mypy src/

clean: ## Clean up everything
	docker compose down -v
	docker system prune -f