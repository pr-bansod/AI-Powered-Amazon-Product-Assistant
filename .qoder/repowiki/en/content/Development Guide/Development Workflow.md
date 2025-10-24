# Development Workflow

<cite>
**Referenced Files in This Document**   
- [makefile](file://makefile)
- [docker-compose.yml](file://docker-compose.yml)
- [Dockerfile.fastapi](file://Dockerfile.fastapi)
- [Dockerfile.streamlit](file://Dockerfile.streamlit)
- [pyproject.toml](file://pyproject.toml)
- [src/api/app.py](file://src/api/app.py)
- [src/chatbot_ui/app.py](file://src/chatbot_ui/app.py)
- [src/api/rag/retrieval_generation.py](file://src/api/rag/retrieval_generation.py)
</cite>

## Table of Contents
1. [Development Environment Setup](#development-environment-setup)
2. [Makefile Automation Overview](#makefile-automation-overview)
3. [Docker Compose Orchestration](#docker-compose-orchestration)
4. [uv Dependency Management](#uv-dependency-management)
5. [Complete Development Cycle](#complete-development-cycle)
6. [Notebook Maintenance](#notebook-maintenance)
7. [Testing Strategy](#testing-strategy)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Performance Optimization](#performance-optimization)

## Development Environment Setup

Before beginning development, ensure the following prerequisites are installed:
- Python 3.12+
- Docker and Docker Desktop
- uv package manager
- Required API keys (OpenAI, optional: LangSmith, Groq, Google)

The development environment is configured for optimal performance and consistency across team members. The project uses uv as the primary package manager for faster dependency resolution and virtual environment management.

**Section sources**
- [documentation/development-environment/README.md](file://documentation/development-environment/README.md#L0-L90)

## Makefile Automation Overview

The Makefile provides a comprehensive set of commands for automating common development tasks, ensuring consistency and reducing the potential for human error in the development workflow.

### run-docker-compose Target

The `run-docker-compose` target orchestrates the complete development environment startup sequence:

```makefile
run-docker-compose:
	uv sync
	docker compose up --build
```

This command performs two critical operations:
1. **Dependency Synchronization**: Uses `uv sync` to install all project dependencies from `pyproject.toml` and `uv.lock` with maximum efficiency
2. **Service Orchestration**: Launches the full stack via Docker Compose with build instructions from `docker-compose.yml`

The target ensures that all services start with up-to-date dependencies and container images, making it the primary entry point for day-to-day development.

**Section sources**
- [makefile](file://makefile#L0-L3)

## Docker Compose Orchestration

The docker-compose.yml file defines a three-service architecture that enables isolated yet interconnected development of the application components.

```yaml
services:
  streamlit-app:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - 8501:8501
    env_file:
      - .env
    restart: unless-stopped
    volumes:
      - ./src/chatbot_ui:/app/src/chatbot_ui

  api:
    build:
      context: .
      dockerfile: Dockerfile.fastapi
    ports:
      - 8000:8000
    env_file:
      - .env
    restart: unless-stopped
    volumes:
      - ./src/api:/app/src/api

  qdrant:
    image: qdrant/qdrant
    ports:
      - 6333:6333
      - 6334:6334
    volumes:
      - ./qdrant_storage:/qdrant/storage:z
    restart: unless-stopped
```

### Service Configuration

**Streamlit UI Service**
- Built from `Dockerfile.streamlit`
- Exposes port 8501 for web access
- Volume-mounted source code enables hot reloading
- Environment variables loaded from `.env` file

**FastAPI Backend Service**
- Built from `Dockerfile.fastapi`
- Exposes port 8000 for API access
- Volume-mounted source code enables hot reloading
- Environment variables loaded from `.env` file

**Qdrant Vector Database**
- Uses official Qdrant image
- Exposes HTTP (6333) and gRPC (6334) ports
- Persistent storage via volume mount to `./qdrant_storage`
- Auto-restart policy ensures service availability

The volume mounting configuration enables real-time code changes without requiring container rebuilds, significantly accelerating the development feedback loop.

**Diagram sources**
- [docker-compose.yml](file://docker-compose.yml#L0-L32)

**Section sources**
- [docker-compose.yml](file://docker-compose.yml#L0-L32)
- [Dockerfile.fastapi](file://Dockerfile.fastapi#L0-L40)
- [Dockerfile.streamlit](file://Dockerfile.streamlit#L0-L49)

## uv Dependency Management

uv serves as the primary Python package manager, providing significant performance advantages over traditional tools like pip.

### Integration with Development Workflow

uv integrates seamlessly with the development workflow through two primary commands:
- `uv sync`: Installs all dependencies from `pyproject.toml` and `uv.lock`
- `uv run`: Executes commands within the synchronized environment

The Dockerfiles for both services leverage uv for dependency management:

```dockerfile
# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen
```

This approach ensures consistent dependency resolution across development, testing, and production environments while maximizing build performance through layer caching.

### Key Advantages

- **Speed**: Rust-based implementation provides significantly faster dependency resolution
- **Reproducibility**: Lock file (`uv.lock`) ensures identical dependency versions across environments
- **Integration**: Seamless compatibility with Docker and Makefile workflows
- **Virtual Environment Management**: Built-in venv creation and management capabilities

**Section sources**
- [pyproject.toml](file://pyproject.toml#L0-L30)
- [Dockerfile.fastapi](file://Dockerfile.fastapi#L0-L40)
- [Dockerfile.streamlit](file://Dockerfile.streamlit#L0-L49)

## Complete Development Cycle

The development workflow follows a structured cycle that enables rapid iteration and validation of changes.

### Setup Environment

1. Create and activate virtual environment:
```bash
uv venv .venv --python 3.12
source .venv/bin/activate
```

2. Install dependencies:
```bash
uv sync
```

3. Configure environment variables by creating a `.env` file with required API keys.

### Run Services

Start the complete development stack:
```bash
make run-docker-compose
```

This command will:
- Synchronize dependencies using uv
- Build container images if necessary
- Start all services with volume-mounted code for hot reloading

### Make Code Changes

Edit files in the appropriate source directories:
- Backend logic: `src/api/`
- Frontend interface: `src/chatbot_ui/`
- RAG pipeline: `src/api/rag/`

Changes are automatically reflected in running containers due to volume mounting, enabling immediate testing of modifications.

### Execute Tests

The Makefile provides several testing targets:

```makefile
test:
	uv sync
	PYTHONPATH=${PWD}/src:$$PYTHONPATH uv run pytest

test-unit:
	uv sync
	PYTHONPATH=${PWD}/src:$$PYTHONPATH uv run pytest -m unit

test-integration:
	uv sync
	PYTHONPATH=${PWD}/src:$$PYTHONPATH uv run pytest -m integration
```

Run tests using:
```bash
make test
make test-unit
make test-integration
```

### Validate Functionality

Access the running services at:
- **Chat Interface**: http://localhost:8501
- **API Backend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard

**Section sources**
- [makefile](file://makefile#L0-L38)
- [src/api/app.py](file://src/api/app.py#L0-L33)
- [src/chatbot_ui/app.py](file://src/chatbot_ui/app.py#L0-L93)

## Notebook Maintenance

Jupyter notebooks are used extensively for exploratory data analysis and pipeline development, organized by phase in the `notebooks/` directory.

### clean-notebook-outputs Command

The `clean-notebook-outputs` target ensures clean version-controlled notebooks:

```makefile
clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/*/*.ipynb
```

This command:
- Removes all output cells from notebooks
- Preserves code and markdown cells
- Modifies files in place
- Targets all notebooks in subdirectories of `notebooks/`

Running this command before committing changes prevents binary output data from bloating the repository and ensures that notebooks execute cleanly when shared or deployed.

**Section sources**
- [makefile](file://makefile#L5-L6)

## Testing Strategy

The project implements a comprehensive testing strategy with multiple targets for different testing scenarios.

### Test Targets

```makefile
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
```

### Testing Approach

- **Unit Tests**: Focus on isolated components with the `unit` marker
- **Integration Tests**: Test component interactions with the `integration` marker
- **Coverage Reporting**: Generate HTML and terminal coverage reports
- **Verbose Output**: Detailed test output for debugging
- **Watch Mode**: Auto-run tests on file changes
- **API-Independent Tests**: Run tests that don't require external API access

The PYTHONPATH configuration ensures proper module resolution during testing, allowing imports from the `src/` directory.

**Section sources**
- [makefile](file://makefile#L10-L38)
- [tests/test_models.py](file://tests/test_models.py#L0-L75)

## Troubleshooting Guide

Common development issues and their solutions:

### Container Startup Failures

**Symptoms**: Containers fail to start or crash immediately
**Solutions**:
1. Verify Docker Desktop is running
2. Check available system resources (memory, CPU)
3. Ensure proper file permissions
4. Rebuild containers: `docker compose down && docker compose up --build`

### Dependency Conflicts

**Symptoms**: Import errors or version conflicts
**Solutions**:
1. Ensure `uv.lock` is up to date: `uv lock`
2. Clean and reinstall: `rm -rf .venv && uv venv .venv && uv sync`
3. Verify Python version compatibility (requires 3.12+)

### Environment Variable Misconfiguration

**Symptoms**: API errors or missing configuration
**Solutions**:
1. Verify `.env` file exists in project root
2. Check required variables are present (OPENAI_API_KEY)
3. Ensure no trailing spaces or quotes in `.env` values
4. Restart containers after modifying `.env`

### Qdrant Connection Issues

**Symptoms**: Retrieval failures or connection timeouts
**Solutions**:
1. Verify Qdrant container is running: `docker compose ps`
2. Check logs: `docker compose logs qdrant`
3. Verify network connectivity between services
4. Ensure collection exists in Qdrant dashboard

**Section sources**
- [docker-compose.yml](file://docker-compose.yml#L0-L32)
- [README.md](file://README.md#L0-L507)

## Performance Optimization

Strategies for optimizing development efficiency and application performance:

### Selective Testing

Use targeted test commands to reduce feedback cycle time:
- `make test-unit` for fast unit test feedback
- `make test-integration` when testing component interactions
- `make test-no-api` for tests that don't require external services

### Incremental Development

Leverage the notebook-to-production pipeline:
1. Prototype in Jupyter notebooks (`notebooks/phase_3/`)
2. Extract validated code to source modules (`src/api/rag/`)
3. Write tests for new functionality
4. Integrate with the main application

### Docker Optimization

- Use volume mounting for hot reloading to avoid container rebuilds
- Leverage Docker layer caching by ordering Dockerfile instructions appropriately
- Clean unused Docker resources periodically: `docker system prune`

### Dependency Management

- Use `uv sync --frozen` in production to prevent accidental dependency updates
- Regularly update `uv.lock` with `uv lock` when adding new dependencies
- Audit dependencies for security vulnerabilities

**Section sources**
- [makefile](file://makefile#L0-L38)
- [docker-compose.yml](file://docker-compose.yml#L0-L32)
- [pyproject.toml](file://pyproject.toml#L0-L30)