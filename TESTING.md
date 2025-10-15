# Testing Guide

This document provides comprehensive information about testing the AI-Powered Amazon Product Assistant.

## Overview

The project uses **pytest** as the testing framework with comprehensive coverage of:
- Unit tests for RAG pipeline functions
- Unit tests for prompt management utilities
- Integration tests for API endpoints
- Mocked external dependencies (OpenAI, Qdrant, Cohere)

## Quick Start

```bash
# Install dependencies
uv sync

# Run all tests
make test

# Run with coverage report
make test-coverage

# Run only unit tests (fast)
make test-unit

# Run only integration tests
make test-integration
```

## Test Organization

### Directory Structure

```
tests/
├── __init__.py                   # Package initialization
├── conftest.py                   # Shared fixtures and configuration
├── test_rag_pipeline.py         # RAG pipeline unit tests (20+ tests)
├── test_prompt_management.py    # Prompt utilities tests (10+ tests)
├── test_api_endpoints.py        # API integration tests (20+ tests)
└── README.md                     # Detailed testing documentation
```

### Test Categories

#### 1. RAG Pipeline Tests (`test_rag_pipeline.py`)

**Coverage:**
- `get_embedding()` - OpenAI embedding generation
- `retrieve_data()` - Hybrid search (semantic + BM25)
- `process_context()` - Context formatting
- `build_prompt()` - Prompt template rendering
- `generate_answer()` - LLM response generation with Instructor
- `rag_pipeline()` - End-to-end pipeline integration
- `rag_pipeline_wrapper()` - Full pipeline with context enrichment

**Example tests:**
```python
# Test embedding generation
def test_get_embedding_success()

# Test hybrid retrieval
def test_retrieve_data_success()

# Test complete pipeline
def test_rag_pipeline_integration()
```

#### 2. Prompt Management Tests (`test_prompt_management.py`)

**Coverage:**
- `prompt_template_config()` - YAML-based template loading
- `prompt_template_registry()` - LangSmith template pulling
- Template rendering with variables
- Error handling for missing templates

**Example tests:**
```python
# Test YAML template loading
def test_prompt_template_config_success()

# Test template rendering
def test_prompt_template_config_rendering()

# Test LangSmith integration
def test_prompt_template_registry_success()
```

#### 3. API Endpoint Tests (`test_api_endpoints.py`)

**Coverage:**
- POST `/rag/` endpoint functionality
- Request/response validation (Pydantic models)
- Error handling and edge cases
- Response structure validation
- Concurrent request handling

**Example tests:**
```python
# Test successful API call
def test_rag_endpoint_success()

# Test validation errors
def test_rag_endpoint_missing_query()

# Test response structure
def test_response_has_correct_fields()
```

## Test Markers

Tests are organized using pytest markers for selective execution:

- `@pytest.mark.unit` - Fast unit tests with mocked dependencies (~50ms each)
- `@pytest.mark.integration` - Integration tests with real components (~500ms each)
- `@pytest.mark.slow` - Tests that take longer to execute (>1s)
- `@pytest.mark.requires_api` - Tests requiring external APIs (OpenAI, Qdrant, LangSmith)

## Running Tests

### All Tests
```bash
make test                    # Run all tests
pytest                       # Alternative
```

### By Category
```bash
make test-unit              # Fast unit tests only
make test-integration       # Integration tests only
make test-no-api           # Skip tests requiring external APIs
```

### With Coverage
```bash
make test-coverage          # Generate HTML + terminal coverage report
open htmlcov/index.html    # View HTML report (macOS)
```

### Specific Tests
```bash
# Single file
pytest tests/test_rag_pipeline.py

# Single class
pytest tests/test_rag_pipeline.py::TestGetEmbedding

# Single test
pytest tests/test_rag_pipeline.py::TestGetEmbedding::test_get_embedding_success

# By marker
pytest -m unit
pytest -m "not requires_api"
```

### Verbose Output
```bash
make test-verbose           # Extra verbose output
pytest -vv -s              # Verbose with stdout/stderr
```

## Fixtures

Shared test fixtures are defined in `tests/conftest.py`:

### Mock Fixtures
- `mock_qdrant_client` - Mocked Qdrant vector database client
- `mock_openai_embedding_response` - Mocked OpenAI embedding API response
- `mock_qdrant_query_results` - Mocked Qdrant search results
- `mock_instructor_response` - Mocked structured LLM output
- `mock_prompt_template` - Mocked Jinja2 template

### Data Fixtures
- `sample_query` - Example user query
- `sample_retrieved_context` - Example retrieved context data

### Usage Example
```python
def test_my_function(mock_qdrant_client, sample_query):
    # Fixtures are automatically injected
    result = retrieve_data(sample_query, mock_qdrant_client)
    assert result is not None
```

## Mocking Strategy

### External Dependencies Mocked in Unit Tests:
- **OpenAI API** - Embedding and chat completion calls
- **Qdrant** - Vector database queries
- **Cohere** - Reranking API (if used)
- **LangSmith** - Observability and prompt registry

### Example Mock Usage
```python
from unittest.mock import patch, Mock

@patch('api.rag.retrieval_generation.openai.embeddings.create')
def test_embedding(mock_create, mock_openai_embedding_response):
    mock_create.return_value = mock_openai_embedding_response
    result = get_embedding("test query")
    assert len(result) == 1536
```

## Coverage Goals

**Current Coverage Targets:**
- Overall: >80%
- RAG Pipeline: >90%
- API Endpoints: >85%
- Utilities: >80%

**View Coverage:**
```bash
make test-coverage
open htmlcov/index.html
```

## Continuous Integration

Tests run automatically on:
- Pull requests to `main` and `develop`
- Direct commits to `main` and `develop`

**CI Configuration:**
- All unit tests must pass
- Integration tests run with mocked APIs
- Coverage report generated
- Tests marked `requires_api` are skipped

## Writing New Tests

### 1. Create Test File
```python
# tests/test_new_feature.py
"""
Tests for new feature.
"""
import pytest

@pytest.mark.unit
class TestNewFeature:
    """Tests for new feature"""

    def test_basic_functionality(self):
        """Test basic functionality"""
        result = my_function()
        assert result is not None
```

### 2. Add Fixtures (if needed)
Add shared fixtures to `tests/conftest.py`:

```python
@pytest.fixture
def my_fixture():
    """My custom fixture"""
    return {"key": "value"}
```

### 3. Use Mocking
```python
from unittest.mock import patch

@patch('module.external_call')
def test_with_mock(mock_external):
    mock_external.return_value = "mocked result"
    # Test code here
```

### 4. Add Markers
```python
@pytest.mark.unit
@pytest.mark.slow
def test_slow_operation():
    # Test code
```

## Best Practices

1. **Isolation** - Each test is independent
2. **Fast Unit Tests** - Mock external dependencies
3. **Clear Naming** - `test_function_name_scenario`
4. **Single Assertion Focus** - Test one thing per test
5. **Arrange-Act-Assert** - Clear test structure
6. **Docstrings** - Document complex tests
7. **Fixtures** - Reuse common test data
8. **Markers** - Categorize tests appropriately

## Troubleshooting

### Import Errors
```bash
# Ensure src is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
# Or use make commands (handles this automatically)
make test
```

### Fixture Not Found
- Check `conftest.py` for fixture definition
- Ensure fixture name matches parameter name
- Check fixture scope (`function`, `module`, `session`)

### Slow Tests
```bash
# Skip slow tests
pytest -m "not slow"

# Profile slow tests
pytest --durations=10
```

### API Tests Failing Locally
```bash
# Skip tests requiring external APIs
make test-no-api
pytest -m "not requires_api"
```

## Test Metrics

**Test Statistics:**
- Total tests: 50+
- Unit tests: 35+
- Integration tests: 15+
- Average test duration: <100ms (unit), <500ms (integration)
- Coverage: >80%

## Dependencies

Testing libraries in `pyproject.toml`:
```toml
pytest>=8.0.0              # Testing framework
pytest-asyncio>=0.23.0     # Async test support
pytest-cov>=4.1.0          # Coverage reporting
pytest-mock>=3.12.0        # Enhanced mocking
httpx>=0.27.0              # HTTP testing
```

## Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [Test Coverage](https://coverage.readthedocs.io/)

## Questions?

For questions about testing:
1. Check `tests/README.md` for detailed documentation
2. Review existing tests for examples
3. Consult pytest documentation
4. Ask in team discussions
