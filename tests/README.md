# Test Suite

Comprehensive test suite for the AI-Powered Amazon Product Assistant.

## Test Structure

```
tests/
├── __init__.py                    # Test package initialization
├── conftest.py                    # Shared fixtures and configuration
├── test_rag_pipeline.py          # Unit tests for RAG pipeline functions
├── test_prompt_management.py     # Unit tests for prompt utilities
└── test_api_endpoints.py         # Integration tests for API endpoints
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage report
```bash
pytest --cov=src --cov-report=html --cov-report=term-missing
```

### Run specific test categories
```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Tests that don't require external APIs
pytest -m "not requires_api"

# Slow tests
pytest -m slow
```

### Run specific test file
```bash
pytest tests/test_rag_pipeline.py
pytest tests/test_api_endpoints.py
pytest tests/test_prompt_management.py
```

### Run specific test class or function
```bash
pytest tests/test_rag_pipeline.py::TestGetEmbedding
pytest tests/test_rag_pipeline.py::TestGetEmbedding::test_get_embedding_success
```

### Run with verbose output
```bash
pytest -v
pytest -vv  # Extra verbose
```

### Run with output capture disabled (see print statements)
```bash
pytest -s
```

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Fast unit tests with mocked dependencies
- `@pytest.mark.integration` - Integration tests with real components
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.requires_api` - Tests requiring external API calls (OpenAI, Qdrant, etc.)

## Test Coverage

View coverage report after running tests:

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Writing New Tests

### Test File Naming
- Test files must start with `test_`
- Example: `test_new_feature.py`

### Test Function Naming
- Test functions must start with `test_`
- Use descriptive names: `test_function_name_scenario`
- Example: `test_retrieve_data_with_empty_results`

### Using Fixtures
Shared fixtures are defined in `conftest.py`:

```python
def test_my_function(mock_qdrant_client, sample_query):
    # Use fixtures directly as function parameters
    result = my_function(mock_qdrant_client, sample_query)
    assert result is not None
```

### Mocking External Dependencies

```python
from unittest.mock import patch, Mock

@patch('api.rag.retrieval_generation.openai.embeddings.create')
def test_with_mocked_openai(mock_create):
    mock_create.return_value = Mock(data=[Mock(embedding=[0.1] * 1536)])
    # Your test code here
```

## Test Best Practices

1. **Isolation**: Each test should be independent and not rely on other tests
2. **Mocking**: Mock external dependencies (APIs, databases) in unit tests
3. **Assertions**: Use clear, specific assertions
4. **Coverage**: Aim for >80% code coverage
5. **Documentation**: Add docstrings to test classes and complex tests
6. **Fast Tests**: Keep unit tests fast (<100ms each)
7. **Integration Tests**: Mark slow/integration tests appropriately

## Continuous Integration

Tests run automatically on:
- Pull requests to `main` and `develop` branches
- Commits to `main` and `develop` branches

## Troubleshooting

### Import Errors
If you see import errors, ensure the src directory is in the Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Fixture Not Found
Make sure fixtures are defined in `conftest.py` or imported properly.

### API Tests Failing
Tests marked with `requires_api` need:
- Valid API keys in environment variables
- Network connectivity
- Running Qdrant instance

Skip these tests locally:
```bash
pytest -m "not requires_api"
```

## Dependencies

Testing dependencies are defined in `pyproject.toml`:
- `pytest` - Test framework
- `pytest-asyncio` - Async test support
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Enhanced mocking
- `httpx` - HTTP client for API testing

Install all dependencies:
```bash
uv sync
```
