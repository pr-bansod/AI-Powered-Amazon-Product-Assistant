# Testing Practices

<cite>
**Referenced Files in This Document**   
- [makefile](file://makefile)
- [pytest.ini](file://pytest.ini)
- [tests/conftest.py](file://tests/conftest.py)
- [tests/test_models.py](file://tests/test_models.py)
- [src/api/api/models.py](file://src/api/api/models.py)
- [src/api/rag/retrieval_generation.py](file://src/api/rag/retrieval_generation.py)
- [src/api/api/endpoints.py](file://src/api/api/endpoints.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Test Directory Structure](#test-directory-structure)
3. [Testing Framework Configuration](#testing-framework-configuration)
4. [Shared Fixtures in conftest.py](#shared-fixtures-in-conftestpy)
5. [Model Validation Testing](#model-validation-testing)
6. [Available Makefile Commands](#available-makefile-commands)
7. [Writing Unit Tests](#writing-unit-tests)
8. [Writing Integration Tests](#writing-integration-tests)
9. [Test Isolation and Mocking](#test-isolation-and-mocking)
10. [Coverage Reporting](#coverage-reporting)
11. [Development Workflows](#development-workflows)
12. [Best Practices](#best-practices)

## Introduction
This document details the testing practices for the AI-Powered Amazon Product Assistant, focusing on the pytest-based testing framework and development workflows. The system employs a comprehensive testing strategy to ensure reliability of both the RAG (Retrieval-Augmented Generation) pipeline and API components. The testing infrastructure supports unit testing, integration testing, coverage analysis, and offline testing modes, all orchestrated through Makefile commands. This documentation covers the structure of the tests/ directory, shared fixtures, available test commands, and best practices for writing and executing tests.

## Test Directory Structure
The tests/ directory contains the complete test suite for the application, organized to support both unit and integration testing. The directory includes essential configuration files and test modules that validate core functionality.

```mermaid
graph TD
tests["tests/"]
init["__init__.py"]
conftest["conftest.py"]
test_models["test_models.py"]
tests --> init
tests --> conftest
tests --> test_models
```

**Diagram sources**
- [tests/__init__.py](file://tests/__init__.py)
- [tests/conftest.py](file://tests/conftest.py)
- [tests/test_models.py](file://tests/test_models.py)

**Section sources**
- [tests/conftest.py](file://tests/conftest.py#L1-L115)
- [tests/test_models.py](file://tests/test_models.py#L1-L76)

## Testing Framework Configuration
The testing framework is configured through pytest.ini, which defines the test discovery patterns, markers, and default options. This configuration ensures consistent test execution across different environments and provides structured reporting.

```mermaid
erDiagram
PYTEST_CONFIG {
string testpaths
string python_files
string python_classes
string python_functions
string addopts
string markers
}
```

**Diagram sources**
- [pytest.ini](file://pytest.ini#L1-L18)

**Section sources**
- [pytest.ini](file://pytest.ini#L1-L18)

## Shared Fixtures in conftest.py
The conftest.py file provides shared fixtures that are available across all test modules. These fixtures mock external dependencies and provide test data, enabling isolated and reliable testing without requiring actual API calls or database connections.

```mermaid
classDiagram
class conftest{
+mock_qdrant_client()
+mock_openai_embedding_response()
+mock_qdrant_query_results()
+mock_instructor_response()
+sample_query()
+sample_retrieved_context()
+mock_prompt_template()
}
```

**Diagram sources**
- [tests/conftest.py](file://tests/conftest.py#L15-L114)

**Section sources**
- [tests/conftest.py](file://tests/conftest.py#L15-L114)

## Model Validation Testing
The test_models.py file contains unit tests for Pydantic models used in the API, specifically validating the RAGRequest and RAGResponse models. These tests ensure that model validation works correctly for various input scenarios, including edge cases like empty queries and missing fields.

```mermaid
classDiagram
class TestRAGRequestModel{
+test_valid_request_model()
+test_request_model_with_empty_query()
+test_request_model_missing_query()
}
class TestRAGResponseModel{
+test_valid_response_model()
+test_response_model_with_none_price()
+test_response_model_missing_fields()
}
```

**Diagram sources**
- [tests/test_models.py](file://tests/test_models.py#L8-L75)
- [src/api/api/models.py](file://src/api/api/models.py#L4-L16)

**Section sources**
- [tests/test_models.py](file://tests/test_models.py#L8-L75)
- [src/api/api/models.py](file://src/api/api/models.py#L4-L16)

## Available Makefile Commands
The Makefile provides several commands for executing tests in different configurations. These commands handle dependency management, Python path configuration, and test execution with appropriate options.

```mermaid
flowchart TD
A["Makefile Commands"] --> B["test"]
A --> C["test-unit"]
A --> D["test-integration"]
A --> E["test-coverage"]
A --> F["test-verbose"]
A --> G["test-watch"]
A --> H["test-no-api"]
B --> I["Run all tests"]
C --> J["Run unit tests only"]
D --> K["Run integration tests only"]
E --> L["Run tests with coverage report"]
F --> M["Run tests with verbose output"]
G --> N["Run tests in watch mode"]
H --> O["Run tests without API calls"]
```

**Diagram sources**
- [makefile](file://makefile#L1-L38)

**Section sources**
- [makefile](file://makefile#L1-L38)

## Writing Unit Tests
Unit tests should focus on isolated components and use the provided fixtures to mock dependencies. The example below demonstrates how to write a unit test for API models using pytest markers and fixture injection.

```mermaid
sequenceDiagram
participant Test as "Test Function"
participant Fixture as "conftest.py"
participant Model as "Pydantic Model"
Test->>Fixture : Request mock data
Fixture-->>Test : Provide sample_query
Test->>Model : Create RAGRequest
Model-->>Test : Return validated model
Test->>Test : Assert expected values
```

**Diagram sources**
- [tests/test_models.py](file://tests/test_models.py#L8-L75)
- [tests/conftest.py](file://tests/conftest.py#L90-L95)
- [src/api/api/models.py](file://src/api/api/models.py#L4-L6)

**Section sources**
- [tests/test_models.py](file://tests/test_models.py#L8-L75)

## Writing Integration Tests
Integration tests validate the interaction between components, particularly the RAG pipeline. These tests should use appropriate markers and can be executed selectively using the test-integration Makefile command.

```mermaid
sequenceDiagram
participant Test as "Integration Test"
participant Endpoint as "API Endpoint"
participant RAG as "RAG Pipeline"
participant MockDB as "Mock Qdrant"
Test->>Endpoint : Send request
Endpoint->>RAG : Process query
RAG->>MockDB : Retrieve data
MockDB-->>RAG : Return mock results
RAG->>Endpoint : Generate answer
Endpoint-->>Test : Return response
```

**Diagram sources**
- [src/api/api/endpoints.py](file://src/api/api/endpoints.py#L15-L73)
- [src/api/rag/retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L331-L400)
- [tests/conftest.py](file://tests/conftest.py#L15-L30)

**Section sources**
- [src/api/api/endpoints.py](file://src/api/api/endpoints.py#L15-L73)
- [src/api/rag/retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L331-L400)

## Test Isolation and Mocking
The testing framework employs mocking to isolate tests from external dependencies. The conftest.py file provides mock objects for Qdrant, OpenAI, and other external services, ensuring tests are reliable and fast.

```mermaid
classDiagram
class MockQdrantClient{
+query_points()
+create_collection()
+upload_points()
}
class MockOpenAIResponse{
+data
+usage
}
class MockInstructorResponse{
+answer
+references
}
MockQdrantClient <|-- mock_qdrant_client
MockOpenAIResponse <|-- mock_openai_embedding_response
MockInstructorResponse <|-- mock_instructor_response
```

**Diagram sources**
- [tests/conftest.py](file://tests/conftest.py#L15-L85)

**Section sources**
- [tests/conftest.py](file://tests/conftest.py#L15-L85)

## Coverage Reporting
The test-coverage Makefile command generates HTML coverage reports, providing detailed insights into test coverage across the codebase. This helps identify untested code paths and ensures comprehensive test coverage.

```mermaid
flowchart TD
A["test-coverage command"] --> B["Run pytest with --cov"]
B --> C["Generate HTML report"]
B --> D["Generate terminal report"]
C --> E["Open htmlcov/index.html"]
D --> F["Display missing lines"]
```

**Diagram sources**
- [makefile](file://makefile#L25-L28)
- [pytest.ini](file://pytest.ini#L7-L10)

**Section sources**
- [makefile](file://makefile#L25-L28)
- [pytest.ini](file://pytest.ini#L7-L10)

## Development Workflows
The testing infrastructure supports various development workflows, including watch mode for continuous testing and verbose output for detailed debugging information. These workflows help maintain a fast feedback loop during development.

```mermaid
flowchart TD
A["Development Workflow"] --> B["Write code"]
B --> C["Run test-watch"]
C --> D["Save file"]
D --> E["Tests auto-run"]
E --> F{"Tests pass?"}
F --> |Yes| G["Commit changes"]
F --> |No| H["Debug with test-verbose"]
H --> I["Fix code"]
I --> B
```

**Diagram sources**
- [makefile](file://makefile#L35-L38)

**Section sources**
- [makefile](file://makefile#L35-L38)

## Best Practices
Adhering to best practices ensures effective and maintainable tests. Key practices include using appropriate pytest markers, maintaining test isolation, and following the provided testing patterns.

```mermaid
flowchart TD
A["Testing Best Practices"] --> B["Use pytest markers"]
A --> C["Mock external dependencies"]
A --> D["Keep tests isolated"]
A --> E["Use descriptive test names"]
A --> F["Test edge cases"]
A --> G["Maintain fast feedback loop"]
A --> H["Use coverage reports"]
B --> I["@pytest.mark.unit"]
B --> J["@pytest.mark.integration"]
B --> K["@pytest.mark.requires_api"]
C --> L["Use conftest.py fixtures"]
D --> M["Avoid shared state"]
```

**Diagram sources**
- [pytest.ini](file://pytest.ini#L15-L18)
- [tests/conftest.py](file://tests/conftest.py#L1-L115)
- [makefile](file://makefile#L1-L38)

**Section sources**
- [pytest.ini](file://pytest.ini#L15-L18)
- [tests/conftest.py](file://tests/conftest.py#L1-L115)
- [makefile](file://makefile#L1-L38)