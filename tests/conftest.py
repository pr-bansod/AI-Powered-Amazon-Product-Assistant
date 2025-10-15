"""
Pytest configuration and shared fixtures for all tests.
"""
import pytest
from unittest.mock import Mock, MagicMock
from qdrant_client.models import ScoredPoint, PointStruct
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing"""
    client = Mock()
    return client


@pytest.fixture
def mock_openai_embedding_response():
    """Mock OpenAI embedding API response"""
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1] * 1536)]
    mock_response.usage = Mock(prompt_tokens=10, total_tokens=10)
    return mock_response


@pytest.fixture
def mock_qdrant_query_results():
    """Mock Qdrant query results"""
    mock_points = [
        ScoredPoint(
            id="1",
            version=1,
            score=0.95,
            payload={
                "parent_asin": "B001",
                "description": "Test product 1 description",
                "average_rating": 4.5,
                "image": "http://example.com/image1.jpg",
                "price": 29.99
            },
            vector=None
        ),
        ScoredPoint(
            id="2",
            version=1,
            score=0.85,
            payload={
                "parent_asin": "B002",
                "description": "Test product 2 description",
                "average_rating": 4.0,
                "image": "http://example.com/image2.jpg",
                "price": 49.99
            },
            vector=None
        ),
    ]

    mock_result = Mock()
    mock_result.points = mock_points
    return mock_result


@pytest.fixture
def mock_instructor_response():
    """Mock Instructor/LLM response"""
    from pydantic import BaseModel, Field

    class RAGUsedContext(BaseModel):
        id: str = Field(description="ID of the item used to answer the question.")
        description: str = Field(description="Short description of the item used to answer the question.")

    class RAGGenerationResponseWithReferences(BaseModel):
        answer: str = Field(description="Answer to the question.")
        references: list[RAGUsedContext] = Field(description="List of items used to answer the question.")

    return RAGGenerationResponseWithReferences(
        answer="This is a test answer about the products.",
        references=[
            RAGUsedContext(id="B001", description="Test product 1 description"),
            RAGUsedContext(id="B002", description="Test product 2 description")
        ]
    )


@pytest.fixture
def sample_query():
    """Sample user query for testing"""
    return "What are the best wireless headphones?"


@pytest.fixture
def sample_retrieved_context():
    """Sample retrieved context for testing"""
    return {
        "retrieved_context_ids": ["B001", "B002"],
        "retrieved_context": [
            "Test product 1 description",
            "Test product 2 description"
        ],
        "retrieved_context_ratings": [4.5, 4.0],
        "similarity_scores": [0.95, 0.85]
    }


@pytest.fixture
def mock_prompt_template():
    """Mock prompt template"""
    from jinja2 import Template
    return Template("Context: {{ preprocessed_context }}\nQuestion: {{ question }}")
