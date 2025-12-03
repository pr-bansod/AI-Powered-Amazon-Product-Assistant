import os
from unittest.mock import Mock, patch

import numpy as np
import pytest
from qdrant_client import QdrantClient


@pytest.fixture
def mock_openai_embedding():
    """Mock OpenAI embedding response"""
    mock_response = Mock()
    mock_response.data = [Mock()]
    mock_response.data[0].embedding = np.random.rand(1536).tolist()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.total_tokens = 10
    return mock_response


@pytest.fixture
def mock_openai_chat_completion():
    """Mock OpenAI chat completion response"""
    mock_response = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 150
    return mock_response


@pytest.fixture
def mock_qdrant_points():
    """Mock Qdrant query points response"""
    points = []
    for i in range(5):
        point = Mock()
        point.payload = {
            "parent_asin": f"B00{i}EXAMPLE",
            "description": f"Test product {i} description",
            "average_rating": 4.5,
            "images": f"https://example.com/image{i}.jpg",
            "price": 29.99 + i * 10,
        }
        point.score = 0.9 - i * 0.1
        points.append(point)

    mock_result = Mock()
    mock_result.points = points
    return mock_result


@pytest.fixture
def mock_qdrant_client(mock_qdrant_points):
    """Mock Qdrant client"""
    client = Mock(spec=QdrantClient)
    client.query_points.return_value = mock_qdrant_points
    return client


@pytest.fixture
def sample_rag_context():
    """Sample RAG context data"""
    return {
        "retrieved_context_ids": ["B001EXAMPLE", "B002EXAMPLE"],
        "retrieved_context": [
            "Wireless Bluetooth headphones with noise cancellation",
            "Portable Bluetooth speaker with waterproof design",
        ],
        "retrieved_context_ratings": [4.5, 4.2],
        "similarity_scores": [0.92, 0.87],
    }


@pytest.fixture
def sample_rag_request():
    """Sample RAG request payload"""
    return {"query": "What are some good wireless headphones?"}


@pytest.fixture
def sample_rag_response_with_references():
    """Sample structured RAG response"""
    mock_response = Mock()
    mock_response.answer = (
        "Based on the available products, I recommend the wireless Bluetooth headphones with noise cancellation (B001EXAMPLE)."
    )

    mock_ref1 = Mock()
    mock_ref1.id = "B001EXAMPLE"
    mock_ref1.description = "Wireless Bluetooth headphones with noise cancellation"

    mock_response.references = [mock_ref1]
    return mock_response


@pytest.fixture
def test_env_vars():
    """Set test environment variables"""
    test_vars = {
        "OPENAI_API_KEY": "test_openai_key",
        "GROQ_API_KEY": "test_groq_key",
        "CO_API_KEY": "test_cohere_key",
    }

    with patch.dict(os.environ, test_vars):
        yield test_vars


@pytest.fixture
def mock_langsmith_client():
    """Mock LangSmith client"""
    client = Mock()
    client.pull_prompt.return_value.messages = [Mock()]
    client.pull_prompt.return_value.messages[0].prompt.template = "Test template: {{ question }}"
    return client


@pytest.fixture
def sample_yaml_config():
    """Sample YAML configuration content"""
    return {"prompts": {"test_prompt": "You are a helpful assistant. Question: {{ question }}"}}
