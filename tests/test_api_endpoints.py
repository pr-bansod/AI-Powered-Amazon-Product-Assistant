"""
Integration tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app"""
    from api.app import app
    return TestClient(app)


@pytest.fixture
def mock_rag_response():
    """Mock RAG pipeline response"""
    return {
        "answer": "Based on the products, I recommend the wireless headphones.",
        "used_context": [
            {
                "image_url": "http://example.com/image1.jpg",
                "price": 29.99,
                "description": "High-quality wireless headphones"
            },
            {
                "image_url": "http://example.com/image2.jpg",
                "price": 49.99,
                "description": "Premium noise-canceling headphones"
            }
        ]
    }


@pytest.mark.integration
class TestRAGEndpoint:
    """Tests for /rag endpoint"""

    @patch('api.api.endpoints.rag_pipeline_wrapper')
    def test_rag_endpoint_success(self, mock_rag_wrapper, test_client, mock_rag_response):
        """Test successful RAG endpoint call"""
        mock_rag_wrapper.return_value = mock_rag_response

        response = test_client.post(
            "/rag/",
            json={"query": "What are the best wireless headphones?"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "request_id" in data
        assert "answer" in data
        assert "used_context" in data
        assert data["answer"] == mock_rag_response["answer"]
        assert len(data["used_context"]) == 2

    @patch('api.api.endpoints.rag_pipeline_wrapper')
    def test_rag_endpoint_with_empty_context(self, mock_rag_wrapper, test_client):
        """Test RAG endpoint with empty context"""
        mock_rag_wrapper.return_value = {
            "answer": "I don't have enough information.",
            "used_context": []
        }

        response = test_client.post(
            "/rag/",
            json={"query": "Tell me about obscure product XYZ"}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["used_context"]) == 0

    def test_rag_endpoint_missing_query(self, test_client):
        """Test RAG endpoint with missing query parameter"""
        response = test_client.post("/rag/", json={})

        assert response.status_code == 422  # Validation error

    def test_rag_endpoint_invalid_json(self, test_client):
        """Test RAG endpoint with invalid JSON"""
        response = test_client.post(
            "/rag/",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    @patch('api.api.endpoints.rag_pipeline_wrapper')
    def test_rag_endpoint_empty_query(self, mock_rag_wrapper, test_client):
        """Test RAG endpoint with empty query string"""
        response = test_client.post(
            "/rag/",
            json={"query": ""}
        )

        # Should still process but may return minimal results
        assert response.status_code in [200, 422]

    @patch('api.api.endpoints.rag_pipeline_wrapper')
    def test_rag_endpoint_long_query(self, mock_rag_wrapper, test_client, mock_rag_response):
        """Test RAG endpoint with long query"""
        mock_rag_wrapper.return_value = mock_rag_response

        long_query = "What are the best " + "very " * 100 + "good wireless headphones?"

        response = test_client.post(
            "/rag/",
            json={"query": long_query}
        )

        assert response.status_code == 200

    @patch('api.api.endpoints.rag_pipeline_wrapper')
    def test_rag_endpoint_special_characters(self, mock_rag_wrapper, test_client, mock_rag_response):
        """Test RAG endpoint with special characters in query"""
        mock_rag_wrapper.return_value = mock_rag_response

        response = test_client.post(
            "/rag/",
            json={"query": "What about headphones with $50 budget & noise-canceling?"}
        )

        assert response.status_code == 200

    @patch('api.api.endpoints.rag_pipeline_wrapper')
    def test_rag_endpoint_error_handling(self, mock_rag_wrapper, test_client):
        """Test RAG endpoint error handling"""
        mock_rag_wrapper.side_effect = Exception("Database connection failed")

        response = test_client.post(
            "/rag/",
            json={"query": "test query"}
        )

        assert response.status_code == 500


@pytest.mark.integration
class TestAPIResponseStructure:
    """Tests for API response structure validation"""

    @patch('api.api.endpoints.rag_pipeline_wrapper')
    def test_response_has_correct_fields(self, mock_rag_wrapper, test_client, mock_rag_response):
        """Test that response has all required fields"""
        mock_rag_wrapper.return_value = mock_rag_response

        response = test_client.post(
            "/rag/",
            json={"query": "test query"}
        )

        data = response.json()
        required_fields = ["request_id", "answer", "used_context"]

        for field in required_fields:
            assert field in data

    @patch('api.api.endpoints.rag_pipeline_wrapper')
    def test_used_context_structure(self, mock_rag_wrapper, test_client, mock_rag_response):
        """Test that used_context has correct structure"""
        mock_rag_wrapper.return_value = mock_rag_response

        response = test_client.post(
            "/rag/",
            json={"query": "test query"}
        )

        data = response.json()

        for context in data["used_context"]:
            assert "image_url" in context
            assert "price" in context
            assert "description" in context

    @patch('api.api.endpoints.rag_pipeline_wrapper')
    def test_request_id_is_unique(self, mock_rag_wrapper, test_client, mock_rag_response):
        """Test that each request gets a unique request_id"""
        mock_rag_wrapper.return_value = mock_rag_response

        response1 = test_client.post("/rag/", json={"query": "query 1"})
        response2 = test_client.post("/rag/", json={"query": "query 2"})

        data1 = response1.json()
        data2 = response2.json()

        assert data1["request_id"] != data2["request_id"]


@pytest.mark.integration
class TestAPIHeaders:
    """Tests for API headers and CORS"""

    @patch('api.api.endpoints.rag_pipeline_wrapper')
    def test_content_type_header(self, mock_rag_wrapper, test_client, mock_rag_response):
        """Test response content type"""
        mock_rag_wrapper.return_value = mock_rag_response

        response = test_client.post(
            "/rag/",
            json={"query": "test"}
        )

        assert response.headers["content-type"] == "application/json"

    def test_options_request(self, test_client):
        """Test OPTIONS request for CORS"""
        response = test_client.options("/rag/")

        # Should handle OPTIONS request
        assert response.status_code in [200, 405]


@pytest.mark.integration
class TestAPIPerformance:
    """Tests for API performance and rate limiting"""

    @patch('api.api.endpoints.rag_pipeline_wrapper')
    def test_concurrent_requests(self, mock_rag_wrapper, test_client, mock_rag_response):
        """Test handling multiple concurrent requests"""
        mock_rag_wrapper.return_value = mock_rag_response

        responses = []
        for i in range(5):
            response = test_client.post(
                "/rag/",
                json={"query": f"test query {i}"}
            )
            responses.append(response)

        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)

        # All should have unique request IDs
        request_ids = [r.json()["request_id"] for r in responses]
        assert len(set(request_ids)) == 5


@pytest.mark.unit
class TestRAGRequestModel:
    """Tests for RAG request model validation"""

    def test_valid_request_model(self):
        """Test valid RAG request model"""
        from api.api.models import RAGRequest

        request = RAGRequest(query="test query")
        assert request.query == "test query"

    def test_request_model_with_empty_query(self):
        """Test request model with empty query"""
        from api.api.models import RAGRequest

        request = RAGRequest(query="")
        assert request.query == ""

    def test_request_model_missing_query(self):
        """Test request model without query field"""
        from api.api.models import RAGRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RAGRequest()


@pytest.mark.unit
class TestRAGResponseModel:
    """Tests for RAG response model validation"""

    def test_valid_response_model(self):
        """Test valid RAG response model"""
        from api.api.models import RAGResponse, RAGUsedContext

        context = RAGUsedContext(
            image_url="http://example.com/image.jpg",
            price=29.99,
            description="Test product"
        )

        response = RAGResponse(
            request_id="test-123",
            answer="Test answer",
            used_context=[context]
        )

        assert response.request_id == "test-123"
        assert response.answer == "Test answer"
        assert len(response.used_context) == 1

    def test_response_model_with_none_price(self):
        """Test response model with None price"""
        from api.api.models import RAGUsedContext

        context = RAGUsedContext(
            image_url="http://example.com/image.jpg",
            price=None,
            description="Test product"
        )

        assert context.price is None

    def test_response_model_missing_fields(self):
        """Test response model with missing required fields"""
        from api.api.models import RAGResponse
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RAGResponse(answer="Test answer")  # Missing request_id and used_context
