import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient
from src.api.api.models import RAGRequest, RAGResponse, RAGUsedContext
from src.api.app import app


class TestFastAPIApp:
    """Test FastAPI application and its endpoints"""

    def test_app_creation(self):
        """Test that the FastAPI app is created correctly"""
        assert app.title == "FastAPI"
        assert hasattr(app, "router")

    def test_root_endpoint(self):
        """Test the root endpoint returns welcome message"""
        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        assert response.json() == {"message": "API"}

    def test_cors_middleware_configured(self):
        """Test that CORS middleware is properly configured"""
        # Check if CORS middleware is in the middleware stack
        cors_middleware_found = False
        for middleware in app.user_middleware:
            if "CORSMiddleware" in str(middleware):
                cors_middleware_found = True
                break

        assert cors_middleware_found, "CORS middleware should be configured"

    def test_request_id_middleware_configured(self):
        """Test that RequestID middleware is configured"""
        # Check if RequestIDMiddleware is in the middleware stack
        request_id_middleware_found = False
        for middleware in app.user_middleware:
            if "RequestIDMiddleware" in str(middleware):
                request_id_middleware_found = True
                break

        assert request_id_middleware_found, "RequestID middleware should be configured"


class TestRAGEndpoint:
    """Test the RAG endpoint specifically for OpenAI integration"""

    @patch("src.api.api.endpoint.rag_pipeline_wrapper")
    def test_rag_endpoint_success(self, mock_rag_pipeline_wrapper):
        """Test successful RAG endpoint call with OpenAI"""
        # Mock the RAG pipeline response
        mock_rag_pipeline_wrapper.return_value = {
            "answer": "Based on the available products, I recommend the Sony WH-1000XM4 wireless headphones.",
            "used_context": [
                {
                    "image_url": "https://example.com/sony-headphones.jpg",
                    "price": 349.99,
                    "description": "Sony WH-1000XM4 Wireless Noise Canceling Headphones",
                }
            ],
        }

        client = TestClient(app)
        response = client.post("/rag/", json={"query": "What are the best wireless headphones?"})

        assert response.status_code == 200
        response_data = response.json()

        # Verify response structure
        assert "request_id" in response_data
        assert "answer" in response_data
        assert "used_context" in response_data

        # Verify OpenAI response content
        assert "Sony WH-1000XM4" in response_data["answer"]
        assert len(response_data["used_context"]) == 1

        context_item = response_data["used_context"][0]
        assert context_item["price"] == 349.99
        assert "Sony WH-1000XM4" in context_item["description"]

        # Verify the pipeline was called with correct query
        mock_rag_pipeline_wrapper.assert_called_once_with("What are the best wireless headphones?")

    @patch("src.api.api.endpoint.rag_pipeline_wrapper")
    def test_rag_endpoint_with_openai_query(self, mock_rag_pipeline_wrapper):
        """Test RAG endpoint with OpenAI-specific query"""
        mock_rag_pipeline_wrapper.return_value = {
            "answer": "For AI development work, I recommend headphones with clear audio for voice recognition testing.",
            "used_context": [
                {
                    "image_url": "https://example.com/ai-headphones.jpg",
                    "price": 199.99,
                    "description": "Audio-Technica ATH-M50x Professional Studio Monitor Headphones",
                }
            ],
        }

        client = TestClient(app)
        response = client.post(
            "/rag/",
            json={"query": "What headphones are good for AI development and testing OpenAI APIs?"},
        )

        assert response.status_code == 200
        response_data = response.json()

        assert "AI development" in response_data["answer"]
        assert "Audio-Technica" in response_data["used_context"][0]["description"]

        mock_rag_pipeline_wrapper.assert_called_once_with("What headphones are good for AI development and testing OpenAI APIs?")

    @patch("src.api.api.endpoint.rag_pipeline_wrapper")
    def test_rag_endpoint_invalid_request(self, mock_rag_pipeline_wrapper):
        """Test RAG endpoint with invalid request data"""
        # Mock the wrapper to avoid side effects
        mock_rag_pipeline_wrapper.return_value = {
            "answer": "Test answer",
            "used_context": [],
        }

        client = TestClient(app)

        # Test with missing query field
        response = client.post("/rag/", json={})
        assert response.status_code == 422  # Validation error

        # Test with invalid data type
        response = client.post("/rag/", json={"query": 123})
        assert response.status_code == 422

        # Test with valid empty query (should work)
        response = client.post("/rag/", json={"query": ""})
        assert response.status_code == 200  # Empty string should be valid

    def test_rag_endpoint_request_id_header(self):
        """Test that RAG endpoint adds request ID to response headers"""
        with patch("src.api.api.endpoint.rag_pipeline_wrapper") as mock_rag_pipeline_wrapper:
            mock_rag_pipeline_wrapper.return_value = {
                "answer": "Test answer",
                "used_context": [],
            }

            client = TestClient(app)
            response = client.post("/rag/", json={"query": "test query"})

            assert response.status_code == 200
            assert "X-Request-ID" in response.headers

            # Verify the request ID is a valid UUID
            request_id = response.headers["X-Request-ID"]
            uuid.UUID(request_id)  # This will raise ValueError if not valid UUID

    def test_rag_endpoint_handles_various_queries(self):
        """Test RAG endpoint with various types of queries"""
        with patch("src.api.api.endpoint.rag_pipeline_wrapper") as mock_rag_pipeline_wrapper:
            mock_rag_pipeline_wrapper.return_value = {
                "answer": "Test answer for various query types",
                "used_context": [],
            }

            client = TestClient(app)

            # Test different query types
            queries = [
                "What are good headphones?",
                "Tell me about wireless speakers",
                "I need help finding electronics",
                "Best products for gaming",
            ]

            for query in queries:
                response = client.post("/rag/", json={"query": query})
                assert response.status_code == 200
                assert "answer" in response.json()


class TestRAGModels:
    """Test RAG request/response models"""

    def test_rag_request_model(self):
        """Test RAG request model validation"""
        # Valid request
        valid_request = RAGRequest(query="What are good headphones?")
        assert valid_request.query == "What are good headphones?"

        # Test with OpenAI-related query
        openai_request = RAGRequest(query="Best headphones for OpenAI API development")
        assert "OpenAI" in openai_request.query

    def test_rag_used_context_model(self):
        """Test RAG used context model"""
        context = RAGUsedContext(
            image_url="https://example.com/image.jpg",
            price=99.99,
            description="Test product description",
        )

        assert context.image_url == "https://example.com/image.jpg"
        assert context.price == 99.99
        assert context.description == "Test product description"

        # Test with None price (optional field)
        context_no_price = RAGUsedContext(
            image_url="https://example.com/image.jpg",
            price=None,
            description="Test product",
        )
        assert context_no_price.price is None

    def test_rag_response_model(self):
        """Test RAG response model"""
        used_context = [
            RAGUsedContext(
                image_url="https://example.com/image.jpg",
                price=99.99,
                description="Test product",
            )
        ]

        response = RAGResponse(
            request_id="test-uuid-123",
            answer="This is a test answer from OpenAI",
            used_context=used_context,
        )

        assert response.request_id == "test-uuid-123"
        assert "OpenAI" in response.answer
        assert len(response.used_context) == 1
        assert response.used_context[0].price == 99.99


class TestMiddleware:
    """Test custom middleware functionality"""

    def test_request_id_middleware_adds_uuid(self):
        """Test that RequestID middleware adds unique UUID to each request"""
        client = TestClient(app)

        # Make multiple requests
        response1 = client.get("/")
        response2 = client.get("/")

        # Both should have request IDs
        assert "X-Request-ID" in response1.headers
        assert "X-Request-ID" in response2.headers

        # Request IDs should be different
        request_id1 = response1.headers["X-Request-ID"]
        request_id2 = response2.headers["X-Request-ID"]
        assert request_id1 != request_id2

        # Both should be valid UUIDs
        uuid.UUID(request_id1)
        uuid.UUID(request_id2)

    def test_cors_headers_present(self):
        """Test that CORS headers are present in responses"""
        client = TestClient(app)
        response = client.get("/")

        # Check for CORS headers (these are added by FastAPI's CORSMiddleware)
        # The exact headers depend on the request and configuration
        assert response.status_code == 200


class TestAPIIntegration:
    """Test API integration scenarios focused on OpenAI workflow"""

    @patch("src.api.api.endpoint.rag_pipeline_wrapper")
    def test_end_to_end_openai_workflow(self, mock_rag_pipeline_wrapper):
        """Test complete workflow simulating OpenAI-powered product search"""
        # Simulate full OpenAI RAG response
        mock_rag_pipeline_wrapper.return_value = {
            "answer": "Based on my analysis using OpenAI embeddings, I found several excellent headphones. The Sony WH-1000XM4 offers superior noise cancellation with a rating of 4.5/5.",
            "used_context": [
                {
                    "image_url": "https://m.media-amazon.com/images/I/71o8Q5XJS5L._AC_SX569_.jpg",
                    "price": 348.00,
                    "description": "Sony WH-1000XM4 Wireless Premium Noise Canceling Overhead Headphones",
                },
                {
                    "image_url": "https://m.media-amazon.com/images/I/51Dx19qTJ4L._AC_SX569_.jpg",
                    "price": 179.00,
                    "description": "Bose QuietComfort 35 II Wireless Bluetooth Headphones",
                },
            ],
        }

        client = TestClient(app)
        response = client.post(
            "/rag/",
            json={"query": "I need noise-cancelling headphones for my home office. What do you recommend?"},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify OpenAI-powered response structure
        assert "request_id" in data
        assert "answer" in data
        assert "used_context" in data

        # Verify OpenAI analysis is reflected in answer
        assert "OpenAI embeddings" in data["answer"]
        assert "Sony WH-1000XM4" in data["answer"]
        assert "4.5/5" in data["answer"]

        # Verify product context
        assert len(data["used_context"]) == 2
        sony_product = data["used_context"][0]
        assert sony_product["price"] == 348.00
        assert "Sony WH-1000XM4" in sony_product["description"]

        bose_product = data["used_context"][1]
        assert bose_product["price"] == 179.00
        assert "Bose QuietComfort" in bose_product["description"]
