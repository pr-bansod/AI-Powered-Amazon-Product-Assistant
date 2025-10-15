"""
Unit tests for API models.
"""
import pytest
from pydantic import ValidationError


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

        with pytest.raises(ValidationError):
            RAGResponse(answer="Test answer")  # Missing request_id and used_context
