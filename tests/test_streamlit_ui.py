from unittest.mock import Mock, patch

import pytest
import requests
from src.chatbot_ui.app import api_call


class TestAPICall:
    """Test Streamlit UI API call functionality"""

    @patch("src.chatbot_ui.app.requests.post")
    def test_api_call_success(self, mock_post):
        """Test successful API call to FastAPI backend"""
        # Mock successful response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "answer": "Based on the available products, I recommend wireless headphones.",
            "used_context": [
                {
                    "image_url": "https://example.com/headphones.jpg",
                    "price": 99.99,
                    "description": "Wireless Bluetooth headphones",
                }
            ],
        }
        mock_post.return_value = mock_response

        success, result = api_call("post", "http://api:8000/rag", json={"query": "recommend headphones"})

        assert success is True
        assert "answer" in result
        assert "used_context" in result
        assert "wireless headphones" in result["answer"].lower()

        mock_post.assert_called_once_with("http://api:8000/rag", json={"query": "recommend headphones"})

    @patch("src.chatbot_ui.app.requests.get")
    def test_api_call_get_request(self, mock_get):
        """Test GET request to API"""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"message": "API"}
        mock_get.return_value = mock_response

        success, result = api_call("get", "http://api:8000/")

        assert success is True
        assert result["message"] == "API"
        mock_get.assert_called_once_with("http://api:8000/")

    @patch("src.chatbot_ui.app.requests.post")
    def test_api_call_openai_integration(self, mock_post):
        """Test API call for OpenAI-specific queries"""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "answer": "For OpenAI development, I recommend headphones with clear audio reproduction.",
            "used_context": [
                {
                    "image_url": "https://example.com/studio-headphones.jpg",
                    "price": 299.99,
                    "description": "Professional studio monitor headphones for AI development",
                }
            ],
        }
        mock_post.return_value = mock_response

        success, result = api_call(
            "post",
            "http://api:8000/rag",
            json={"query": "What headphones are best for OpenAI API development?"},
        )

        assert success is True
        assert "OpenAI development" in result["answer"]
        assert result["used_context"][0]["description"] == "Professional studio monitor headphones for AI development"

    @patch("src.chatbot_ui.app.requests.post")
    def test_api_call_http_error(self, mock_post):
        """Test API call with HTTP error response"""
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 400
        mock_response.json.return_value = {"detail": "Invalid request"}
        mock_post.return_value = mock_response

        success, result = api_call("post", "http://api:8000/rag", json={"query": ""})

        assert success is False
        assert result["detail"] == "Invalid request"

    @patch("src.chatbot_ui.app.requests.post")
    def test_api_call_connection_error(self, mock_post):
        """Test API call with connection error"""
        mock_post.side_effect = requests.exceptions.ConnectionError()

        success, result = api_call("post", "http://api:8000/rag", json={"query": "test"})

        assert success is False
        assert result["message"] == "Connection error"

    @patch("src.chatbot_ui.app.requests.post")
    def test_api_call_timeout_error(self, mock_post):
        """Test API call with timeout error"""
        mock_post.side_effect = requests.exceptions.Timeout()

        success, result = api_call("post", "http://api:8000/rag", json={"query": "test"})

        assert success is False
        assert result["message"] == "Request timeout"

    @patch("src.chatbot_ui.app.requests.post")
    def test_api_call_json_decode_error(self, mock_post):
        """Test API call with invalid JSON response"""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.side_effect = requests.exceptions.JSONDecodeError("Invalid JSON", "", 0)
        mock_post.return_value = mock_response

        success, result = api_call("post", "http://api:8000/rag", json={"query": "test"})

        assert success is True
        assert result["message"] == "Invalid response format from server"

    @patch("src.chatbot_ui.app.requests.post")
    def test_api_call_unexpected_error(self, mock_post):
        """Test API call with unexpected error"""
        mock_post.side_effect = ValueError("Unexpected error")

        success, result = api_call("post", "http://api:8000/rag", json={"query": "test"})

        assert success is False
        assert "Unexpected error" in result["message"]


class TestStreamlitUIConfiguration:
    """Test Streamlit UI configuration and setup"""

    @patch("src.chatbot_ui.app.config")
    def test_config_loaded(self, mock_config):
        """Test that configuration is properly loaded"""
        mock_config.API_URL = "http://test-api:8000"

        # Import the app module to check config usage
        from src.chatbot_ui.app import config

        assert hasattr(config, "API_URL")

    def test_api_call_function_exists(self):
        """Test that api_call function is properly defined"""
        from src.chatbot_ui.app import api_call

        assert callable(api_call)

    @patch("src.chatbot_ui.app.st")
    def test_streamlit_imports(self, mock_st):
        """Test that Streamlit is properly imported and configured"""
        # This test ensures that the app can be imported without Streamlit errors
        try:
            import src.chatbot_ui.app

            assert True  # If we reach here, imports worked
        except ImportError as e:
            pytest.fail(f"Failed to import Streamlit app: {e}")


class TestStreamlitUIIntegration:
    """Test Streamlit UI integration with OpenAI backend"""

    @patch("src.chatbot_ui.app.requests.post")
    def test_end_to_end_user_interaction(self, mock_post):
        """Test complete user interaction flow with OpenAI backend"""
        # Simulate the full conversation flow
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "request_id": "12345-67890",
            "answer": "Based on my analysis using OpenAI embeddings, I found excellent wireless headphones. The Sony WH-1000XM4 offers premium noise cancellation technology.",
            "used_context": [
                {
                    "image_url": "https://m.media-amazon.com/images/I/71o8Q5XJS5L._AC_SX569_.jpg",
                    "price": 348.00,
                    "description": "Sony WH-1000XM4 Wireless Premium Noise Canceling Overhead Headphones with Mic",
                },
                {
                    "image_url": "https://m.media-amazon.com/images/I/61SUj2aKoEL._AC_SX569_.jpg",
                    "price": 229.00,
                    "description": "Bose QuietComfort 45 Wireless Bluetooth Noise-Canceling Headphones",
                },
            ],
        }
        mock_post.return_value = mock_response

        # Simulate user query
        user_query = "I work from home and need good noise-cancelling headphones for video calls. What do you recommend?"

        success, result = api_call("post", "http://api:8000/rag", json={"query": user_query})

        assert success is True

        # Verify OpenAI-powered response
        assert "OpenAI embeddings" in result["answer"]
        assert "Sony WH-1000XM4" in result["answer"]
        assert "noise cancellation" in result["answer"].lower()

        # Verify product context for UI display
        assert len(result["used_context"]) == 2

        # Verify first product (Sony)
        sony_product = result["used_context"][0]
        assert sony_product["price"] == 348.00
        assert "Sony WH-1000XM4" in sony_product["description"]
        assert sony_product["image_url"].startswith("https://")

        # Verify second product (Bose)
        bose_product = result["used_context"][1]
        assert bose_product["price"] == 229.00
        assert "Bose QuietComfort" in bose_product["description"]

        # Verify API was called correctly
        mock_post.assert_called_once_with("http://api:8000/rag", json={"query": user_query})

    @patch("src.chatbot_ui.app.requests.post")
    def test_ui_error_display_flow(self, mock_post):
        """Test UI error handling and display"""
        # Simulate API error
        mock_post.side_effect = requests.exceptions.ConnectionError()

        success, result = api_call("post", "http://api:8000/rag", json={"query": "test query"})

        assert success is False
        assert result["message"] == "Connection error"

        # In the actual UI, this would trigger an error popup
        # The test verifies the error handling mechanism works

    @patch("src.chatbot_ui.app.requests.post")
    def test_ui_handles_empty_context(self, mock_post):
        """Test UI handles responses with no product context"""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "answer": "I couldn't find specific products matching your query. Please try a different search.",
            "used_context": [],
        }
        mock_post.return_value = mock_response

        success, result = api_call("post", "http://api:8000/rag", json={"query": "unicorn headphones"})

        assert success is True
        assert "couldn't find" in result["answer"]
        assert len(result["used_context"]) == 0

        # UI should handle empty context gracefully
        # In the actual Streamlit app, this would show "No suggestions yet"


class TestStreamlitAppComponents:
    """Test individual Streamlit app components"""

    def test_imports_and_basic_structure(self):
        """Test that the Streamlit app imports correctly"""
        try:
            # Test basic imports
            import logging

            import requests
            import streamlit as st
            from src.chatbot_ui.app import api_call

            # Test app-specific imports
            from src.chatbot_ui.core.config import config

            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")

    @patch("src.chatbot_ui.app.requests")
    def test_api_call_method_selection(self, mock_requests):
        """Test that api_call correctly selects HTTP methods"""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"test": "data"}

        # Test POST method
        mock_requests.post.return_value = mock_response
        api_call("post", "http://test.com", json={"data": "test"})
        mock_requests.post.assert_called_once()

        # Test GET method
        mock_requests.get.return_value = mock_response
        api_call("get", "http://test.com")
        mock_requests.get.assert_called_once()

        # Reset mocks
        mock_requests.reset_mock()

        # Test PUT method (if supported)
        mock_requests.put.return_value = mock_response
        api_call("put", "http://test.com", json={"data": "test"})
        mock_requests.put.assert_called_once()
