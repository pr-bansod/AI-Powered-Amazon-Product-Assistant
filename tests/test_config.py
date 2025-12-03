import os
from unittest.mock import patch

from src.api.core.config import Config as APIConfig
from src.chatbot_ui.core.config import Config as UIConfig


class TestAPIConfig:
    """Test API configuration management - focusing on OpenAI integration"""

    def test_config_with_openai_key(self, test_env_vars):
        """Test config loads successfully with OpenAI API key"""
        config = APIConfig()

        assert config.OPENAI_API_KEY == "test_openai_key"
        assert isinstance(config.OPENAI_API_KEY, str)
        assert len(config.OPENAI_API_KEY) > 0

    def test_openai_key_field_exists(self, test_env_vars):
        """Test that config has OpenAI API key field"""
        config = APIConfig()

        assert hasattr(config, "OPENAI_API_KEY")
        assert isinstance(config.OPENAI_API_KEY, str)

    def test_config_extra_fields_ignored(self, test_env_vars):
        """Test that config ignores extra environment variables"""
        extra_env = {**test_env_vars, "RANDOM_VAR": "random_value"}

        with patch.dict(os.environ, extra_env):
            config = APIConfig()
            assert not hasattr(config, "RANDOM_VAR")


class TestUIConfig:
    """Test UI configuration management - OpenAI"""

    def test_ui_config_with_openai_key(self, test_env_vars):
        """Test UI config loads successfully with OpenAI API key"""
        config = UIConfig()

        # OpenAI is the main API being used
        assert config.OPENAI_API_KEY == "test_openai_key"
        assert isinstance(config.OPENAI_API_KEY, str)
        assert len(config.OPENAI_API_KEY) > 0

    def test_ui_config_with_custom_api_url(self, test_env_vars):
        """Test UI config with custom API_URL"""
        custom_env = {**test_env_vars, "API_URL": "http://localhost:8000"}

        with patch.dict(os.environ, custom_env):
            config = UIConfig()
            assert config.API_URL == "http://localhost:8000"

    def test_ui_config_default_values(self, test_env_vars):
        """Test UI config has proper default values"""
        config = UIConfig()

        # Test default API URL
        assert config.API_URL == "http://api:8000"

        # Test that OpenAI is properly configured
        assert config.OPENAI_API_KEY == "test_openai_key"


class TestConfigIntegration:
    """Test config integration scenarios - OpenAI focus"""

    def test_both_configs_share_openai_key(self, test_env_vars):
        """Test that both API and UI configs share OpenAI key"""
        api_config = APIConfig()
        ui_config = UIConfig()

        # Both should have the same OpenAI key
        assert api_config.OPENAI_API_KEY == ui_config.OPENAI_API_KEY
        assert api_config.OPENAI_API_KEY == "test_openai_key"

    def test_config_structure_differences(self, test_env_vars):
        """Test that API and UI configs have appropriate fields"""
        api_config = APIConfig()
        ui_config = UIConfig()

        # Both have OpenAI
        assert hasattr(api_config, "OPENAI_API_KEY")
        assert hasattr(ui_config, "OPENAI_API_KEY")

        # UI config has additional fields for frontend
        assert hasattr(ui_config, "API_URL")

        # API config has Cohere for reranking
        assert hasattr(api_config, "CO_API_KEY")

    def test_config_env_file_setting(self, test_env_vars):
        """Test that configs are set to read from .env file"""
        api_config = APIConfig()
        ui_config = UIConfig()

        assert api_config.model_config["env_file"] == ".env"
        assert ui_config.model_config["env_file"] == ".env"
