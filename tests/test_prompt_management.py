import os
import tempfile
from unittest.mock import Mock, patch

import pytest
import yaml
from jinja2 import Template
from src.api.rag.utils.prompt_management import (
    prompt_template_config,
    prompt_template_registry,
)


class TestPromptTemplateConfig:
    """Test prompt template configuration loading from YAML files"""

    def test_load_prompt_from_yaml(self, sample_yaml_config):
        """Test loading a prompt template from YAML configuration"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_yaml_config, f)
            f.flush()

            try:
                template = prompt_template_config(f.name, "test_prompt")
                assert isinstance(template, Template)

                # Test template rendering
                rendered = template.render(question="What is AI?")
                expected = "You are a helpful assistant. Question: What is AI?"
                assert rendered == expected

            finally:
                os.unlink(f.name)

    def test_load_retrieval_generation_prompt(self):
        """Test loading the actual retrieval generation prompt"""
        yaml_path = "src/api/rag/prompts/retrieval_generation.yaml"
        template = prompt_template_config(yaml_path, "retrieval_generation")

        assert isinstance(template, Template)

        # Test rendering with sample data
        context = "- B001: Wireless headphones\n- B002: Bluetooth speaker"
        question = "What audio products are available?"

        rendered = template.render(preprocessed_context=context, question=question)

        assert "shopping assistant" in rendered.lower()
        assert "Wireless headphones" in rendered
        assert "What audio products are available?" in rendered

    def test_template_missing_file(self):
        """Test error handling when YAML file doesn't exist"""
        with pytest.raises(FileNotFoundError):
            prompt_template_config("nonexistent.yaml", "test_prompt")

    def test_template_missing_key(self, sample_yaml_config):
        """Test error handling when prompt key doesn't exist"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_yaml_config, f)
            f.flush()

            try:
                with pytest.raises(KeyError):
                    prompt_template_config(f.name, "nonexistent_prompt")
            finally:
                os.unlink(f.name)

    def test_template_malformed_yaml(self):
        """Test error handling when YAML file is malformed"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()

            try:
                with pytest.raises(yaml.YAMLError):
                    prompt_template_config(f.name, "test_prompt")
            finally:
                os.unlink(f.name)


class TestPromptTemplateRegistry:
    """Test prompt template registry functionality with LangSmith"""

    @patch("src.api.rag.utils.prompt_management.ls_client")
    def test_load_prompt_from_registry(self, mock_ls_client):
        """Test loading a prompt from LangSmith registry"""
        # Mock the LangSmith client response
        mock_prompt = Mock()
        mock_prompt.messages = [Mock()]
        mock_prompt.messages[0].prompt.template = "Registry template: {{ question }}"
        mock_ls_client.pull_prompt.return_value = mock_prompt

        template = prompt_template_registry("test-prompt")

        assert isinstance(template, Template)
        mock_ls_client.pull_prompt.assert_called_once_with("test-prompt")

        # Test template rendering
        rendered = template.render(question="Test question")
        assert rendered == "Registry template: Test question"

    @patch("src.api.rag.utils.prompt_management.ls_client")
    def test_registry_client_error(self, mock_ls_client):
        """Test error handling when LangSmith client fails"""
        mock_ls_client.pull_prompt.side_effect = Exception("API Error")

        with pytest.raises(Exception) as exc_info:
            prompt_template_registry("test-prompt")

        assert "API Error" in str(exc_info.value)


class TestPromptIntegration:
    """Test integration scenarios for prompt management"""

    def test_jinja2_template_features(self, sample_yaml_config):
        """Test that Jinja2 template features work correctly"""
        config_with_features = {
            "prompts": {
                "advanced_prompt": """
                {% if context %}
                Context: {{ context }}
                {% endif %}
                Question: {{ question }}
                {% for item in items %}
                - Item {{ loop.index }}: {{ item }}
                {% endfor %}
                """
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_with_features, f)
            f.flush()

            try:
                template = prompt_template_config(f.name, "advanced_prompt")

                rendered = template.render(
                    context="Product information",
                    question="What products are available?",
                    items=["Product A", "Product B"],
                )

                assert "Context: Product information" in rendered
                assert "Question: What products are available?" in rendered
                assert "Item 1: Product A" in rendered
                assert "Item 2: Product B" in rendered

            finally:
                os.unlink(f.name)

    def test_template_with_openai_context(self):
        """Test template rendering with OpenAI-specific context"""
        yaml_path = "src/api/rag/prompts/retrieval_generation.yaml"
        template = prompt_template_config(yaml_path, "retrieval_generation")

        # Simulate real RAG context with OpenAI embeddings
        openai_context = """- B001EXAMPLE:, rating:4.5, description:Sony WH-1000XM4 Wireless Noise Canceling Headphones
- B002EXAMPLE:, rating:4.3, description:Apple AirPods Pro (2nd Generation) with Noise Cancellation"""

        openai_question = "Which wireless headphones have the best noise cancellation?"

        rendered = template.render(preprocessed_context=openai_context, question=openai_question)

        # Verify OpenAI context is properly embedded
        assert "Sony WH-1000XM4" in rendered
        assert "Apple AirPods Pro" in rendered
        assert "rating:4.5" in rendered
        assert "noise cancellation" in rendered.lower()
        assert "shopping assistant" in rendered.lower()
