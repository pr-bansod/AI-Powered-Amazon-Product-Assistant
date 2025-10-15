"""
Unit tests for prompt management utilities.
"""
import pytest
from unittest.mock import Mock, patch, mock_open
from jinja2 import Template


@pytest.mark.unit
class TestPromptTemplateConfig:
    """Tests for prompt_template_config function"""

    @patch('builtins.open', new_callable=mock_open, read_data="""
prompts:
  retrieval_generation: |
    You are a helpful assistant.
    Context: {{ preprocessed_context }}
    Question: {{ question }}
    Answer:
  test_prompt: |
    Test prompt with {{ variable }}
""")
    @patch('api.rag.utils.prompt_management.yaml.safe_load')
    def test_prompt_template_config_success(self, mock_yaml_load, mock_file):
        """Test successful prompt template loading from YAML"""
        from api.rag.utils.prompt_management import prompt_template_config

        # Setup mock YAML data
        mock_yaml_load.return_value = {
            'prompts': {
                'retrieval_generation': 'Context: {{ preprocessed_context }}\nQuestion: {{ question }}',
                'test_prompt': 'Test prompt with {{ variable }}'
            }
        }

        template = prompt_template_config("test.yaml", "retrieval_generation")

        assert isinstance(template, Template)
        mock_file.assert_called_once_with("test.yaml", 'r')
        mock_yaml_load.assert_called_once()

    @patch('builtins.open', new_callable=mock_open, read_data="""
prompts:
  retrieval_generation: |
    Context: {{ preprocessed_context }}
    Question: {{ question }}
""")
    @patch('api.rag.utils.prompt_management.yaml.safe_load')
    def test_prompt_template_config_rendering(self, mock_yaml_load, mock_file):
        """Test template rendering with variables"""
        from api.rag.utils.prompt_management import prompt_template_config

        mock_yaml_load.return_value = {
            'prompts': {
                'retrieval_generation': 'Context: {{ preprocessed_context }}\nQuestion: {{ question }}'
            }
        }

        template = prompt_template_config("test.yaml", "retrieval_generation")
        rendered = template.render(
            preprocessed_context="Test context",
            question="Test question"
        )

        assert "Test context" in rendered
        assert "Test question" in rendered

    @patch('builtins.open', new_callable=mock_open)
    @patch('api.rag.utils.prompt_management.yaml.safe_load')
    def test_prompt_template_config_missing_key(self, mock_yaml_load, mock_file):
        """Test handling of missing prompt key"""
        from api.rag.utils.prompt_management import prompt_template_config

        mock_yaml_load.return_value = {
            'prompts': {
                'other_prompt': 'Some other prompt'
            }
        }

        with pytest.raises(KeyError):
            prompt_template_config("test.yaml", "missing_prompt")


@pytest.mark.unit
@pytest.mark.requires_api
class TestPromptTemplateRegistry:
    """Tests for prompt_template_registry function"""

    @patch('api.rag.utils.prompt_management.ls_client.pull_prompt')
    def test_prompt_template_registry_success(self, mock_pull_prompt):
        """Test successful prompt template pulling from LangSmith"""
        from api.rag.utils.prompt_management import prompt_template_registry

        # Setup mock LangSmith response
        mock_message = Mock()
        mock_message.prompt.template = "LangSmith template with {{ variable }}"
        mock_pull_prompt.return_value.messages = [mock_message]

        template = prompt_template_registry("test-prompt")

        assert isinstance(template, Template)
        mock_pull_prompt.assert_called_once_with("test-prompt")

    @patch('api.rag.utils.prompt_management.ls_client.pull_prompt')
    def test_prompt_template_registry_rendering(self, mock_pull_prompt):
        """Test template rendering from registry"""
        from api.rag.utils.prompt_management import prompt_template_registry

        mock_message = Mock()
        mock_message.prompt.template = "Hello {{ name }}!"
        mock_pull_prompt.return_value.messages = [mock_message]

        template = prompt_template_registry("greeting-prompt")
        rendered = template.render(name="World")

        assert rendered == "Hello World!"


@pytest.mark.unit
class TestPromptTemplateVariables:
    """Tests for prompt template variable handling"""

    @patch('builtins.open', new_callable=mock_open)
    @patch('api.rag.utils.prompt_management.yaml.safe_load')
    def test_template_with_multiple_variables(self, mock_yaml_load, mock_file):
        """Test template with multiple variables"""
        from api.rag.utils.prompt_management import prompt_template_config

        mock_yaml_load.return_value = {
            'prompts': {
                'complex_prompt': 'Var1: {{ var1 }}, Var2: {{ var2 }}, Var3: {{ var3 }}'
            }
        }

        template = prompt_template_config("test.yaml", "complex_prompt")
        rendered = template.render(var1="A", var2="B", var3="C")

        assert "Var1: A" in rendered
        assert "Var2: B" in rendered
        assert "Var3: C" in rendered

    @patch('builtins.open', new_callable=mock_open)
    @patch('api.rag.utils.prompt_management.yaml.safe_load')
    def test_template_with_missing_variable(self, mock_yaml_load, mock_file):
        """Test template rendering with missing variable"""
        from api.rag.utils.prompt_management import prompt_template_config

        mock_yaml_load.return_value = {
            'prompts': {
                'test_prompt': 'Value: {{ missing_var }}'
            }
        }

        template = prompt_template_config("test.yaml", "test_prompt")
        rendered = template.render()  # Missing variable should render as empty

        assert "Value:" in rendered


@pytest.mark.integration
class TestPromptManagementIntegration:
    """Integration tests for prompt management"""

    def test_yaml_file_structure(self):
        """Test actual YAML prompt file structure"""
        import yaml
        from pathlib import Path

        yaml_path = Path("src/api/rag/prompts/retrieval_generation.yaml")

        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)

            assert 'prompts' in config
            assert 'retrieval_generation' in config['prompts']
            assert isinstance(config['prompts']['retrieval_generation'], str)

    @patch('builtins.open', new_callable=mock_open)
    @patch('api.rag.utils.prompt_management.yaml.safe_load')
    def test_end_to_end_prompt_loading(self, mock_yaml_load, mock_file):
        """Test end-to-end prompt loading and rendering"""
        from api.rag.utils.prompt_management import prompt_template_config

        mock_yaml_load.return_value = {
            'prompts': {
                'retrieval_generation': '''You are a helpful assistant.

Context:
{{ preprocessed_context }}

Question: {{ question }}

Please provide a detailed answer based on the context.'''
            }
        }

        template = prompt_template_config("test.yaml", "retrieval_generation")

        context = "Product A: $29.99\nProduct B: $49.99"
        question = "What products are available?"

        rendered = template.render(
            preprocessed_context=context,
            question=question
        )

        assert "Product A" in rendered
        assert "Product B" in rendered
        assert "What products are available?" in rendered
        assert "helpful assistant" in rendered
