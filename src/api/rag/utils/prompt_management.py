import yaml
from jinja2 import Template
from langsmith import Client

ls_client = Client()


def prompt_template_config(yaml_file: str, prompt_key: str) -> Template:
    """
    Loads a Jinja2 template from a YAML configuration file.

    This function reads a YAML file that contains prompt templates, extracts
    a specific template by key, and returns it as a compiled Jinja2 Template
    object ready for rendering.

    Args:
        yaml_file (str): Path to the YAML configuration file containing prompt templates.
        prompt_key (str): Key identifying the specific prompt template to retrieve
            from the 'prompts' section of the YAML file.

    Returns:
        Template: A compiled Jinja2 Template object that can be rendered with
            context variables using the `.render()` method.

    Raises:
        FileNotFoundError: If the YAML file does not exist at the specified path.
        KeyError: If the prompt_key is not found in the 'prompts' section.
        yaml.YAMLError: If the YAML file is malformed or cannot be parsed.

    Example:
        >>> template = prompt_template_config("prompts/rag.yaml", "retrieval_generation")
        >>> rendered = template.render(context="Product info...", question="What is X?")
    """
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    template_content = config['prompts'][prompt_key]

    template = Template(template_content)

    return template


def prompt_template_registry(prompt_name: str) -> Template:
    """
    Retrieves a Jinja2 template from LangSmith's prompt registry.

    This function fetches a prompt template from the LangSmith prompt registry
    by name, extracts the template content, and returns it as a compiled
    Jinja2 Template object ready for rendering.

    Args:
        prompt_name (str): The name/identifier of the prompt in the LangSmith
            registry (e.g., "retrieval-generation").

    Returns:
        Template: A compiled Jinja2 Template object that can be rendered with
            context variables using the `.render()` method.
    """
    template_content = ls_client.pull_prompt(prompt_name).messages[0].prompt.template
    template = Template(template_content)
    return template

