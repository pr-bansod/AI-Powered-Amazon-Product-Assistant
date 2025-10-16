import yaml
import logging
from typing import Optional
from jinja2 import Template, TemplateError
from langsmith import Client


logger = logging.getLogger(__name__)
ls_client = Client()


def prompt_template_config(yaml_file: str, prompt_key: str) -> Optional[Template]:
    """Load and return a Jinja2 template from a YAML configuration file.

    :param yaml_file: Path to the YAML configuration file
    :type yaml_file: str
    :param prompt_key: Key name of the prompt in the YAML config
    :type prompt_key: str
    :returns: Jinja2 Template object if successful, None if failed
    :rtype: Optional[Template]
    """
    try:
        logger.info(f"Loading prompt template from {yaml_file} with key '{prompt_key}'")

        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)

        if 'prompts' not in config:
            logger.error(f"No 'prompts' key found in {yaml_file}")
            return None

        if prompt_key not in config['prompts']:
            logger.error(f"Prompt key '{prompt_key}' not found in {yaml_file}")
            return None

        template_content = config['prompts'][prompt_key]
        template = Template(template_content)

        logger.info(f"Successfully loaded prompt template '{prompt_key}' from {yaml_file}")
        return template

    except FileNotFoundError as e:
        logger.error(f"YAML file not found: {yaml_file} - {e}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {yaml_file}: {e}")
        return None
    except TemplateError as e:
        logger.error(f"Error creating Jinja2 template from '{prompt_key}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading prompt template: {e}")
        return None


def prompt_template_registry(prompt_name: str) -> Optional[Template]:
    """Pull a prompt template from LangSmith registry and return as Jinja2 template.

    :param prompt_name: Name of the prompt in LangSmith registry
    :type prompt_name: str
    :returns: Jinja2 Template object if successful, None if failed
    :rtype: Optional[Template]
    """
    try:
        logger.info(f"Pulling prompt template '{prompt_name}' from LangSmith registry")

        prompt_commit = ls_client.pull_prompt(prompt_name)

        if not prompt_commit or not prompt_commit.messages:
            logger.error(f"No messages found in prompt '{prompt_name}'")
            return None

        template_content = prompt_commit.messages[0].prompt.template
        template = Template(template_content)

        logger.info(f"Successfully pulled prompt template '{prompt_name}' from registry")
        return template

    except Exception as e:
        logger.error(f"Error pulling prompt '{prompt_name}' from LangSmith registry: {e}")
        return None