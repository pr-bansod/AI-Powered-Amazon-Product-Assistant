from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    OPENAI_API_KEY: str
    GROQ_API_KEY: str
    LANGSMITH_API_KEY: str
    LANGSMITH_TRACING: bool = True 
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_PROJECT : str = "rag-tracing"


    model_config = SettingsConfigDict(env_file=".env")

config = Config()