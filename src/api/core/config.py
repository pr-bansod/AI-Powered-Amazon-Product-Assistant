from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    OPENAI_API_KEY: str
    GROQ_API_KEY: str

    API_URL:str = "https://api:8000"

    model_config = SettingsConfigDict(env_file=".env")

config = Config()