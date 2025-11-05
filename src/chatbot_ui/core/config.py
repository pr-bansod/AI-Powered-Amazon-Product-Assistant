from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    OPENAI_API_KEY: str
    GROQ_API_KEY: str
    GOOGLE_API_KEY: str | None = None

    API_URL: str = "http://api:8000"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


config = Config()
