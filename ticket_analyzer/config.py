from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    openai_api_key: str = ""
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    max_retries: int = 3
    log_level: str = "INFO"
    app_env: str = "development"


settings = Settings()
