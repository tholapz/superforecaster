from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    app_name: str = "superforecaster"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # Server
    port: int = 8000

    # Required
    anthropic_api_key: str
    database_url: str

    # LLM
    llm_model: str = "claude-sonnet-4-20250514"
    context_max_tokens: int = 1500

    # Forecasting
    persona_count: int = 6
    trim_alpha: float = 0.10
    extremize_beta: float = 2.5
    min_quorum: int = 3

    # Scheduler
    rerun_interval_hours: int = 24

    # API auth
    api_key: str = "changeme"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
