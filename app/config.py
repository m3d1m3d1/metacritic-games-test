from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://games:games@localhost:5432/games"
    crawl_batch_size: int = 20  # games per hourly run
    crawl_interval_minutes: int = 60
    crawl_queue_poll_seconds: int = 5
    crawl_max_failures_per_day: int = 2
    metacritic_max_concurrency: int = Field(default=2, ge=1, le=4)
    timezone: str = "UTC"
    llm_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GROQ_API_KEY", "LLM_API_KEY", "OPENROUTER_API_KEY", "SILICONFLOW_API_KEY"
        ),
    )
    llm_api_base_url: str = "https://api.groq.com/openai/v1"
    summary_model: str = "openai/gpt-oss-120b"
    summary_fallback_models: str = (
        "google/gemma-4-26b-a4b-it:free,"
        "google/gemma-4-31b-it:free"
    )
    youtube_enrichment_enabled: bool = True
    demo_auth_username: str | None = None
    demo_auth_password: SecretStr | None = None

    @model_validator(mode="after")
    def validate_demo_auth(self):
        username = (self.demo_auth_username or "").strip()
        password = (
            self.demo_auth_password.get_secret_value()
            if self.demo_auth_password is not None
            else ""
        )
        if bool(username) != bool(password):
            raise ValueError(
                "DEMO_AUTH_USERNAME and DEMO_AUTH_PASSWORD must be configured together"
            )
        if username and ":" in username:
            raise ValueError("DEMO_AUTH_USERNAME must not contain a colon")
        if password and len(password) < 16:
            raise ValueError("DEMO_AUTH_PASSWORD must contain at least 16 characters")
        object.__setattr__(self, "demo_auth_username", username or None)
        object.__setattr__(
            self,
            "demo_auth_password",
            SecretStr(password) if password else None,
        )
        return self
    metacritic_proxy: str | None = None  # e.g. http://user:pass@host:port


settings = Settings()


def get_timezone() -> ZoneInfo:
    try:
        return ZoneInfo(settings.timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone configured in TIMEZONE: {settings.timezone}") from exc


def current_day() -> date:
    return datetime.now(get_timezone()).date()
