from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AliasChoices, Field
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
            "LLM_API_KEY", "OPENROUTER_API_KEY", "SILICONFLOW_API_KEY"
        ),
    )
    llm_api_base_url: str = "https://openrouter.ai/api/v1"
    summary_model: str = "nvidia/nemotron-3.5-super-vl:free"
    summary_fallback_models: str = (
        "google/gemma-4-26b-a4b-it:free,"
        "google/gemma-4-31b-it:free"
    )
    youtube_enrichment_enabled: bool = True
    metacritic_proxy: str | None = None  # e.g. http://user:pass@host:port


settings = Settings()


def get_timezone() -> ZoneInfo:
    try:
        return ZoneInfo(settings.timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone configured in TIMEZONE: {settings.timezone}") from exc


def current_day() -> date:
    return datetime.now(get_timezone()).date()
