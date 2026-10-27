import pytest
from pydantic import SecretStr, ValidationError

from app.config import Settings


def test_demo_auth_is_optional_for_local_development():
    settings = Settings(_env_file=None)
    assert settings.demo_auth_username is None
    assert settings.demo_auth_password is None


def test_demo_auth_requires_both_credentials():
    with pytest.raises(ValidationError, match="must be configured together"):
        Settings(_env_file=None, demo_auth_username="reviewer")

    with pytest.raises(ValidationError, match="must be configured together"):
        Settings(
            _env_file=None,
            demo_auth_password=SecretStr("a-strong-password-123"),
        )


@pytest.mark.parametrize(
    ("username", "password", "error"),
    [
        ("bad:name", "a-strong-password-123", "must not contain a colon"),
        ("reviewer", "too-short", "at least 16 characters"),
    ],
)
def test_demo_auth_rejects_invalid_credentials(username, password, error):
    with pytest.raises(ValidationError, match=error):
        Settings(
            _env_file=None,
            demo_auth_username=username,
            demo_auth_password=SecretStr(password),
        )


def test_demo_auth_password_is_kept_secret():
    settings = Settings(
        _env_file=None,
        demo_auth_username="reviewer",
        demo_auth_password=SecretStr("a-strong-password-123"),
    )
    assert settings.demo_auth_password.get_secret_value() == "a-strong-password-123"
    assert "a-strong-password-123" not in repr(settings)


def test_groq_is_the_default_llm_provider():
    settings = Settings(_env_file=None)

    assert settings.llm_api_base_url == "https://api.groq.com/openai/v1"
    assert settings.summary_model == "openai/gpt-oss-120b"


def test_groq_api_key_takes_precedence_over_legacy_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.setenv("LLM_API_KEY", "legacy-test-key")
    settings = Settings(_env_file=None)

    assert settings.llm_api_key == "groq-test-key"
