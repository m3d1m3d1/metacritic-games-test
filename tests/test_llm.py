import pytest

from app import llm
from app.config import settings
from app.scraper.parsers import Review, ReviewsPage


def test_review_source_hash_is_stable_and_tracks_review_changes():
    page = ReviewsPage(
        kind="critic",
        platform="pc",
        reviews=[Review(kind="critic", text="A great game", score=90, author="Example")],
    )
    same_page = ReviewsPage(
        kind="critic",
        platform="pc",
        reviews=[Review(kind="critic", text="A great game", score=90, author="Example")],
    )
    changed_page = ReviewsPage(
        kind="critic",
        platform="pc",
        reviews=[Review(kind="critic", text="A different view", score=90, author="Example")],
    )

    assert llm.review_source_hash(page) == llm.review_source_hash(same_page)
    assert llm.review_source_hash(page) != llm.review_source_hash(changed_page)


@pytest.mark.asyncio
async def test_summarize_reviews_uses_openai_compatible_chat_completions(monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "google/gemma-4-26b-a4b-it:free",
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"summary":"Critics praise the combat.",'
                                '"likes":["Responsive combat"],'
                                '"dislikes":["Repetitive levels"]}'
                            )
                        }
                    }
                ]
            }

    class FakeHTTPClient:
        def __init__(self, **kwargs):
            assert kwargs["headers"]["Authorization"] == "Bearer test-key"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, path, json):
            calls.append((path, json))
            return FakeResponse()

    monkeypatch.setattr(settings, "summary_model", "test-model:free")
    monkeypatch.setattr(
        settings,
        "summary_fallback_models",
        "google/gemma-4-26b-a4b-it:free,google/gemma-4-31b-it:free",
    )
    monkeypatch.setattr(settings, "llm_api_base_url", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeHTTPClient)

    result = await llm.summarize_reviews(
        ReviewsPage(
            kind="critic",
            platform="pc",
            reviews=[Review(kind="critic", text="The combat is responsive.", score=90)],
        )
    )

    assert result.summary == "Critics praise the combat."
    assert result.likes == ["Responsive combat"]
    assert result.dislikes == ["Repetitive levels"]
    assert result.model == "google/gemma-4-26b-a4b-it:free"
    assert calls[0][0] == "https://openrouter.ai/api/v1/chat/completions"
    assert calls[0][1]["models"] == [
        "test-model:free",
        "google/gemma-4-26b-a4b-it:free",
        "google/gemma-4-31b-it:free",
    ]
    assert all(model.endswith(":free") for model in calls[0][1]["models"])
    assert calls[0][1]["response_format"] == {"type": "json_object"}
    assert calls[0][1]["reasoning"] == {"effort": "low", "exclude": True}


@pytest.mark.asyncio
async def test_summarize_reviews_uses_groq_without_openrouter_parameters(monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "openai/gpt-oss-120b",
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"summary":"Critics praise the combat.",'
                                '"likes":["Responsive combat"],"dislikes":[]}'
                            )
                        }
                    }
                ],
            }

    class FakeHTTPClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, path, json):
            calls.append((path, json))
            return FakeResponse()

    monkeypatch.setattr(settings, "summary_model", "openai/gpt-oss-120b")
    monkeypatch.setattr(settings, "llm_api_base_url", "https://api.groq.com/openai/v1")
    monkeypatch.setattr(settings, "llm_api_key", "groq-test-key")
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeHTTPClient)

    result = await llm.summarize_reviews(
        ReviewsPage(
            kind="critic",
            platform="pc",
            reviews=[Review(kind="critic", text="Responsive combat.")],
        )
    )

    assert result.summary == "Critics praise the combat."
    assert calls[0][0] == "https://api.groq.com/openai/v1/chat/completions"
    assert calls[0][1]["model"] == "openai/gpt-oss-120b"
    assert calls[0][1]["response_format"] == {"type": "json_object"}
    assert "models" not in calls[0][1]
    assert "reasoning" not in calls[0][1]


@pytest.mark.asyncio
async def test_summarize_reviews_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "llm_api_key", None)

    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        await llm.summarize_reviews(
            ReviewsPage(
                kind="user",
                platform="pc",
                reviews=[Review(kind="user", text="Great story")],
            )
        )


@pytest.mark.asyncio
async def test_translate_game_content_uses_free_model_fallback(monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"description":"Описание игры.",'
                                '"letsplay_summary":"Автор хвалит игру.",'
                                '"summaries":[{"audience":"critic",'
                                '"summary":"Критикам нравится игра.",'
                                '"likes":["Боевая система"],"dislikes":[]}]}'
                            )
                        }
                    }
                ]
            }

    class FakeHTTPClient:
        def __init__(self, **kwargs):
            assert kwargs["headers"]["Authorization"].startswith("Bearer ")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, path, json):
            calls.append((path, json))
            return FakeResponse()

    monkeypatch.setattr(settings, "summary_model", "test-model:free")
    monkeypatch.setattr(
        settings,
        "summary_fallback_models",
        "google/gemma-4-26b-a4b-it:free",
    )
    monkeypatch.setattr(settings, "llm_api_base_url", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeHTTPClient)

    result = await llm.translate_game_content(
        description="Game description",
        letsplay_summary="Creator likes the game.",
        summaries=[
            {
                "audience": "critic",
                "summary": "Critics like it.",
                "likes": ["Combat"],
                "dislikes": [],
            }
        ],
    )

    assert result.description == "Описание игры."
    assert result.summaries[0].likes == ["Боевая система"]
    assert calls[0][1]["models"] == [
        "test-model:free",
        "google/gemma-4-26b-a4b-it:free",
    ]
    assert calls[0][1]["response_format"] == {"type": "json_object"}
    assert calls[0][1]["reasoning"] == {"effort": "low", "exclude": True}


@pytest.mark.asyncio
async def test_translate_game_content_uses_groq_without_openrouter_parameters(monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"description":"Описание игры.",'
                                '"letsplay_summary":null,"summaries":[]}'
                            )
                        }
                    }
                ]
            }

    class FakeHTTPClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, path, json):
            calls.append((path, json))
            return FakeResponse()

    monkeypatch.setattr(settings, "summary_model", "openai/gpt-oss-120b")
    monkeypatch.setattr(settings, "llm_api_base_url", "https://api.groq.com/openai/v1")
    monkeypatch.setattr(settings, "llm_api_key", "groq-test-key")
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeHTTPClient)

    result = await llm.translate_game_content(
        description="Game description",
        letsplay_summary=None,
        summaries=[],
    )

    assert result.description == "Описание игры."
    assert calls[0][0] == "https://api.groq.com/openai/v1/chat/completions"
    assert calls[0][1]["model"] == "openai/gpt-oss-120b"
    assert "models" not in calls[0][1]
    assert "reasoning" not in calls[0][1]


@pytest.mark.asyncio
async def test_openrouter_fallback_rejects_paid_models(monkeypatch):
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_api_base_url", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(settings, "summary_model", "nvidia/nemotron:free")
    monkeypatch.setattr(settings, "summary_fallback_models", "some/provider-model")

    with pytest.raises(ValueError, match="only supports free models"):
        await llm.summarize_reviews(
            ReviewsPage(
                kind="critic",
                platform="pc",
                reviews=[Review(kind="critic", text="The story is good.")],
            )
        )


@pytest.mark.asyncio
async def test_openrouter_fallback_rejects_more_than_three_models(monkeypatch):
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_api_base_url", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(settings, "summary_model", "primary:free")
    monkeypatch.setattr(
        settings,
        "summary_fallback_models",
        "fallback-one:free,fallback-two:free,fallback-three:free",
    )

    with pytest.raises(ValueError, match="at most three models"):
        await llm.translate_game_content(
            description="Game description",
            letsplay_summary=None,
            summaries=[],
        )


@pytest.mark.asyncio
async def test_summarize_reviews_retries_empty_model_content(monkeypatch):
    calls = []
    payloads = iter(
        [
            {"choices": [{"message": {"content": None}, "finish_reason": "length"}]},
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"summary":"Critics praise the combat.",'
                                '"likes":["Combat"],"dislikes":[]}'
                            )
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        ]
    )

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return next(payloads)

    class FakeHTTPClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, path, json):
            calls.append((path, json))
            return FakeResponse()

    async def no_wait(delay):
        return None

    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_api_base_url", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(settings, "summary_model", "primary:free")
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeHTTPClient)
    monkeypatch.setattr(llm.asyncio, "sleep", no_wait)

    result = await llm.summarize_reviews(
        ReviewsPage(
            kind="critic",
            platform="pc",
            reviews=[Review(kind="critic", text="The combat is responsive.")],
        )
    )

    assert len(calls) == 2
    assert result.summary == "Critics praise the combat."


@pytest.mark.asyncio
async def test_summarize_reviews_defers_rate_limit_without_immediate_retries(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 429
        headers = {"Retry-After": "120"}

        def raise_for_status(self):
            request = llm.httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            response = llm.httpx.Response(429, headers=self.headers, request=request)
            raise llm.httpx.HTTPStatusError("rate limited", request=request, response=response)

    class FakeHTTPClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, path, json):
            calls.append(path)
            return FakeResponse()

    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeHTTPClient)

    with pytest.raises(llm.ReviewSummaryRateLimited) as exc_info:
        await llm.summarize_reviews(
            ReviewsPage(
                kind="critic",
                platform="pc",
                reviews=[Review(kind="critic", text="Good combat.")],
            )
        )

    assert len(calls) == 1
    assert exc_info.value.retry_after_seconds == 120


def test_parse_summary_response_rejects_empty_content_with_finish_reason():
    with pytest.raises(ValueError, match="finish_reason=length"):
        llm._parse_summary_response(
            {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}
        )
