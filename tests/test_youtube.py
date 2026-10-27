import pytest

from app import youtube
from app.config import settings


def test_search_selects_most_viewed_non_live_result(monkeypatch):
    class FakeYoutubeDL:
        def __init__(self, options):
            assert options["skip_download"] is True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def extract_info(self, query, download):
            assert query == "ytsearch10:Example Game gameplay let's play"
            assert download is False
            return {
                "entries": [
                    {"id": "lowview00001", "title": "Small channel", "view_count": 50},
                    {"id": "livevideo001", "title": "Live stream", "view_count": 5000, "is_live": True},
                    {"id": "highview0001", "title": "Most viewed", "view_count": 900},
                    {"id": "noview000001", "title": "No view metadata"},
                ]
            }

    monkeypatch.setattr(youtube, "YoutubeDL", FakeYoutubeDL)

    found = youtube._search_most_viewed("Example Game")

    assert found is not None
    assert found.video_id == "highview0001"
    assert found.views == 900
    assert found.url == "https://www.youtube.com/watch?v=highview0001"


def test_letsplay_source_hash_depends_on_video_and_transcript():
    original = youtube.letsplay_source_hash("video1", "Transcript text")

    assert original == youtube.letsplay_source_hash("video1", "Transcript text")
    assert original != youtube.letsplay_source_hash("video2", "Transcript text")
    assert original != youtube.letsplay_source_hash("video1", "Updated transcript")


@pytest.mark.asyncio
async def test_summarize_letsplay_uses_groq_without_openrouter_parameters(monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "Автор отмечает сильный сюжет."}}]
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
    monkeypatch.setattr(youtube.httpx, "AsyncClient", FakeHTTPClient)

    conclusion = await youtube.summarize_letsplay("Example Game", "The story is great.")

    assert conclusion == "Автор отмечает сильный сюжет."
    assert calls[0][0] == "https://api.groq.com/openai/v1/chat/completions"
    assert calls[0][1]["model"] == "openai/gpt-oss-120b"
    assert "reasoning" not in calls[0][1]
