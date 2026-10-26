from datetime import date, datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app.db.session import get_session
from app.main import _worker_payload, app


@pytest.fixture
def client():
    class FakeSession:
        async def commit(self):
            return None

    async def fake_session():
        yield FakeSession()

    app.dependency_overrides[get_session] = fake_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_dashboard_shows_empty_state(client, monkeypatch):
    async def no_runs(session, limit=20):
        return []

    monkeypatch.setattr("app.main.repo.recent_runs", no_runs)
    monkeypatch.setattr("app.main.repo.worker_statuses", _async_workers([]))
    monkeypatch.setattr("app.main.repo.recent_worker_logs", _async_logs([]))
    monkeypatch.setattr("app.main.repo.pending_review_summary_stats", _async_summary_stats())

    response = client.get("/monitor")

    assert response.status_code == 200
    assert "Запусков пока нет." in response.text
    assert "Запустить сбор сейчас" in response.text
    assert "EventSource" in response.text
    assert "Живые логи воркера" in response.text
    assert "Очередь резюме" in response.text
    assert "catalog-link" in response.text
    assert "Вернуться в каталог игр" in response.text


def test_monitor_status_api_reports_worker_and_run_counts(client, monkeypatch):
    run = SimpleNamespace(
        id=8,
        trigger="schedule",
        status="running",
        day=date(2026, 10, 21),
        started_at=datetime(2026, 10, 21, tzinfo=timezone.utc),
        finished_at=None,
        planned=3,
        ok_count=1,
        failed_count=0,
        error=None,
        items=[],
    )
    worker = SimpleNamespace(
        worker_id="host:123",
        state="running",
        last_seen_at=datetime.now(timezone.utc),
        current_run_id=8,
        error=None,
    )

    async def recent_runs(session, limit=20):
        return [run]

    async def worker_statuses(session):
        return [worker]

    log = SimpleNamespace(
        id=12,
        worker_id="host:123",
        level="WARNING",
        message="Provider rate limit",
        run_id=8,
        created_at=datetime.now(timezone.utc),
    )
    monkeypatch.setattr("app.main.repo.recent_runs", recent_runs)
    monkeypatch.setattr("app.main.repo.worker_statuses", worker_statuses)
    monkeypatch.setattr("app.main.repo.recent_worker_logs", _async_logs([log]))
    monkeypatch.setattr(
        "app.main.repo.pending_review_summary_stats",
        _async_summary_stats("HTTP 429 Too Many Requests"),
    )

    response = client.get("/api/monitor/status")

    assert response.status_code == 200
    assert response.json()["active_runs"] == 1
    assert response.json()["total_games"] == 1
    assert response.json()["workers"][0]["state"] == "running"
    assert response.json()["workers"][0]["online"] is True
    assert response.json()["pending_summaries"] == {
        "pending": 4,
        "due": 2,
        "retrying": 3,
        "last_error": "HTTP 429 Too Many Requests",
    }
    assert response.json()["logs"][0]["message"] == "Provider rate limit"
    assert response.json()["logs"][0]["level"] == "WARNING"


def test_worker_payload_marks_stale_worker_offline():
    worker = SimpleNamespace(
        worker_id="host:123",
        state="running",
        last_seen_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        current_run_id=8,
        error=None,
    )
    assert _worker_payload([worker])[0]["state"] == "offline"


def _async_workers(workers):
    async def get_workers(session):
        return workers

    return get_workers


def _async_logs(logs):
    async def get_logs(session):
        return logs

    return get_logs


def _async_summary_stats(error=None):
    async def get_stats(session):
        return {
            "pending": 4,
            "due": 2,
            "retrying": 3,
            "last_error": error,
        }

    return get_stats


def test_catalog_renders_search_platform_filter_and_ratings(client, monkeypatch):
    game = SimpleNamespace(
        slug="demo-game",
        title="Demo & Game",
        cover_url=None,
        best_metascore=88,
        best_userscore=8.4,
        genres=["Action"],
        platforms=[],
    )

    async def list_games(session, **kwargs):
        assert kwargs["query"] == "Demo"
        assert kwargs["platform"] == "pc"
        assert kwargs["sort"] == "userscore"
        return [game], 1

    async def available_platforms(session):
        return [("pc", "PC")]

    monkeypatch.setattr("app.main.repo.list_games", list_games)
    monkeypatch.setattr("app.main.repo.available_platforms", available_platforms)

    response = client.get("/?q=Demo&platform=pc&sort=userscore")

    assert response.status_code == 200
    assert "Demo &amp; Game" in response.text
    assert "88" in response.text
    assert "value='pc' selected" in response.text
    assert (
        "href='/games/demo-game?q=Demo&amp;platform=pc&amp;sort=userscore"
        "&amp;order=desc&amp;page=1'"
    ) in response.text


def test_game_detail_renders_reviews_and_similar_links(client, monkeypatch):
    game = SimpleNamespace(
        slug="demo-game",
        title="Demo Game",
        cover_url=None,
        best_metascore=88,
        best_userscore=8.4,
        release_date=date(2026, 10, 16),
        genres=["Action"],
        developers=["Studio"],
        publishers=["Publisher"],
        description="A game description.",
        video_url=None,
        video_title=None,
        letsplay_url="https://www.youtube.com/watch?v=abcdefghijk",
        letsplay_title="Popular gameplay",
        letsplay_views=1200,
        letsplay_summary="The creator praises the combat.",
        platforms=[
            SimpleNamespace(
                name="PC",
                platform="pc",
                metascore=88,
                userscore=8.4,
                critic_count=20,
                user_count=100,
            )
        ],
        summaries=[
            SimpleNamespace(
                audience="critic",
                summary="Critics like the combat.",
                likes=["Combat"],
                dislikes=["Pacing"],
                reviews_used=10,
            )
        ],
    )
    similar = [SimpleNamespace(slug="similar-game", title="Similar Game")]

    async def game_by_slug(session, slug):
        assert slug == "demo-game"
        return game

    async def similar_games(session, selected_game):
        assert selected_game is game
        return similar

    async def game_neighbors(session, slug, **kwargs):
        assert slug == "demo-game"
        assert kwargs == {
            "query": "Demo",
            "platform": "pc",
            "sort": "userscore",
            "order": "asc",
        }
        return "previous-game", "next-game"

    monkeypatch.setattr("app.main.repo.game_by_slug", game_by_slug)
    monkeypatch.setattr("app.main.repo.similar_games", similar_games)
    monkeypatch.setattr("app.main.repo.game_neighbors", game_neighbors)

    response = client.get(
        "/games/demo-game?q=Demo&platform=pc&sort=userscore&order=asc&page=2"
    )

    assert response.status_code == 200
    assert "A game description." in response.text
    assert "Critics like the combat." in response.text
    assert "Пользователи" in response.text
    assert "/games/similar-game" in response.text
    assert "Popular gameplay" in response.text
    assert "The creator praises the combat." in response.text
    assert "Оригинал (Metacritic)" in response.text
    assert "Русский" in response.text
    assert "/api/games/demo-game/translation" in response.text
    assert "data-translation-field=\"description\"" in response.text
    assert "position: fixed" in response.text
    assert "display: flex; align-items: center; justify-content: center" in response.text
    assert ".game-navigation.previous { left: 1rem; }" in response.text
    assert ".game-navigation.next { right: 1rem; }" in response.text
    assert "href='/games/previous-game?q=Demo&amp;platform=pc&amp;sort=userscore&amp;order=asc&amp;page=2'" in response.text
    assert "href='/games/next-game?q=Demo&amp;platform=pc&amp;sort=userscore&amp;order=asc&amp;page=2'" in response.text
    assert "href='/?q=Demo&amp;platform=pc&amp;sort=userscore&amp;order=asc&amp;page=2'" in response.text


def test_game_translation_is_generated_once_and_cached(client, monkeypatch):
    summary = SimpleNamespace(
        audience="critic",
        summary="Critics like the combat.",
        likes=["Combat"],
        dislikes=["Pacing"],
        summary_ru=None,
        likes_ru=None,
        dislikes_ru=None,
    )
    game = SimpleNamespace(
        id=7,
        slug="demo-game",
        description="A game description.",
        description_ru=None,
        letsplay_summary="The creator praises the combat.",
        letsplay_summary_ru=None,
        summaries=[summary],
    )
    translation_calls = []
    saved = []

    async def game_by_slug(session, slug):
        assert slug == "demo-game"
        return game

    async def translate_game_content(**kwargs):
        translation_calls.append(kwargs)
        return SimpleNamespace(
            description="Описание игры.",
            letsplay_summary="Автор хвалит боевую систему.",
            summaries=[
                SimpleNamespace(
                    audience="critic",
                    summary="Критикам нравится боевая система.",
                    likes=["Боевая система"],
                    dislikes=["Темп"],
                )
            ],
        )

    async def save_game_translations(session, game_id, **kwargs):
        saved.append((game_id, kwargs))

    monkeypatch.setattr("app.main.repo.game_by_slug", game_by_slug)
    monkeypatch.setattr("app.main.translate_game_content", translate_game_content)
    monkeypatch.setattr("app.main.repo.save_game_translations", save_game_translations)

    first = client.post("/api/games/demo-game/translation")
    second = client.post("/api/games/demo-game/translation")

    assert first.status_code == second.status_code == 200
    assert first.json()["description"] == "Описание игры."
    assert first.json()["summaries"][0]["likes"] == ["Боевая система"]
    assert translation_calls[0]["description"] == "A game description."
    assert translation_calls[0]["summaries"][0]["audience"] == "critic"
    assert len(translation_calls) == 1
    assert len(saved) == 1 and saved[0][0] == 7


def test_game_translation_reports_invalid_provider_key(client, monkeypatch):
    game = SimpleNamespace(
        id=9,
        slug="demo-game",
        description="Original description",
        description_ru=None,
        letsplay_summary=None,
        letsplay_summary_ru=None,
        summaries=[],
    )

    async def game_by_slug(session, slug):
        return game

    async def invalid_key(**kwargs):
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        response = httpx.Response(401, request=request)
        raise httpx.HTTPStatusError("Unauthorized", request=request, response=response)

    monkeypatch.setattr("app.main.repo.game_by_slug", game_by_slug)
    monkeypatch.setattr("app.main.translate_game_content", invalid_key)

    response = client.post("/api/games/demo-game/translation")

    assert response.status_code == 503
    assert "OpenRouter отклонил API-ключ (401)" in response.json()["detail"]


def test_game_detail_returns_404_for_unknown_slug(client, monkeypatch):
    async def no_game(session, slug):
        return None

    monkeypatch.setattr("app.main.repo.game_by_slug", no_game)
    response = client.get("/games/missing")
    assert response.status_code == 404


def test_run_list_serializes_runs_and_items(client, monkeypatch):
    item = SimpleNamespace(
        id=1,
        slug="demo-game",
        source="home",
        status="ok",
        error=None,
        processed_at=datetime(2026, 10, 16, tzinfo=timezone.utc),
    )
    run = SimpleNamespace(
        id=4,
        trigger="manual",
        status="ok",
        day=date(2026, 10, 16),
        started_at=datetime(2026, 10, 16, tzinfo=timezone.utc),
        finished_at=datetime(2026, 10, 16, 0, 1, tzinfo=timezone.utc),
        planned=1,
        ok_count=1,
        failed_count=0,
        error=None,
        items=[item],
    )

    async def get_runs(session, limit=20):
        assert limit == 1
        return [run]

    monkeypatch.setattr("app.main.repo.recent_runs", get_runs)

    response = client.get("/api/crawl-runs?limit=1")

    assert response.status_code == 200
    assert response.json()[0]["items"][0]["slug"] == "demo-game"
    assert response.json()[0]["status"] == "ok"


def test_manual_endpoint_queues_crawl(client, monkeypatch):
    requests = []

    async def fake_enqueue_run(session, day):
        requests.append((day, session))
        return SimpleNamespace(id=15)

    monkeypatch.setattr("app.main.repo.enqueue_run", fake_enqueue_run)

    response = client.post("/api/crawl-runs")

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["id"] == 15
    assert len(requests) == 1
