import asyncio
from datetime import date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from app.config import current_day, settings
from app.scraper.client import BlockedError
from app.llm import ReviewSummaryRateLimited, ReviewSummaryResult
from app.scraper.parsers import GamePage, PlatformScore, Review, ReviewsPage
from app.worker import (
    _collect_candidates,
    _fill_platform_scores,
    _maybe_refresh_letsplay,
    process_pending_review_summary,
    _refresh_review_summary,
    _scheduler,
    run_crawl,
)


class FakeSession:
    async def commit(self):
        return None


class FakeClient:
    def __init__(self, home_slugs=None, browse_items=None):
        self.home_slugs = home_slugs or ["aaa", "bbb"]
        self.browse_items = browse_items or ["fatal-train", "frosthaven"]

    async def home(self):
        return """
        <html><body>
          <h2>New Releases</h2>
          <a href='/game/aaa/'>AAA</a>
          <a href='/game/bbb/'>BBB</a>
          <a href='/game/ccc/'>CCC</a>
        </body></html>
        """

    async def browse_new(self, page):
        return """
        <html><body>
          <a href='/game/fatal-train/'>Fatal Train Oct 15, 2026</a>
          <a href='/game/frosthaven/'>Frosthaven Oct 15, 2026</a>
        </body></html>
        """

    async def user_reviews(self, slug, platform):
        return f"<html><body><script id='__NUXT_DATA__'>[{{'data':{{'user-score-summary':{{'data':{{'item':{{'score':8.5,'reviewCount':40,'url':'/game/{slug}/?platform={platform}'}}}}}}}}}}]</script></body></html>"  # noqa: E501


@pytest.mark.asyncio
async def test_collect_candidates_skips_processed_and_duplicates(monkeypatch):
    async def fake_processed(session, day, max_failures):
        assert max_failures == 2
        return {"bbb"}

    async def fake_get_browse_page(session, day):
        return 1

    async def fake_set_browse_page(session, day, page):
        assert page == 1

    monkeypatch.setattr("app.worker.repo.processed_slugs", fake_processed)
    monkeypatch.setattr("app.worker.repo.get_browse_page", fake_get_browse_page)
    monkeypatch.setattr("app.worker.repo.set_browse_page", fake_set_browse_page)

    selected = await _collect_candidates(FakeSession(), FakeClient(), date(2026, 10, 16), 3)
    assert selected == [("aaa", "home"), ("ccc", "home"), ("fatal-train", "browse:1")]


@pytest.mark.asyncio
async def test_fill_platform_scores_sets_lead_user_score(monkeypatch):
    page = GamePage(
        slug="demo",
        title="Demo",
        shown_platform="pc",
        platforms=[
            PlatformScore(platform="pc", is_lead=True),
            PlatformScore(platform="switch"),
        ],
    )

    class Client:
        async def user_reviews(self, slug, platform):
            return platform

    def fake_parse_reviews_page(html, kind):
        platform = html
        score = 8.5 if platform == "pc" else 7.0
        count = 40 if platform == "pc" else 15
        return SimpleNamespace(score=score, review_count=count)

    monkeypatch.setattr("app.worker.parse_reviews_page", fake_parse_reviews_page)

    lead_platform, lead_reviews = await _fill_platform_scores(Client(), "demo", page)
    assert page.userscore == 8.5
    assert page.platforms[0].userscore == 8.5
    assert page.platforms[1].userscore == 7.0
    assert lead_platform == "pc"
    assert lead_reviews.score == 8.5


@pytest.mark.asyncio
async def test_fill_platform_scores_fetches_lead_first_and_limits_to_eight(monkeypatch):
    platforms = [PlatformScore(platform=f"platform-{i}") for i in range(9)]
    platforms[-1].is_lead = True
    page = GamePage(slug="demo", platforms=platforms)
    requested = []

    class Client:
        async def user_reviews(self, slug, platform):
            requested.append(platform)
            return platform

    def fake_parse_reviews_page(html, kind):
        return SimpleNamespace(score=8.0, review_count=1)

    monkeypatch.setattr("app.worker.parse_reviews_page", fake_parse_reviews_page)
    lead_platform, _ = await _fill_platform_scores(Client(), "demo", page)

    assert lead_platform == "platform-8"
    assert requested[0] == "platform-8"
    assert len(requested) == 8


@pytest.mark.asyncio
async def test_fill_platform_scores_respects_bounded_concurrency(monkeypatch):
    monkeypatch.setattr(settings, "metacritic_max_concurrency", 2)
    monkeypatch.setattr(
        "app.worker.parse_reviews_page",
        lambda platform, kind: SimpleNamespace(score=8.0, review_count=10),
    )
    active = 0
    maximum_active = 0

    class Client:
        async def user_reviews(self, slug, platform):
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return platform

    page = GamePage(
        slug="demo",
        platforms=[
            PlatformScore(platform=f"platform-{index}", is_lead=index == 0)
            for index in range(5)
        ],
    )
    lead_platform, _ = await _fill_platform_scores(Client(), "demo", page)

    assert lead_platform == "platform-0"
    assert maximum_active == 2


@pytest.mark.asyncio
async def test_review_summary_is_not_recomputed_for_unchanged_reviews(monkeypatch):
    class Session:
        commits = 0

        async def commit(self):
            self.commits += 1

    async def fail_if_called(page):
        raise AssertionError("unchanged reviews must not call the LLM")

    async def get_same_hash(session, game_id, audience):
        return "same-hash"

    async def delete_pending(session, game_id, audience):
        return None

    async def record_log(*args, **kwargs):
        return None

    monkeypatch.setattr("app.worker.review_source_hash", lambda page: "same-hash")
    monkeypatch.setattr("app.worker.repo.record_worker_log", record_log)
    monkeypatch.setattr("app.worker.repo.get_summary_source_hash", get_same_hash)
    monkeypatch.setattr(
        "app.worker.repo.delete_pending_review_summary_for_game", delete_pending
    )
    monkeypatch.setattr("app.worker.summarize_reviews", fail_if_called)

    session = Session()
    await _refresh_review_summary(
        session,
        1,
        ReviewsPage(kind="critic", reviews=[Review(kind="critic", text="A good game.")]),
    )

    assert session.commits == 1


@pytest.mark.asyncio
async def test_changed_review_snapshot_is_queued_without_waiting_for_llm(monkeypatch):
    class Session:
        commits = 0

        async def commit(self):
            self.commits += 1

    queued = []

    async def fake_get_hash(session, game_id, audience):
        return "old-hash"

    async def no_pending_hash(session, game_id, audience):
        return None

    async def queue_snapshot(session, *args):
        queued.append(args)

    async def record_log(*args, **kwargs):
        return None

    monkeypatch.setattr("app.worker.review_source_hash", lambda page: "new-hash")
    monkeypatch.setattr("app.worker.repo.get_summary_source_hash", fake_get_hash)
    monkeypatch.setattr(
        "app.worker.repo.get_pending_review_summary_source_hash", no_pending_hash
    )
    monkeypatch.setattr(
        "app.worker.repo.enqueue_pending_review_summary", queue_snapshot
    )
    monkeypatch.setattr("app.worker.repo.record_worker_log", record_log)

    async def fail_if_called(page):
        raise AssertionError("the crawl must not wait for LLM generation")

    monkeypatch.setattr("app.worker.summarize_reviews", fail_if_called)

    session = Session()
    reviews = ReviewsPage(
        kind="user",
        review_count=1,
        reviews=[Review(kind="user", text="The story is great.")],
    )
    await _refresh_review_summary(session, 7, reviews)

    assert queued[0][0:3] == (7, "user", "")
    assert queued[0][4][0]["text"] == "The story is great."
    assert queued[0][5] == "new-hash"
    assert session.commits == 1


@pytest.mark.asyncio
async def test_review_ratings_without_written_text_skip_summary(monkeypatch):
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("a summary cannot be generated without review text")

    async def delete_pending(*args, **kwargs):
        return None

    async def record_log(*args, **kwargs):
        return None

    monkeypatch.setattr("app.worker.repo.delete_summary", fail_if_called)
    monkeypatch.setattr(
        "app.worker.repo.delete_pending_review_summary_for_game", delete_pending
    )
    monkeypatch.setattr("app.worker.repo.record_worker_log", record_log)
    monkeypatch.setattr("app.worker.summarize_reviews", fail_if_called)

    await _refresh_review_summary(
        FakeSession(),
        1,
        ReviewsPage(kind="user", review_count=4),
    )


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_pending_review_summary_is_processed_and_removed(monkeypatch):
    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def commit(self):
            return None

    pending = SimpleNamespace(
        id=17,
        game_id=3,
        audience="user",
        platform="pc",
        review_count=1,
        reviews=[{"kind": "user", "text": "Great combat.", "score": 9}],
        source_hash="source-hash",
    )
    saved = []
    deleted = []

    async def get_due(session):
        return pending

    async def summarize(reviews):
        assert reviews.platform == "pc"
        assert reviews.reviews[0].text == "Great combat."
        return ReviewSummaryResult(
            summary="Players praise combat.",
            likes=["Combat"],
            dislikes=[],
        )

    async def upsert(session, *args):
        saved.append(args)

    async def delete_pending(session, pending_id):
        deleted.append(pending_id)

    async def record_log(*args, **kwargs):
        return None

    monkeypatch.setattr("app.worker.SessionLocal", Session)
    monkeypatch.setattr("app.worker.repo.due_pending_review_summary", get_due)
    monkeypatch.setattr("app.worker.repo.upsert_summary", upsert)
    monkeypatch.setattr("app.worker.repo.delete_pending_review_summary", delete_pending)
    monkeypatch.setattr("app.worker.repo.record_worker_log", record_log)
    monkeypatch.setattr("app.worker.summarize_reviews", summarize)

    assert await process_pending_review_summary()
    assert saved[0][0:2] == (3, "user")
    assert saved[0][6] == "source-hash"
    assert deleted == [17]


@pytest.mark.asyncio
async def test_rate_limited_summary_uses_provider_retry_after(monkeypatch):
    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def commit(self):
            return None

    pending = SimpleNamespace(
        id=18,
        game_id=4,
        audience="critic",
        platform="pc",
        review_count=1,
        reviews=[{"kind": "critic", "text": "Good combat."}],
        source_hash="source-hash",
        attempts=1,
    )
    deferred = []

    async def get_due(session):
        return pending

    async def rate_limited(reviews):
        raise ReviewSummaryRateLimited("HTTP 429", 240)

    async def defer(session, summary, error, retry_after_seconds=None):
        deferred.append((summary.id, error, retry_after_seconds))

    async def defer_all(session, error, delay_seconds):
        deferred.append(("all", error, delay_seconds))

    async def record_log(*args, **kwargs):
        return None

    monkeypatch.setattr("app.worker.SessionLocal", Session)
    monkeypatch.setattr("app.worker.repo.due_pending_review_summary", get_due)
    monkeypatch.setattr("app.worker.repo.defer_pending_review_summary", defer)
    monkeypatch.setattr("app.worker.repo.defer_all_pending_review_summaries", defer_all)
    monkeypatch.setattr("app.worker.repo.record_worker_log", record_log)
    monkeypatch.setattr("app.worker.summarize_reviews", rate_limited)

    assert not await process_pending_review_summary()
    assert deferred == [
        (18, "HTTP 429", 240),
        ("all", "HTTP 429", 240),
    ]


@pytest.mark.asyncio
async def test_youtube_enrichment_can_be_disabled(monkeypatch):
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("YouTube enrichment should be skipped")

    monkeypatch.setattr(settings, "youtube_enrichment_enabled", False)
    monkeypatch.setattr("app.worker._refresh_letsplay", fail_if_called)

    await _maybe_refresh_letsplay(FakeSession(), 1, "Demo Game")


@pytest.mark.asyncio
async def test_fill_platform_scores_propagates_blocked_error(monkeypatch):
    monkeypatch.setattr(settings, "metacritic_max_concurrency", 1)
    page = GamePage(
        slug="demo",
        platforms=[
            PlatformScore(platform="pc"),
            PlatformScore(platform="switch"),
        ],
    )
    requested = []

    class Client:
        async def user_reviews(self, slug, platform):
            requested.append(platform)
            raise BlockedError("challenge page")

    with pytest.raises(BlockedError, match="challenge page"):
        await _fill_platform_scores(Client(), "demo", page)
    assert requested == ["pc"]
    assert requested == ["pc"]


@pytest.mark.asyncio
async def test_crawl_stops_after_blocked_page(monkeypatch):
    run = SimpleNamespace(id=10, planned=0, ok_count=0, failed_count=0, status="running")
    recorded = []
    requested_games = []

    class Connection:
        async def scalar(self, *args, **kwargs):
            return True

        async def commit(self):
            pass

        async def close(self):
            pass

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def commit(self):
            pass

        async def rollback(self):
            pass

        async def get(self, model, run_id):
            return run

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def game(self, slug):
            requested_games.append(slug)
            raise BlockedError("challenge page")

    async def fake_connect():
        return Connection()

    async def fake_start_run(session, trigger, day):
        return run

    async def fake_record_item(session, run_id, day, slug, source, status, error=None):
        recorded.append((slug, status, error))

    async def fake_finish_run(session, crawl_run, error=None):
        crawl_run.status = "failed" if error else "ok"
        crawl_run.error = error

    async def fake_heartbeat(session, worker_id, **kwargs):
        return None

    async def fake_candidates(*args):
        return [("blocked-game", "home"), ("later-game", "home")]

    async def fake_log(*args, **kwargs):
        return None

    monkeypatch.setattr("app.worker.engine", SimpleNamespace(connect=fake_connect))
    monkeypatch.setattr("app.worker.SessionLocal", Session)
    monkeypatch.setattr("app.worker.repo.fail_abandoned_runs", lambda session: _async_value(0))
    monkeypatch.setattr("app.worker.repo.start_run", fake_start_run)
    monkeypatch.setattr("app.worker.repo.record_item", fake_record_item)
    monkeypatch.setattr("app.worker.repo.finish_run", fake_finish_run)
    monkeypatch.setattr("app.worker.repo.heartbeat_worker", fake_heartbeat)
    monkeypatch.setattr("app.worker.repo.record_worker_log", fake_log)
    monkeypatch.setattr("app.worker._collect_candidates", fake_candidates)

    result = await run_crawl(day=date(2026, 10, 16), client=Client())

    assert result["status"] == "failed"
    assert requested_games == ["blocked-game"]
    assert [status for _, status, _ in recorded] == ["pending", "failed"]
    assert recorded[-1][2] == "challenge page"


@pytest.mark.asyncio
async def test_run_crawl_rejects_zero_batch_size():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        await run_crawl(batch_size=0)


def test_configured_timezone_controls_crawl_day_and_scheduler(monkeypatch):
    monkeypatch.setattr(settings, "timezone", "Pacific/Kiritimati")

    assert current_day() == datetime.now(ZoneInfo("Pacific/Kiritimati")).date()
    scheduler = _scheduler()
    assert str(scheduler.timezone) == "Pacific/Kiritimati"
    assert "pending-review-summaries" in {job.id for job in scheduler.get_jobs()}


async def _async_value(value):
    return value
