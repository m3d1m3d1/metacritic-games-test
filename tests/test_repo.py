"""DB tests run against a real Postgres+pgvector (set TEST_DATABASE_URL, e.g.
postgresql+asyncpg://games:games@localhost:5432/games_test). The schema is created by the real
Alembic migration, so these tests also cover the migration itself."""
import os
import subprocess
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.db import repo
from app.db.models import (
    EMBEDDING_DIM,
    CrawlItem,
    Game,
    GamePlatform,
    PendingReviewSummary,
    ReviewSummary,
    WorkerLog,
)
from app.scraper.parsers import GamePage, PlatformScore

TEST_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_URL, reason="TEST_DATABASE_URL not set")
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def migrated():
    if not TEST_URL:
        return
    env = {**os.environ, "ALEMBIC_DATABASE_URL": TEST_URL}
    subprocess.run([sys.executable, "-m", "alembic", "downgrade", "base"], cwd=ROOT, env=env, check=True)
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True)


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine(TEST_URL, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "TRUNCATE games, crawl_runs, crawl_state, worker_status, worker_logs "
                "RESTART IDENTITY CASCADE"
            )
        )
    async with async_sessionmaker(engine, expire_on_commit=False)() as s:
        yield s
    await engine.dispose()


@pytest.fixture
def page():
    return GamePage(
        slug="example-game",
        title="Example Game",
        description="A fictional game used in repository tests.",
        cover_url="https://example.test/portrait.jpg",
        developers=["Example Studio"],
        publishers=["Sample Publisher"],
        genres=["Adventure"],
        release_date="2026-10-15",
        metascore=76,
        userscore=None,
        shown_platform="nintendo-switch-2",
        platforms=[
            PlatformScore(platform="pc", critic_count=2),
            PlatformScore(
                platform="nintendo-switch-2",
                name="Switch 2",
                is_lead=True,
                release_date="2026-10-15",
                metascore=76,
                critic_count=8,
            ),
            PlatformScore(platform="playstation-5", critic_count=3),
            PlatformScore(platform="xbox-series-x", critic_count=2),
        ],
        video_id="testPlayer42",
        video_title="Example trailer",
        video_url="https://cdn.example.test/players/testPlayer42.html",
    )


async def count(session: AsyncSession, model) -> int:
    return await session.scalar(select(func.count()).select_from(model))


async def test_upsert_creates_then_updates_in_place(session, page):
    gid, created = await repo.upsert_game(session, page)
    assert created
    gid2, created2 = await repo.upsert_game(session, page)
    assert (gid2, created2) == (gid, False)
    assert await count(session, Game) == 1 and await count(session, GamePlatform) == 4

    game = await session.get(Game, gid)
    assert game.title == "Example Game" and game.developers == ["Example Studio"]
    assert game.release_date == date(2026, 10, 15)
    assert game.best_metascore == 76 and game.best_userscore is None
    assert game.video_url.endswith("testPlayer42.html")


async def test_refresh_updates_scores_but_does_not_erase_data_with_nulls(session, page):
    gid, _ = await repo.upsert_game(session, page)
    broken = replace(page, description=None, developers=[], cover_url=None)
    for p in broken.platforms:
        if p.platform == "pc":
            p.metascore, p.userscore, p.user_count = 80, 7.5, 12
    await repo.upsert_game(session, broken)
    game = await session.get(Game, gid)
    await session.refresh(game)
    assert (
        game.description
        and game.developers == ["Example Studio"]
        and game.cover_url
    )
    assert game.best_metascore == 80 and float(game.best_userscore) == 7.5

    # a later crawl without user score must not wipe the stored one
    again = replace(page, platforms=[replace(p) for p in page.platforms])
    await repo.upsert_game(session, again)
    pc = await session.scalar(select(GamePlatform).where(GamePlatform.platform == "pc"))
    await session.refresh(pc)
    assert float(pc.userscore) == 7.5 and pc.user_count == 12


async def test_vanished_platform_is_removed_and_cascade_works(session, page):
    gid, _ = await repo.upsert_game(session, page)
    fewer = replace(page, platforms=[p for p in page.platforms if p.platform != "pc"])
    await repo.upsert_game(session, fewer)
    assert await count(session, GamePlatform) == 3
    await session.execute(Game.__table__.delete().where(Game.id == gid))
    assert await count(session, GamePlatform) == 0


async def test_upsert_rejects_page_without_slug(session, page):
    with pytest.raises(ValueError):
        await repo.upsert_game(session, replace(page, slug=None))


async def test_embedding_nearest_neighbour(session, page):
    ids = []
    for i, slug in enumerate(("a", "b", "c")):
        gid, _ = await repo.upsert_game(session, replace(page, slug=slug, title=slug.upper()))
        vec = [0.0] * EMBEDDING_DIM
        vec[i] = 1.0
        await repo.set_embedding(session, gid, vec, f"h{i}")
        assert await repo.get_embedding_source_hash(session, gid) == f"h{i}"
        ids.append(gid)
    query = [0.0] * EMBEDDING_DIM
    query[1], query[0] = 1.0, 0.2
    rows = (await session.execute(
        select(Game.slug).where(Game.embedding.is_not(None)).order_by(Game.embedding.cosine_distance(query)).limit(2)
    )).scalars().all()
    assert rows == ["b", "a"]


async def test_game_catalog_search_platform_filter_and_rating_sort(session, page):
    await repo.upsert_game(session, replace(page, slug="game-low", title="Alpha Game"))
    high_page = replace(
        page,
        slug="game-high",
        title="Beta Game",
        platforms=[replace(platform, metascore=91 if platform.is_lead else platform.metascore) for platform in page.platforms],
    )
    await repo.upsert_game(session, high_page)

    ranked, total = await repo.list_games(session, sort="metascore", order="desc")
    assert [game.slug for game in ranked] == ["game-high", "game-low"]
    assert total == 2

    found, total = await repo.list_games(session, query="beta", platform="pc")
    assert total == 1
    assert [game.slug for game in found] == ["game-high"]


async def test_game_neighbors_respect_catalog_filters_and_sort_order(session, page):
    await repo.upsert_game(session, replace(page, slug="game-a", title="A Game"))
    await repo.upsert_game(session, replace(page, slug="game-b", title="B Game"))
    await repo.upsert_game(session, replace(page, slug="other", title="C Other"))

    previous_slug, next_slug = await repo.game_neighbors(
        session,
        "game-b",
        query="Game",
        platform="pc",
        sort="title",
        order="asc",
    )
    assert (previous_slug, next_slug) == ("game-a", None)

    previous_slug, next_slug = await repo.game_neighbors(
        session,
        "game-a",
        query="Game",
        platform="pc",
        sort="title",
        order="asc",
    )
    assert (previous_slug, next_slug) == (None, "game-b")


async def test_similar_games_uses_genres_until_embeddings_exist(session, page):
    await repo.upsert_game(session, replace(page, slug="current", title="Current"))
    await repo.upsert_game(session, replace(page, slug="related", title="Related"))
    await repo.upsert_game(session, replace(page, slug="unrelated", title="Unrelated", genres=["RPG"]))

    current = await repo.game_by_slug(session, "current")
    assert current is not None
    results = await repo.similar_games(session, current)
    assert [game.slug for game in results] == ["related"]


async def test_summary_upsert_is_one_row_per_audience_and_validates(session, page):
    gid, _ = await repo.upsert_game(session, page)
    for text_ in ("v1", "v2"):
        await repo.upsert_summary(session, gid, "critic", text_, ["fun"], ["short"], 8, "h", "m")
    await repo.upsert_summary(session, gid, "user", "u", [], [], 0, "h", "m")
    assert await count(session, ReviewSummary) == 2
    row = await session.scalar(select(ReviewSummary).where(ReviewSummary.audience == "critic"))
    assert row.summary == "v2"
    assert await repo.get_summary_source_hash(session, gid, "critic") == "h"
    await repo.delete_summary(session, gid, "user")
    assert await count(session, ReviewSummary) == 1
    with pytest.raises(IntegrityError):
        await repo.upsert_summary(session, gid, "robot", "x", [], [], 0, "h", "m")


async def test_translations_are_persisted_and_invalidated_when_source_changes(session, page):
    gid, _ = await repo.upsert_game(session, page)
    await repo.upsert_summary(
        session, gid, "critic", "Original summary", ["Combat"], [], 3, "hash-1", "model"
    )
    await repo.update_letsplay(
        session,
        gid,
        url="https://www.youtube.com/watch?v=abcdefghijk",
        title="Gameplay",
        views=100,
        summary="Original letsplay summary",
        source_hash="video-1",
    )
    await repo.save_game_translations(
        session,
        gid,
        description_ru="Перевод описания",
        letsplay_summary_ru="Перевод летсплея",
        summaries={
            "critic": {
                "summary": "Перевод резюме",
                "likes": ["Боевая система"],
                "dislikes": [],
            }
        },
    )
    game = await repo.game_by_slug(session, page.slug)
    assert game.description_ru == "Перевод описания"
    assert game.letsplay_summary_ru == "Перевод летсплея"
    assert game.summaries[0].summary_ru == "Перевод резюме"
    assert game.summaries[0].likes_ru == ["Боевая система"]

    await repo.upsert_game(session, replace(page, description="Updated description"))
    await repo.update_letsplay(
        session,
        gid,
        url="https://www.youtube.com/watch?v=abcdefghijk",
        title="Gameplay",
        views=100,
        summary="Updated letsplay summary",
        source_hash="video-2",
    )
    await repo.upsert_summary(
        session, gid, "critic", "Updated summary", ["Story"], [], 4, "hash-2", "model"
    )
    await session.refresh(game)
    await session.refresh(game.summaries[0])
    assert game.description_ru is None
    assert game.letsplay_summary_ru is None
    assert game.summaries[0].summary_ru is None
    assert game.summaries[0].likes_ru is None


async def test_pending_review_summary_is_persisted_and_retryable(session, page):
    game_id, _ = await repo.upsert_game(session, page)
    payload = [{"kind": "user", "text": "Combat feels great.", "score": 9}]
    await repo.enqueue_pending_review_summary(
        session,
        game_id,
        "user",
        "pc",
        12,
        payload,
        "review-source-hash",
    )
    await session.flush()

    pending = await session.scalar(select(PendingReviewSummary))
    assert pending is not None
    assert pending.game_id == game_id
    assert pending.reviews == payload
    assert pending.source_hash == "review-source-hash"
    assert pending.attempts == 0
    assert (
        await repo.get_pending_review_summary_source_hash(session, game_id, "user")
        == "review-source-hash"
    )

    pending.next_attempt_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    await session.flush()
    claimed = await repo.due_pending_review_summary(session)
    assert claimed is not None and claimed.id == pending.id

    retry_started = datetime.now(timezone.utc)
    await repo.defer_pending_review_summary(
        session, claimed, "still unavailable", retry_after_seconds=180
    )
    await session.flush()
    await session.refresh(claimed)
    assert claimed.attempts == 1
    assert claimed.last_error == "still unavailable"
    assert claimed.next_attempt_at >= retry_started + timedelta(seconds=179)

    await repo.delete_pending_review_summary(session, claimed.id)
    assert await count(session, PendingReviewSummary) == 0


async def test_worker_logs_and_pending_summary_stats(session, page):
    game_id, _ = await repo.upsert_game(session, page)
    await repo.enqueue_pending_review_summary(
        session,
        game_id,
        "user",
        "pc",
        1,
        [{"kind": "user", "text": "Good combat."}],
        "review-source-hash",
    )
    await repo.record_worker_log(session, "worker-test", "WARNING", "Provider is rate limited")
    await session.commit()

    logs = await repo.recent_worker_logs(session)
    stats = await repo.pending_review_summary_stats(session)

    assert len(logs) == 1
    assert isinstance(logs[0], WorkerLog)
    assert logs[0].level == "WARNING"
    assert logs[0].message == "Provider is rate limited"
    assert stats == {
        "pending": 1,
        "due": 1,
        "retrying": 0,
        "last_error": None,
    }


async def test_rate_limit_cooldown_defers_all_pending_summaries(session, page):
    game_id, _ = await repo.upsert_game(session, page)
    for audience in ("critic", "user"):
        await repo.enqueue_pending_review_summary(
            session,
            game_id,
            audience,
            "pc",
            1,
            [{"kind": audience, "text": "A review."}],
            f"{audience}-source",
        )
    await session.flush()

    retry_started = datetime.now(timezone.utc)
    await repo.defer_all_pending_review_summaries(
        session, "provider rate limit", 300
    )
    await session.flush()

    pending = (await session.scalars(select(PendingReviewSummary))).all()
    assert len(pending) == 2
    assert all(item.last_error == "provider rate limit" for item in pending)
    assert all(
        item.next_attempt_at >= retry_started + timedelta(seconds=299)
        for item in pending
    )


async def test_crawl_bookkeeping(session):
    day = date(2026, 10, 16)
    assert await repo.get_browse_page(session, day) == 1
    await repo.set_browse_page(session, day, 3)
    await repo.set_browse_page(session, day, 4)
    assert await repo.get_browse_page(session, day) == 4
    assert await repo.get_browse_page(session, date(2026, 10, 17)) == 1  # new day starts over

    run = await repo.start_run(session, "manual", day)
    await repo.record_item(session, run.id, day, "g1", "home", "ok")
    await repo.record_item(session, run.id, day, "g2", "browse:1", "failed", "boom")
    await repo.record_item(session, run.id, day, "g3", "browse:1", "pending")
    await repo.record_item(session, run.id, day, "g3", "browse:1", "ok")  # retry updates, no duplicate
    assert await count(session, CrawlItem) == 3
    assert await repo.processed_slugs(session, day) == {"g1", "g3"}
    assert await repo.processed_slugs(session, date(2026, 10, 17)) == set()

    await repo.finish_run(session, run)
    assert (run.status, run.ok_count, run.failed_count) == ("partial", 2, 1)


async def test_failed_game_is_retried_once_then_skipped_for_the_day(session):
    day = date(2026, 10, 16)
    first_run = await repo.start_run(session, "schedule", day)
    await repo.record_item(session, first_run.id, day, "unstable-game", "home", "failed", "timeout")
    await session.commit()
    assert await repo.processed_slugs(session, day) == set()

    second_run = await repo.start_run(session, "schedule", day)
    await repo.record_item(session, second_run.id, day, "unstable-game", "home", "failed", "timeout")
    await session.commit()
    assert await repo.processed_slugs(session, day) == {"unstable-game"}
    assert await repo.processed_slugs(session, date(2026, 10, 17)) == set()


async def test_abandoned_run_and_pending_items_are_marked_failed(session):
    day = date(2026, 10, 16)
    run = await repo.start_run(session, "schedule", day)
    await repo.record_item(session, run.id, day, "unfinished-game", "home", "pending")
    await session.commit()

    assert await repo.fail_abandoned_runs(session) == 1
    await session.refresh(run)
    item = await session.scalar(select(CrawlItem).where(CrawlItem.run_id == run.id))
    assert run.status == "failed"
    assert run.finished_at is not None
    assert item.status == "failed"
    assert item.error == "Worker stopped before completing this item"


async def test_queued_run_can_be_claimed(session):
    queued = await repo.enqueue_run(session, date(2026, 10, 16))
    await session.commit()
    assert queued.status == "queued"

    claimed = await repo.claim_queued_run(session)
    assert claimed is not None
    assert claimed.id == queued.id
    assert claimed.status == "running"


async def test_worker_heartbeat_preserves_current_state(session):
    run = await repo.start_run(session, "schedule", date(2026, 10, 21))
    await repo.heartbeat_worker(
        session, "test-worker", state="running", current_run_id=run.id
    )
    await session.commit()

    await repo.heartbeat_worker(session, "test-worker")
    await session.commit()

    workers = await repo.worker_statuses(session)
    assert len(workers) == 1
    assert workers[0].state == "running"
    assert workers[0].current_run_id == run.id
    assert workers[0].last_seen_at is not None
