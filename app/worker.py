from __future__ import annotations

import argparse
import asyncio
import logging
import os
import socket
from datetime import date
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import current_day, get_timezone, settings
from app.db import repo
from app.db.models import CrawlRun
from app.db.session import SessionLocal, engine
from app.embeddings import embed_game, embedding_source_hash
from app.scraper.client import BlockedError, MetacriticClient
from app.llm import ReviewSummaryRateLimited, review_source_hash, summarize_reviews
from app.scraper.parsers import (
    GamePage,
    Review,
    ReviewsPage,
    parse_browse,
    parse_game_page,
    parse_home_new_releases,
    parse_reviews_page,
)
from app.youtube import (
    fetch_transcript,
    find_most_viewed_letsplay,
    letsplay_source_hash,
    summarize_letsplay,
)

logger = logging.getLogger(__name__)
_CRAWL_LOCK_ID = 734829105
_WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"
_WORKER_HEARTBEAT_SECONDS = 5


async def _record_worker_log(
    session: AsyncSession, level: str, message: str, run_id: int | None = None
) -> None:
    await repo.record_worker_log(session, _WORKER_ID, level, message, run_id)


async def _collect_candidates(
    session: AsyncSession,
    client: MetacriticClient,
    day: date,
    batch_size: int,
) -> list[tuple[str, str]]:
    """Return up to batch_size (slug, source) pairs still unprocessed for this day."""
    selected: list[tuple[str, str]] = []
    seen: set[str] = set()
    processed = await repo.processed_slugs(session, day, settings.crawl_max_failures_per_day)

    def add_slug(slug: str, source: str) -> None:
        if slug and slug not in seen and slug not in processed:
            seen.add(slug)
            selected.append((slug, source))

    home = parse_home_new_releases(await client.home())
    for slug in home:
        add_slug(slug, "home")
        if len(selected) >= batch_size:
            return selected[:batch_size]

    page_no = await repo.get_browse_page(session, day)
    while len(selected) < batch_size:
        await repo.set_browse_page(session, day, page_no)
        html = await client.browse_new(page_no)
        items = parse_browse(html)
        if not items:
            break
        for item in items:
            add_slug(item.slug, f"browse:{page_no}")
            if len(selected) >= batch_size:
                await repo.set_browse_page(session, day, page_no)
                return selected[:batch_size]
        if len(items) < 24:
            break
        page_no += 1
        if page_no > 20:
            break

    await repo.set_browse_page(session, day, page_no)
    return selected[:batch_size]


async def _fill_platform_scores(
    client: MetacriticClient, slug: str, page: GamePage
) -> tuple[str | None, ReviewsPage | None]:
    lead = next((p for p in page.platforms if p.is_lead), None)
    if lead is None:
        lead = next((p for p in page.platforms if p.platform == page.shown_platform), None)
    if lead is None and page.platforms:
        lead = page.platforms[0]

    platforms = ([lead] if lead else []) + [p for p in page.platforms if p is not lead]
    lead_reviews = None

    async def fetch_user_reviews(platform):
        if not platform.platform:
            return platform, None, None
        try:
            user_html = await client.user_reviews(slug, platform.platform)
            user_page = parse_reviews_page(user_html, "user")
        except BlockedError:
            raise
        except Exception as exc:
            return platform, None, exc
        return platform, user_page, None

    results = []
    requested_platforms = platforms[:8]
    for start in range(0, len(requested_platforms), settings.metacritic_max_concurrency):
        batch = requested_platforms[start : start + settings.metacritic_max_concurrency]
        tasks = [asyncio.create_task(fetch_user_reviews(platform)) for platform in batch]
        try:
            results.extend(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    for platform, user_page, error in results:
        if error is not None:
            if isinstance(error, BlockedError):
                raise error
            logger.error(
                "Unable to fetch user score for %s on %s: %s",
                slug,
                platform.platform,
                error,
            )
            if platform is lead:
                raise RuntimeError(
                    f"Unable to fetch user reviews on lead platform {platform.platform}"
                ) from error
            continue
        if user_page is None:
            continue
        platform.userscore = user_page.score
        platform.user_count = user_page.review_count
        if page.shown_platform == platform.platform:
            page.userscore = user_page.score
        if platform is lead:
            lead_reviews = user_page
    return lead.platform if lead else None, lead_reviews


async def _refresh_review_summary(
    session: AsyncSession, game_id: int, reviews: ReviewsPage
) -> None:
    source_hash = review_source_hash(reviews)
    if not reviews.reviews:
        if reviews.review_count:
            logger.warning(
                "No %s review text available for %s reported reviews; skipping summary",
                reviews.kind,
                reviews.review_count,
            )
            await _record_worker_log(
                session,
                "WARNING",
                f"No {reviews.kind} review text available for game {game_id} "
                f"({reviews.review_count} reported reviews)",
            )
            await repo.delete_pending_review_summary_for_game(
                session, game_id, reviews.kind
            )
            await session.commit()
            return
        await repo.delete_summary(session, game_id, reviews.kind)
        await repo.delete_pending_review_summary_for_game(
            session, game_id, reviews.kind
        )
        await session.commit()
        return

    existing_hash = await repo.get_summary_source_hash(session, game_id, reviews.kind)
    if existing_hash == source_hash:
        await repo.delete_pending_review_summary_for_game(
            session, game_id, reviews.kind
        )
        await session.commit()
        return

    pending_hash = await repo.get_pending_review_summary_source_hash(
        session, game_id, reviews.kind
    )
    if pending_hash == source_hash:
        await session.commit()
        return

    pending_reviews = [
        {
            "kind": review.kind,
            "text": review.text,
            "score": review.score,
            "author": review.author,
            "date": review.date,
            "url": review.url,
            "platform": review.platform,
            "spoiler": review.spoiler,
        }
        for review in reviews.reviews
    ]
    await repo.enqueue_pending_review_summary(
        session,
        game_id,
        reviews.kind,
        reviews.platform or "",
        reviews.review_count,
        pending_reviews,
        source_hash,
    )
    await _record_worker_log(
        session, "INFO", f"Queued {reviews.kind} review summary for game {game_id}"
    )
    await session.commit()
    logger.info(
        "Queued %s review summary for game %s",
        reviews.kind,
        game_id,
    )


async def process_pending_review_summary() -> bool:
    async with SessionLocal() as session:
        pending = await repo.due_pending_review_summary(session)
        if pending is None:
            return False

        reviews = ReviewsPage(
            kind=pending.audience,
            platform=pending.platform,
            review_count=pending.review_count,
            reviews=[
                Review(
                    kind=item["kind"],
                    text=item["text"],
                    score=item.get("score"),
                    author=item.get("author"),
                    date=item.get("date"),
                    url=item.get("url"),
                    platform=item.get("platform"),
                    spoiler=item.get("spoiler", False),
                )
                for item in pending.reviews
            ],
        )
        pending_id = pending.id
        try:
            result = await summarize_reviews(reviews)
        except Exception as exc:
            retry_after_seconds = (
                exc.retry_after_seconds
                if isinstance(exc, ReviewSummaryRateLimited)
                else None
            )
            rate_limit_cooldown = None
            if isinstance(exc, ReviewSummaryRateLimited):
                rate_limit_cooldown = max(
                    retry_after_seconds or 300,
                    min(60 * 2 ** min(pending.attempts, 6), 3600),
                )
            await repo.defer_pending_review_summary(
                session, pending, str(exc), retry_after_seconds
            )
            if rate_limit_cooldown is not None:
                await repo.defer_all_pending_review_summaries(
                    session, str(exc), rate_limit_cooldown
                )
            await _record_worker_log(
                session,
                "WARNING",
                f"Summary retry scheduled for game {pending.game_id} "
                f"({pending.audience}; attempt {pending.attempts}): {str(exc)[:500]}"
                + (
                    f"; queue paused for {rate_limit_cooldown}s"
                    if rate_limit_cooldown is not None
                    else ""
                ),
            )
            await session.commit()
            logger.exception(
                "Pending %s summary %s failed; retry scheduled",
                pending.audience,
                pending_id,
            )
            return False

        await repo.upsert_summary(
            session,
            pending.game_id,
            pending.audience,
            result.summary,
            result.likes,
            result.dislikes,
            len(reviews.reviews),
            pending.source_hash,
            result.model or settings.summary_model,
        )
        await repo.delete_pending_review_summary(session, pending_id)
        selected_model = result.model or settings.summary_model
        await _record_worker_log(
            session,
            "INFO",
            f"Generated {pending.audience} review summary for game {pending.game_id} "
            f"with model {selected_model}",
        )
        await session.commit()
        logger.info(
            "Generated pending %s summary %s with model %s",
            pending.audience,
            pending_id,
            result.model or settings.summary_model,
        )
        return True


async def _refresh_letsplay(session: AsyncSession, game_id: int, title: str) -> None:
    letsplay = await find_most_viewed_letsplay(title)
    if letsplay is None:
        logger.info("No YouTube letsplay found for %s", title)
        return
    transcript = await fetch_transcript(letsplay.video_id)
    if not transcript:
        logger.info("Skipping YouTube conclusion for %s: transcript unavailable", title)
        return

    source_hash = letsplay_source_hash(letsplay.video_id, transcript)
    existing_hash = await repo.get_letsplay_source_hash(session, game_id)
    await session.commit()
    if existing_hash == source_hash:
        return

    conclusion = await summarize_letsplay(title, transcript)
    await repo.update_letsplay(
        session,
        game_id,
        url=letsplay.url,
        title=letsplay.title,
        views=letsplay.views,
        summary=conclusion,
        source_hash=source_hash,
    )
    await session.commit()


async def _maybe_refresh_letsplay(
    session: AsyncSession, game_id: int, title: str
) -> None:
    if not settings.youtube_enrichment_enabled:
        logger.info("Skipping YouTube enrichment for %s because it is disabled", title)
        return
    try:
        await _refresh_letsplay(session, game_id, title)
    except Exception:
        logger.exception("Unable to process YouTube letsplay for %s", title)


async def run_crawl(
    trigger: str = "schedule",
    *,
    day: date | None = None,
    batch_size: int | None = None,
    client: MetacriticClient | None = None,
    from_queue: bool = False,
) -> dict[str, Any]:
    """Process one crawl run and return a summary dictionary."""
    run_day = day or current_day()
    wanted = settings.crawl_batch_size if batch_size is None else batch_size
    if wanted <= 0:
        raise ValueError("batch_size must be positive")

    lock_connection = await engine.connect()
    try:
        acquired = await lock_connection.scalar(
            text("SELECT pg_try_advisory_lock(:lock_id)"), {"lock_id": _CRAWL_LOCK_ID}
        )
        await lock_connection.commit()
        if not acquired:
            logger.warning("Skipping %s crawl because another crawl is already running", trigger)
            return {
                "trigger": trigger,
                "day": run_day.isoformat(),
                "status": "skipped",
                "reason": "already_running",
            }

        async with SessionLocal() as session:
            abandoned = await repo.fail_abandoned_runs(session)
            if abandoned:
                logger.warning("Marked %d abandoned crawl run(s) as failed", abandoned)
                await session.commit()
            run = (
                await repo.claim_queued_run(session)
                if from_queue
                else await repo.start_run(session, trigger, run_day)
            )
            if run is None:
                return {"trigger": trigger, "day": run_day.isoformat(), "status": "idle"}
            await session.commit()
            run_id = run.id
            await repo.heartbeat_worker(
                session, _WORKER_ID, state="running", current_run_id=run_id
            )
            await _record_worker_log(
                session, "INFO", f"Started {trigger} crawl run #{run_id}", run_id
            )
            await session.commit()
            error: str | None = None
            try:
                async with client or MetacriticClient(proxy=settings.metacritic_proxy) as http:
                    selected = await _collect_candidates(session, http, run_day, wanted)
                    run.planned = len(selected)
                    await session.commit()
                    for slug, source in selected:
                        await repo.record_item(session, run_id, run_day, slug, source, "pending")
                        await session.commit()
                        try:
                            game_html = await http.game(slug)
                            page = parse_game_page(game_html)
                            lead_platform, user_reviews = await _fill_platform_scores(http, slug, page)
                            if not page.slug or not page.title:
                                raise ValueError(f"empty page for {slug}")
                            game_id, _ = await repo.upsert_game(session, page)
                            await session.commit()
                            embedding_hash = await repo.get_embedding_source_hash(session, game_id)
                            source_hash = embedding_source_hash(page)
                            if embedding_hash != source_hash:
                                vector, source_hash = await embed_game(page)
                                await repo.set_embedding(session, game_id, vector, source_hash)
                                await session.commit()
                            if lead_platform is not None:
                                if user_reviews is None:
                                    raise RuntimeError(
                                        f"User reviews unavailable on lead platform {lead_platform}"
                                    )
                                await _refresh_review_summary(session, game_id, user_reviews)
                                critic_html = await http.critic_reviews(slug, lead_platform)
                                critic_reviews = parse_reviews_page(critic_html, "critic")
                                await _refresh_review_summary(session, game_id, critic_reviews)
                            else:
                                logger.warning("No platform found for %s; review summaries were skipped", slug)
                            await _maybe_refresh_letsplay(session, game_id, page.title)
                            await repo.record_item(session, run_id, run_day, slug, source, "ok")
                            await _record_worker_log(
                                session, "INFO", f"Processed game: {page.title}", run_id
                            )
                            await session.commit()
                        except BlockedError as exc:
                            await session.rollback()
                            logger.error("Metacritic blocked the crawl while processing %s: %s", slug, exc)
                            await repo.record_item(session, run_id, run_day, slug, source, "failed", str(exc))
                            await _record_worker_log(
                                session, "ERROR", f"Metacritic blocked while processing {slug}: {exc}", run_id
                            )
                            await session.commit()
                            raise
                        except Exception as exc:
                            await session.rollback()
                            logger.exception("Failed to crawl game %s", slug)
                            await repo.record_item(session, run_id, run_day, slug, source, "failed", str(exc))
                            await _record_worker_log(
                                session, "ERROR", f"Failed to process game {slug}: {str(exc)[:500]}", run_id
                            )
                            await session.commit()
            except Exception as exc:
                await session.rollback()
                logger.exception("Crawl run %s failed before completion", run_id)
                error = str(exc)
            finally:
                run = await session.get(CrawlRun, run_id)
                if run is not None:
                    await repo.finish_run(session, run, error)
                    await repo.heartbeat_worker(
                        session, _WORKER_ID, state="idle", error=error
                    )
                    await _record_worker_log(
                        session,
                        "ERROR" if error else "INFO",
                        f"Finished crawl run #{run_id}: {run.status}, "
                        f"{run.ok_count}/{run.planned} successful, {run.failed_count} failed",
                        run_id,
                    )
                    await session.commit()

            return {
                "trigger": trigger,
                "day": run_day.isoformat(),
                "planned": run.planned,
                "ok_count": run.ok_count,
                "failed_count": run.failed_count,
                "status": run.status,
            }
    finally:
        try:
            await lock_connection.scalar(
                text("SELECT pg_advisory_unlock(:lock_id)"), {"lock_id": _CRAWL_LOCK_ID}
            )
            await lock_connection.commit()
        finally:
            await lock_connection.close()


def _scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=get_timezone())
    scheduler.add_job(
        run_crawl,
        "interval",
        minutes=settings.crawl_interval_minutes,
        id="metacritic-crawl-schedule",
        args=["schedule"],
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        run_crawl,
        "interval",
        seconds=settings.crawl_queue_poll_seconds,
        id="metacritic-manual-queue",
        args=["manual"],
        kwargs={"from_queue": True},
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        process_pending_review_summary,
        "interval",
        seconds=30,
        id="pending-review-summaries",
        max_instances=1,
        coalesce=True,
    )
    return scheduler


async def run_scheduler() -> None:
    scheduler = _scheduler()
    heartbeat_task = asyncio.create_task(_heartbeat_worker())
    async with SessionLocal() as session:
        await _record_worker_log(session, "INFO", f"Worker started: {_WORKER_ID}")
        await session.commit()
    scheduler.start()
    try:
        await asyncio.Event().wait()
    finally:
        scheduler.shutdown(wait=False)
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        async with SessionLocal() as session:
            await repo.heartbeat_worker(session, _WORKER_ID, state="stopped")
            await _record_worker_log(session, "INFO", f"Worker stopped: {_WORKER_ID}")
            await session.commit()


async def _heartbeat_worker() -> None:
    while True:
        try:
            async with SessionLocal() as session:
                await repo.heartbeat_worker(session, _WORKER_ID)
                await session.commit()
        except Exception:
            logger.exception("Unable to update worker heartbeat")
        await asyncio.sleep(_WORKER_HEARTBEAT_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Metacritic crawl worker")
    parser.add_argument("--once", action="store_true", help="run one crawl and exit")
    parser.add_argument("--manual", action="store_true", help="alias for one manual crawl")
    parser.add_argument("--batch-size", type=int, default=settings.crawl_batch_size)
    parser.add_argument("--day", type=str, default=None, help="ISO date, e.g. 2026-10-16")
    args = parser.parse_args()

    if args.once or args.manual:
        day = date.fromisoformat(args.day) if args.day else None
        result = asyncio.run(run_crawl(trigger="manual", day=day, batch_size=args.batch_size))
        print(result)
        return

    asyncio.run(run_scheduler())


if __name__ == "__main__":
    main()
