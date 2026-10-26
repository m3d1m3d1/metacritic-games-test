"""Data access: idempotent upserts and crawl bookkeeping. Callers own the transaction."""
from __future__ import annotations

from datetime import date

from sqlalchemy import case, delete, func, literal_column, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import (
    CrawlItem,
    CrawlRun,
    CrawlState,
    Game,
    GamePlatform,
    PendingReviewSummary,
    ReviewSummary,
    WorkerLog,
    WorkerStatus,
)
from app.scraper.parsers import GamePage


def _iso_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value[:10]) if value else None
    except ValueError:
        return None


def _keep_old_scalar(stmt, column):
    """Take the new value, but never erase stored data with NULL (a transient parse miss)."""
    return func.coalesce(stmt.excluded[column.key], column)


def _keep_old_array(stmt, column):
    return case((func.cardinality(stmt.excluded[column.key]) > 0, stmt.excluded[column.key]), else_=column)


# --------------------------------------------------------------------------- games
async def upsert_game(session: AsyncSession, page: GamePage) -> tuple[int, bool]:
    """Insert the game or refresh it by slug, with all its platform scores.

    Returns (game_id, created). `page.platforms[*].userscore` must already be filled by the caller.
    Embedding columns are untouched here (see set_embedding).
    """
    if not page.slug or not page.title:
        raise ValueError("game page without slug/title")

    metas = [p.metascore for p in page.platforms if p.metascore is not None]
    users = [p.userscore for p in page.platforms if p.userscore is not None]
    values = dict(
        slug=page.slug,
        title=page.title,
        description=page.description,
        cover_url=page.cover_url,
        developers=page.developers,
        publishers=page.publishers,
        genres=page.genres,
        release_date=_iso_date(page.release_date),
        video_url=page.video_url,
        video_title=page.video_title,
        best_metascore=max(metas) if metas else page.metascore,
        best_userscore=max(users) if users else page.userscore,
        last_scraped_at=func.now(),
    )
    stmt = pg_insert(Game).values(**values)
    update = {
        c.key: (_keep_old_array(stmt, c) if c.key in ("developers", "publishers", "genres") else _keep_old_scalar(stmt, c))
        for c in (
            Game.description, Game.cover_url, Game.developers, Game.publishers, Game.genres, Game.release_date,
            Game.video_url, Game.video_title, Game.best_metascore, Game.best_userscore,
        )
    }
    update.update(title=stmt.excluded.title, last_scraped_at=func.now(), updated_at=func.now())
    update["description_ru"] = case(
        (
            stmt.excluded.description.is_not(None)
            & stmt.excluded.description.is_distinct_from(Game.description),
            None,
        ),
        else_=Game.description_ru,
    )
    stmt = stmt.on_conflict_do_update(index_elements=[Game.slug], set_=update).returning(
        Game.id, literal_column("(xmax = 0)").label("created")
    )
    game_id, created = (await session.execute(stmt)).one()

    slugs = [p.platform for p in page.platforms if p.platform]
    for p in page.platforms:
        if not p.platform:
            continue
        pstmt = pg_insert(GamePlatform).values(
            game_id=game_id, platform=p.platform, name=p.name, is_lead=p.is_lead,
            release_date=_iso_date(p.release_date), metascore=p.metascore, critic_count=p.critic_count,
            userscore=p.userscore, user_count=p.user_count,
        )
        await session.execute(
            pstmt.on_conflict_do_update(
                constraint="uq_game_platform",
                set_=dict(
                    name=pstmt.excluded.name, is_lead=pstmt.excluded.is_lead,
                    release_date=pstmt.excluded.release_date, metascore=pstmt.excluded.metascore,
                    critic_count=pstmt.excluded.critic_count,
                    userscore=_keep_old_scalar(pstmt, GamePlatform.userscore),
                    user_count=_keep_old_scalar(pstmt, GamePlatform.user_count),
                ),
            )
        )
    if slugs:  # platform disappeared from the page -> drop it
        await session.execute(
            delete(GamePlatform).where(GamePlatform.game_id == game_id, GamePlatform.platform.not_in(slugs))
        )
    return game_id, bool(created)


async def update_letsplay(
    session: AsyncSession,
    game_id: int,
    *,
    url: str,
    title: str,
    views: int,
    summary: str,
    source_hash: str,
) -> None:
    await session.execute(
        Game.__table__.update()
        .where(Game.id == game_id)
        .values(
            letsplay_url=url,
            letsplay_title=title,
            letsplay_views=views,
            letsplay_summary=summary,
            letsplay_summary_ru=case(
                (Game.letsplay_summary.is_distinct_from(summary), None),
                else_=Game.letsplay_summary_ru,
            ),
            letsplay_source_hash=source_hash,
            updated_at=func.now(),
        )
    )


async def get_letsplay_source_hash(session: AsyncSession, game_id: int) -> str | None:
    return await session.scalar(
        select(Game.letsplay_source_hash).where(Game.id == game_id)
    )


async def set_embedding(session: AsyncSession, game_id: int, vector: list[float], source_hash: str) -> None:
    await session.execute(
        Game.__table__.update().where(Game.id == game_id).values(embedding=vector, embedding_source_hash=source_hash)
    )


async def get_embedding_source_hash(session: AsyncSession, game_id: int) -> str | None:
    return await session.scalar(
        select(Game.embedding_source_hash).where(Game.id == game_id)
    )


async def upsert_summary(
    session: AsyncSession, game_id: int, audience: str, summary: str, likes: list[str], dislikes: list[str],
    reviews_used: int, source_hash: str, model: str,
) -> None:
    stmt = pg_insert(ReviewSummary).values(
        game_id=game_id, audience=audience, summary=summary, likes=likes, dislikes=dislikes,
        reviews_used=reviews_used, source_hash=source_hash, model=model,
    )
    await session.execute(
        stmt.on_conflict_do_update(
            constraint="uq_summary_game_audience",
            set_=dict(
                summary=stmt.excluded.summary, likes=stmt.excluded.likes, dislikes=stmt.excluded.dislikes,
                reviews_used=stmt.excluded.reviews_used, source_hash=stmt.excluded.source_hash,
                model=stmt.excluded.model, summary_ru=None, likes_ru=None, dislikes_ru=None,
                updated_at=func.now(),
            ),
        )
    )


async def save_game_translations(
    session: AsyncSession,
    game_id: int,
    *,
    description_ru: str | None,
    letsplay_summary_ru: str | None,
    summaries: dict[str, dict[str, str | list[str]]],
) -> None:
    await session.execute(
        update(Game)
        .where(Game.id == game_id)
        .values(
            description_ru=description_ru,
            letsplay_summary_ru=letsplay_summary_ru,
            updated_at=func.now(),
        )
    )
    for audience, translation in summaries.items():
        await session.execute(
            update(ReviewSummary)
            .where(
                ReviewSummary.game_id == game_id,
                ReviewSummary.audience == audience,
            )
            .values(
                summary_ru=translation["summary"],
                likes_ru=translation["likes"],
                dislikes_ru=translation["dislikes"],
                updated_at=func.now(),
            )
        )


async def get_summary_source_hash(session: AsyncSession, game_id: int, audience: str) -> str | None:
    return await session.scalar(
        select(ReviewSummary.source_hash).where(
            ReviewSummary.game_id == game_id, ReviewSummary.audience == audience
        )
    )


async def delete_summary(session: AsyncSession, game_id: int, audience: str) -> None:
    await session.execute(
        delete(ReviewSummary).where(
            ReviewSummary.game_id == game_id, ReviewSummary.audience == audience
        )
    )


async def enqueue_pending_review_summary(
    session: AsyncSession,
    game_id: int,
    audience: str,
    platform: str,
    review_count: int,
    reviews: list[dict],
    source_hash: str,
) -> None:
    stmt = pg_insert(PendingReviewSummary).values(
        game_id=game_id,
        audience=audience,
        platform=platform,
        review_count=review_count,
        reviews=reviews,
        source_hash=source_hash,
        attempts=0,
        next_attempt_at=func.now(),
        last_error=None,
    )
    await session.execute(
        stmt.on_conflict_do_update(
            constraint="uq_pending_summary_game_audience",
            set_=dict(
                platform=stmt.excluded.platform,
                review_count=stmt.excluded.review_count,
                reviews=stmt.excluded.reviews,
                source_hash=stmt.excluded.source_hash,
                attempts=0,
                next_attempt_at=func.now(),
                last_error=None,
                updated_at=func.now(),
            ),
        )
    )


async def get_pending_review_summary_source_hash(
    session: AsyncSession, game_id: int, audience: str
) -> str | None:
    return await session.scalar(
        select(PendingReviewSummary.source_hash).where(
            PendingReviewSummary.game_id == game_id,
            PendingReviewSummary.audience == audience,
        )
    )


async def due_pending_review_summary(
    session: AsyncSession,
) -> PendingReviewSummary | None:
    return await session.scalar(
        select(PendingReviewSummary)
        .where(PendingReviewSummary.next_attempt_at <= func.now())
        .order_by(PendingReviewSummary.next_attempt_at, PendingReviewSummary.id)
        .with_for_update(skip_locked=True)
        .limit(1)
    )


async def defer_pending_review_summary(
    session: AsyncSession,
    pending: PendingReviewSummary,
    error: str,
    retry_after_seconds: int | None = None,
) -> None:
    delay_seconds = min(60 * 2 ** min(pending.attempts, 6), 3600)
    if retry_after_seconds is not None:
        delay_seconds = max(delay_seconds, min(max(retry_after_seconds, 0), 86400))
    pending.attempts += 1
    pending.next_attempt_at = func.now() + text(f"interval '{delay_seconds} seconds'")
    pending.last_error = error[:2000]
    pending.updated_at = func.now()


async def defer_all_pending_review_summaries(
    session: AsyncSession, error: str, delay_seconds: int
) -> None:
    delay_seconds = min(max(delay_seconds, 0), 86400)
    await session.execute(
        update(PendingReviewSummary)
        .values(
            next_attempt_at=func.greatest(
                PendingReviewSummary.next_attempt_at,
                func.now() + text(f"interval '{delay_seconds} seconds'"),
            ),
            last_error=error[:2000],
            updated_at=func.now(),
        )
    )


async def delete_pending_review_summary(
    session: AsyncSession, pending_id: int
) -> None:
    await session.execute(
        delete(PendingReviewSummary).where(PendingReviewSummary.id == pending_id)
    )


async def delete_pending_review_summary_for_game(
    session: AsyncSession, game_id: int, audience: str
) -> None:
    await session.execute(
        delete(PendingReviewSummary).where(
            PendingReviewSummary.game_id == game_id,
            PendingReviewSummary.audience == audience,
        )
    )


async def record_worker_log(
    session: AsyncSession, worker_id: str, level: str, message: str, run_id: int | None = None
) -> None:
    session.add(
        WorkerLog(worker_id=worker_id, level=level, message=message[:2000], run_id=run_id)
    )
    await session.execute(
        delete(WorkerLog).where(
            WorkerLog.created_at < func.now() - text("interval '30 days'")
        )
    )


async def recent_worker_logs(
    session: AsyncSession, limit: int = 80
) -> list[WorkerLog]:
    result = await session.scalars(
        select(WorkerLog).order_by(WorkerLog.id.desc()).limit(limit)
    )
    return list(reversed(result.all()))


async def pending_review_summary_stats(
    session: AsyncSession,
) -> dict[str, int | str | None]:
    row = (
        await session.execute(
            select(
                func.count(PendingReviewSummary.id),
                func.count(PendingReviewSummary.id).filter(
                    PendingReviewSummary.next_attempt_at <= func.now()
                ),
                func.count(PendingReviewSummary.id).filter(
                    PendingReviewSummary.last_error.is_not(None)
                ),
            )
        )
    ).one()
    latest_error = await session.scalar(
        select(PendingReviewSummary.last_error)
        .where(PendingReviewSummary.last_error.is_not(None))
        .order_by(PendingReviewSummary.updated_at.desc())
        .limit(1)
    )
    return {
        "pending": row[0],
        "due": row[1],
        "retrying": row[2],
        "last_error": latest_error,
    }


def _game_filters(query: str | None, platform: str | None) -> list:
    filters = []
    if query:
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        filters.append(Game.title.ilike(f"%{escaped}%", escape="\\"))
    if platform:
        filters.append(Game.platforms.any(GamePlatform.platform == platform))
    return filters


def _game_order(sort: str, order: str):
    sort_columns = {
        "metascore": Game.best_metascore,
        "userscore": Game.best_userscore,
        "title": Game.title,
        "release_date": Game.release_date,
    }
    sort_column = sort_columns[sort]
    primary_order = sort_column.asc() if order == "asc" else sort_column.desc()
    return [primary_order.nulls_last(), Game.title.asc(), Game.id.asc()]


async def list_games(
    session: AsyncSession,
    *,
    query: str | None = None,
    platform: str | None = None,
    sort: str = "metascore",
    order: str = "desc",
    limit: int = 24,
    offset: int = 0,
) -> tuple[list[Game], int]:
    filters = _game_filters(query, platform)

    count = await session.scalar(select(func.count(Game.id)).where(*filters))
    result = await session.execute(
        select(Game)
        .where(*filters)
        .options(selectinload(Game.platforms))
        .order_by(*_game_order(sort, order))
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().unique()), count or 0


async def game_neighbors(
    session: AsyncSession,
    slug: str,
    *,
    query: str | None = None,
    platform: str | None = None,
    sort: str = "metascore",
    order: str = "desc",
) -> tuple[str | None, str | None]:
    filters = _game_filters(query, platform)
    ordered_games = (
        select(
            Game.slug.label("slug"),
            func.row_number()
            .over(order_by=_game_order(sort, order))
            .label("position"),
        )
        .where(*filters)
        .subquery()
    )
    position = await session.scalar(
        select(ordered_games.c.position).where(ordered_games.c.slug == slug)
    )
    if position is None:
        return None, None

    previous_slug = await session.scalar(
        select(ordered_games.c.slug).where(ordered_games.c.position == position - 1)
    )
    next_slug = await session.scalar(
        select(ordered_games.c.slug).where(ordered_games.c.position == position + 1)
    )
    return previous_slug, next_slug


async def available_platforms(session: AsyncSession) -> list[tuple[str, str | None]]:
    result = await session.execute(
        select(GamePlatform.platform, GamePlatform.name)
        .group_by(GamePlatform.platform, GamePlatform.name)
        .order_by(GamePlatform.name, GamePlatform.platform)
    )
    return list(result.all())


async def game_by_slug(session: AsyncSession, slug: str) -> Game | None:
    result = await session.execute(
        select(Game)
        .where(Game.slug == slug)
        .options(selectinload(Game.platforms), selectinload(Game.summaries))
    )
    return result.scalar_one_or_none()


async def similar_games(session: AsyncSession, game: Game, limit: int = 6) -> list[Game]:
    query = select(Game).where(Game.id != game.id).options(selectinload(Game.platforms))
    if game.embedding is not None:
        query = (
            query.where(Game.embedding.is_not(None))
            .order_by(Game.embedding.cosine_distance(game.embedding).asc(), Game.title.asc())
        )
    elif game.genres:
        query = (
            query.where(Game.genres.op("&&")(game.genres))
            .order_by(Game.best_metascore.desc().nulls_last(), Game.title.asc())
        )
    else:
        return []
    result = await session.execute(query.limit(limit))
    return list(result.scalars().unique())


# --------------------------------------------------------------------------- crawl bookkeeping
async def processed_slugs(session: AsyncSession, day: date, max_failures: int = 2) -> set[str]:
    """Return successes and slugs that exhausted their daily failure allowance."""
    if max_failures < 1:
        raise ValueError("max_failures must be positive")
    successes = await session.scalars(
        select(CrawlItem.slug).where(CrawlItem.day == day, CrawlItem.status == "ok")
    )
    exhausted = await session.scalars(
        select(CrawlItem.slug)
        .where(CrawlItem.day == day, CrawlItem.status == "failed")
        .group_by(CrawlItem.slug)
        .having(func.count(CrawlItem.id) >= max_failures)
    )
    return set(successes) | set(exhausted)


async def get_browse_page(session: AsyncSession, day: date) -> int:
    page = await session.scalar(select(CrawlState.browse_page).where(CrawlState.day == day))
    return page or 1


async def set_browse_page(session: AsyncSession, day: date, page: int) -> None:
    stmt = pg_insert(CrawlState).values(day=day, browse_page=page)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[CrawlState.day], set_=dict(browse_page=page, updated_at=func.now())
        )
    )


async def start_run(session: AsyncSession, trigger: str, day: date) -> CrawlRun:
    run = CrawlRun(trigger=trigger, day=day, status="running")
    session.add(run)
    await session.flush()
    return run


async def record_item(
    session: AsyncSession, run_id: int, day: date, slug: str, source: str, status: str, error: str | None = None
) -> None:
    stmt = pg_insert(CrawlItem).values(
        run_id=run_id, day=day, slug=slug, source=source, status=status, error=error,
        processed_at=func.now() if status in ("ok", "failed") else None,
    )
    await session.execute(
        stmt.on_conflict_do_update(
            constraint="uq_crawl_item_run_slug",
            set_=dict(status=stmt.excluded.status, error=stmt.excluded.error, processed_at=stmt.excluded.processed_at),
        )
    )


async def finish_run(session: AsyncSession, run: CrawlRun, error: str | None = None) -> None:
    counts = dict(
        (await session.execute(
            select(CrawlItem.status, func.count()).where(CrawlItem.run_id == run.id).group_by(CrawlItem.status)
        )).all()
    )
    run.ok_count, run.failed_count = counts.get("ok", 0), counts.get("failed", 0)
    run.error = error
    run.finished_at = func.now()
    run.status = "failed" if error else ("partial" if run.failed_count else "ok")
    await session.flush()


async def fail_abandoned_runs(session: AsyncSession) -> int:
    """Mark runs left active by a worker process that exited unexpectedly."""
    runs = list(
        (
            await session.scalars(
                select(CrawlRun).where(CrawlRun.status == "running").with_for_update()
            )
        ).all()
    )
    if not runs:
        return 0

    run_ids = [run.id for run in runs]
    await session.execute(
        CrawlItem.__table__.update()
        .where(CrawlItem.run_id.in_(run_ids), CrawlItem.status == "pending")
        .values(
            status="failed",
            error="Worker stopped before completing this item",
            processed_at=func.now(),
        )
    )
    for run in runs:
        await finish_run(session, run, "Worker stopped before completing this run")
    return len(runs)


async def heartbeat_worker(
    session: AsyncSession,
    worker_id: str,
    *,
    state: str | None = None,
    current_run_id: int | None = None,
    error: str | None = None,
) -> None:
    values = {
        "worker_id": worker_id,
        "state": state or "idle",
        "last_seen_at": func.now(),
        "current_run_id": current_run_id,
        "error": error,
    }
    stmt = pg_insert(WorkerStatus).values(**values)
    updates = {"last_seen_at": func.now()}
    if state is not None:
        updates.update(state=state, current_run_id=current_run_id, error=error)
    await session.execute(
        stmt.on_conflict_do_update(index_elements=[WorkerStatus.worker_id], set_=updates)
    )


async def worker_statuses(session: AsyncSession) -> list[WorkerStatus]:
    return list(
        (
            await session.scalars(
                select(WorkerStatus).order_by(WorkerStatus.worker_id)
            )
        ).all()
    )


async def recent_runs(session: AsyncSession, limit: int = 20) -> list[CrawlRun]:
    """Return recent runs and their items for the monitoring page."""
    result = await session.execute(
        select(CrawlRun)
        .options(selectinload(CrawlRun.items))
        .order_by(CrawlRun.started_at.desc(), CrawlRun.id.desc())
        .limit(max(1, min(limit, 100)))
    )
    return list(result.scalars().unique())


async def enqueue_run(session: AsyncSession, day: date) -> CrawlRun:
    """Persist a manual crawl request for the worker to claim."""
    run = CrawlRun(trigger="manual", status="queued", day=day)
    session.add(run)
    await session.flush()
    return run


async def claim_queued_run(session: AsyncSession) -> CrawlRun | None:
    """Atomically claim the oldest queued run, if one is waiting."""
    run = await session.scalar(
        select(CrawlRun)
        .where(CrawlRun.status == "queued")
        .order_by(CrawlRun.id)
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    if run is not None:
        run.status = "running"
        run.started_at = func.now()
        run.finished_at = None
        await session.flush()
    return run
