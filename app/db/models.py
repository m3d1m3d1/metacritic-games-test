from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Size of the embedding vector. Tied to the embedding model (BAAI/bge-small-en-v1.5 = 384).
# Changing the model to one with another size needs a new migration.
EMBEDDING_DIM = 384


class Base(DeclarativeBase):
    pass


_NOW = func.now()


class Game(Base):
    __tablename__ = "games"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(255), unique=True)
    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[str | None] = mapped_column(Text)
    description_ru: Mapped[str | None] = mapped_column(Text)
    cover_url: Mapped[str | None] = mapped_column(Text)
    developers: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'"))
    publishers: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'"))
    genres: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'"))
    release_date: Mapped[date | None] = mapped_column(Date)
    video_url: Mapped[str | None] = mapped_column(Text)
    video_title: Mapped[str | None] = mapped_column(Text)
    letsplay_url: Mapped[str | None] = mapped_column(Text)
    letsplay_title: Mapped[str | None] = mapped_column(Text)
    letsplay_views: Mapped[int | None] = mapped_column(Integer)
    letsplay_summary: Mapped[str | None] = mapped_column(Text)
    letsplay_summary_ru: Mapped[str | None] = mapped_column(Text)
    letsplay_source_hash: Mapped[str | None] = mapped_column(String(64))

    # Denormalised for sorting/listing: best score over all platforms (NULL = tbd).
    best_metascore: Mapped[int | None] = mapped_column(Integer)
    best_userscore: Mapped[Decimal | None] = mapped_column(Numeric(3, 1))

    # Similar games: embedding of title + genres + description. Recomputed only when the
    # text changes (embedding_source_hash).
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    embedding_source_hash: Mapped[str | None] = mapped_column(String(64))

    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_NOW)
    last_scraped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_NOW)

    platforms: Mapped[list[GamePlatform]] = relationship(
        back_populates="game", cascade="all, delete-orphan", order_by="GamePlatform.id"
    )
    summaries: Mapped[list[ReviewSummary]] = relationship(back_populates="game", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_games_best_metascore", "best_metascore"),
        Index("ix_games_best_userscore", "best_userscore"),
        Index("ix_games_release_date", "release_date"),
        Index(
            "ix_games_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class GamePlatform(Base):
    """Per-platform scores. NULL score = tbd (critics: <4 reviews; users: no ratings)."""

    __tablename__ = "game_platforms"

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"))
    platform: Mapped[str] = mapped_column(String(64))  # url slug, e.g. "xbox-series-x"
    name: Mapped[str | None] = mapped_column(String(100))
    is_lead: Mapped[bool] = mapped_column(server_default=text("false"))
    release_date: Mapped[date | None] = mapped_column(Date)
    metascore: Mapped[int | None] = mapped_column(Integer)
    critic_count: Mapped[int | None] = mapped_column(Integer)
    userscore: Mapped[Decimal | None] = mapped_column(Numeric(3, 1))
    user_count: Mapped[int | None] = mapped_column(Integer)

    game: Mapped[Game] = relationship(back_populates="platforms")

    __table_args__ = (
        UniqueConstraint("game_id", "platform", name="uq_game_platform"),
        Index("ix_game_platforms_platform", "platform"),
    )


class ReviewSummary(Base):
    """LLM summary of reviews, one row per (game, audience). Refreshed on re-crawl only when
    the reviews it was built from changed (source_hash)."""

    __tablename__ = "review_summaries"

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"))
    audience: Mapped[str] = mapped_column(String(10))  # "critic" | "user"
    summary: Mapped[str] = mapped_column(Text)
    summary_ru: Mapped[str | None] = mapped_column(Text)
    likes: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'"))
    likes_ru: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    dislikes: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'"))
    dislikes_ru: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    reviews_used: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    source_hash: Mapped[str | None] = mapped_column(String(64))
    model: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_NOW)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_NOW)

    game: Mapped[Game] = relationship(back_populates="summaries")

    __table_args__ = (
        UniqueConstraint("game_id", "audience", name="uq_summary_game_audience"),
        CheckConstraint("audience IN ('critic', 'user')", name="ck_summary_audience"),
    )


class PendingReviewSummary(Base):
    __tablename__ = "pending_review_summaries"

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"))
    audience: Mapped[str] = mapped_column(String(10))
    platform: Mapped[str] = mapped_column(String(64))
    review_count: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    reviews: Mapped[list[dict]] = mapped_column(JSONB)
    source_hash: Mapped[str] = mapped_column(String(64))
    attempts: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=_NOW
    )
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_NOW)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_NOW)

    __table_args__ = (
        UniqueConstraint("game_id", "audience", name="uq_pending_summary_game_audience"),
        CheckConstraint("audience IN ('critic', 'user')", name="ck_pending_summary_audience"),
        Index("ix_pending_review_summaries_next_attempt", "next_attempt_at"),
    )


class CrawlRun(Base):
    """One execution of the hourly job (scheduled or manual). Also feeds the monitoring page."""

    __tablename__ = "crawl_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    trigger: Mapped[str] = mapped_column(String(16))  # "schedule" | "manual"
    status: Mapped[str] = mapped_column(String(16), server_default="running")  # queued|running|ok|partial|failed
    day: Mapped[date] = mapped_column(Date)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_NOW)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    planned: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    ok_count: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    failed_count: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    error: Mapped[str | None] = mapped_column(Text)

    items: Mapped[list[CrawlItem]] = relationship(back_populates="run", cascade="all, delete-orphan")


class CrawlItem(Base):
    """A game processed in a run. Rows with status='ok' for today's `day` are the
    'already processed today' set that the selection logic skips."""

    __tablename__ = "crawl_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("crawl_runs.id", ondelete="CASCADE"))
    day: Mapped[date] = mapped_column(Date)
    slug: Mapped[str] = mapped_column(String(255))
    source: Mapped[str] = mapped_column(String(32))  # "home" | "browse:<page>"
    status: Mapped[str] = mapped_column(String(16), server_default="pending")  # pending|ok|failed
    error: Mapped[str | None] = mapped_column(Text)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    run: Mapped[CrawlRun] = relationship(back_populates="items")

    __table_args__ = (
        UniqueConstraint("run_id", "slug", name="uq_crawl_item_run_slug"),
        Index("ix_crawl_items_day_slug", "day", "slug"),
    )


class CrawlState(Base):
    """Per-day cursor: which /browse page the next run starts from. Resets implicitly each day
    (no row for the new day = start from the home page block, then page 1)."""

    __tablename__ = "crawl_state"

    day: Mapped[date] = mapped_column(Date, primary_key=True)
    browse_page: Mapped[int] = mapped_column(Integer, server_default=text("1"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_NOW)


class WorkerStatus(Base):
    __tablename__ = "worker_status"

    worker_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    state: Mapped[str] = mapped_column(String(16), server_default="idle")
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_NOW)
    current_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("crawl_runs.id", ondelete="SET NULL")
    )
    error: Mapped[str | None] = mapped_column(Text)


class WorkerLog(Base):
    __tablename__ = "worker_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    worker_id: Mapped[str] = mapped_column(String(255))
    level: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(Text)
    run_id: Mapped[int | None] = mapped_column(
        ForeignKey("crawl_runs.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_NOW)

    __table_args__ = (Index("ix_worker_logs_created_at", "created_at"),)
