from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

engine = create_async_engine(settings.database_url, pool_size=5, max_overflow=5, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session():
    """FastAPI dependency."""
    async with SessionLocal() as session:
        yield session


__all__ = ["AsyncSession", "SessionLocal", "engine", "get_session"]
